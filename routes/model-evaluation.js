/**
 * 训练后自动评估编排层（/api/model-training/jobs/:id/evaluate）。
 *
 * 设计要点（对标 LLaMA-Factory / lm-evaluation-harness / kev.benchmark）：
 *  1. 评估与训练解耦：训练状态只由训练进程决定，评估结果单独写 job.evaluation；
 *     评估失败不会把 completed 改回 failed，也不会丢掉上一次成功结果。
 *  2. 引擎可插拔，统一「预测结果 → 逐样本行 → 指标」契约：
 *     - kev：调用官方 `python -m kev.benchmark --run ... --data ... --out ...`，
 *       解析 rows.json（逐问题概率 + label）与 report.json（官方 acc/ECE/Brier/NLL），
 *       平台指标与官方 acc 交叉校验。
 *     - llamafactory：do_predict + predict_with_generate 生成 generated_predictions.jsonl，
 *       解析 label/predict 后按 SQuAD/ROUGE-L 口径打分。
 *     - import：外部评测 / 离线推理的 predictions.jsonl 直接导入，保证不绑定单一引擎。
 *  3. 评估集必须与训练集分开：内容指纹 + 重复率检测，命中即给泄漏警告（不覆盖数字）。
 *  4. 逐样本结果留痕（对标 lm-eval --log_samples），报告里能查到每个样本的预测。
 *
 * 本模块不含任何 HTTP / 文件系统硬编码路径：所有 IO 由调用方通过 io 注入，便于单测。
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const STORE = require('./json-store');
const M = require('./eval-metrics');

const TASKS = new Map();      // taskId -> { id, jobId, state, progress, ... }
const TASK_TTL_MS = 6 * 60 * 60 * 1000;
const MAX_TASKS = 60;

// ── 任务持久化（对标 dataset-synthesis 的 synth-tasks 落盘模式）──────────
// 历史缺陷：TASKS 只在内存——服务重启时运行中的评估任务直接消失，
// job.evaluation 永远停在 running。落盘后：重启恢复历史记录（可查），
// running 中的任务标记 interrupted（后台协程已不存在，不假装还在跑）。
const EVAL_TASKS_DIR = path.join(__dirname, '..', 'data', 'model-training', 'eval-tasks');
try { fs.mkdirSync(EVAL_TASKS_DIR, { recursive: true }); } catch (e) {}
const EVAL_TASK_ID_RE = /^evaltask-[a-z0-9]+$/;
const EVAL_FILE_TTL_MS = 7 * 24 * 60 * 60 * 1000;

function evalTaskFile(id) { return path.join(EVAL_TASKS_DIR, id + '.json'); }
function isEvalTaskId(id) { return EVAL_TASK_ID_RE.test(String(id == null ? '' : id)); }

// 落盘只保留可序列化字段（密钥/内部引用一律不写）
function serializeEvalTask(t) {
  const o = {};
  for (const [k, v] of Object.entries(t)) {
    if (k.startsWith('_')) continue;
    if (/apiKey|secret|token|password/i.test(k)) continue;
    if (v === undefined) continue;
    try { JSON.stringify(v); o[k] = v; } catch (e) { /* 不可序列化字段跳过 */ }
  }
  return o;
}
function persistEvalTask(t) {
  if (!isEvalTaskId(t.id)) return;
  try { STORE.writeJsonSync(evalTaskFile(t.id), serializeEvalTask(t)); } catch (e) { /* 落盘失败不阻断评估 */ }
}
function readEvalTaskFile(id) {
  if (!isEvalTaskId(id)) return null;
  try {
    const o = STORE.readJson(evalTaskFile(id), null);
    return o && o.id === id ? o : null;
  } catch (e) { return null; }
}

// 启动时恢复：running/pending 的任务标记 interrupted（后台协程已随旧进程死亡）
function loadPersistedEvalTasks() {
  let files = [];
  try { files = fs.readdirSync(EVAL_TASKS_DIR).filter(f => /^evaltask-[a-z0-9]+\.json$/.test(f)); } catch (e) { return; }
  const cutoff = Date.now() - EVAL_FILE_TTL_MS;
  for (const f of files) {
    const id = f.slice(0, -5);
    const o = readEvalTaskFile(id);
    if (!o) continue;
    if (o.finishedAt && o.finishedAt < cutoff) {
      try { fs.unlinkSync(evalTaskFile(id)); } catch (e) {}
      continue;
    }
    if (o.state === 'running' || o.state === 'queued' || o.state === 'cancelling') {
      o.state = 'interrupted';
      o.error = '服务重启导致评估中断：后台推理协程已不存在，请重新发起评估';
      o.finishedAt = Date.now();
      STORE.writeJsonSync(evalTaskFile(id), o);
    }
    if (TASKS.size < MAX_TASKS) TASKS.set(id, o);
  }
}
loadPersistedEvalTasks();


function newId(prefix) {
  return prefix + '-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}
function now() { return Date.now(); }

function pruneTasks() {
  const cutoff = now() - TASK_TTL_MS;
  for (const [k, v] of TASKS) if (v.finishedAt && v.finishedAt < cutoff) TASKS.delete(k);
  // 防止异常路径无限堆积（保留最新的 MAX_TASKS 个）
  if (TASKS.size > MAX_TASKS) {
    const keys = Array.from(TASKS.keys());
    for (const k of keys.slice(0, TASKS.size - MAX_TASKS)) TASKS.delete(k);
  }
}

// ── 评估集解析 ──────────────────────────────────────────────────────────

/** 支持 .json array / .jsonl / 带 samples 字段的对象；不做任何格式转换，保留原始样本 */
function parseSamples(fileName, content) {
  const text = String(content == null ? '' : content).replace(/^\uFEFF/, '');
  const isJsonl = /\.jsonl$/i.test(String(fileName || '')) || /^[{\[][^\n]*\n[{\[]/.test(text.trim());
  if (isJsonl) {
    const rows = [];
    text.split(/\r?\n/).forEach(function (line, i) {
      if (!line.trim()) return;
      try { rows.push(JSON.parse(line)); }
      catch (e) { throw new Error('评估集第 ' + (i + 1) + ' 行不是合法 JSON：' + e.message); }
    });
    return rows;
  }
  let v = JSON.parse(text);
  if (v && !Array.isArray(v) && Array.isArray(v.samples)) v = v.samples;
  if (!Array.isArray(v)) throw new Error('评估集必须是数组或 JSONL');
  return v;
}

function strLabel(v) {
  if (v == null) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** 从样本里抽取「逐预测单元」的标准答案。返回 { kind, rows } */
function extractGold(samples) {
  const rows = [];
  let kind = null;
  (samples || []).forEach(function (s, i) {
    if (s && s.state && s.questions && typeof s.questions === 'object') {
      kind = 'classification';
      Object.keys(s.questions).forEach(function (qid) {
        const q = s.questions[qid] || {};
        // options：该问题的选项标签数组（下标即 rlcd rows 的 gold_index/pred_index 语义）。
        // choice: options 数组或 criteria 键序；noul: [false, true]；score: 档位标签
        let options = null;
        if (q.type === 'noul') options = ['false', 'true'];
        else if (Array.isArray(q.options) && q.options.length) options = q.options.map(strLabel);
        else if (q.options && typeof q.options === 'object') options = Object.keys(q.options);
        else if (q.criteria && typeof q.criteria === 'object') {
          const keys = Object.keys(q.criteria);
          // criteria 可能是 {标签: 描述} 也可能是 {档位: 描述}——都是选项名
          options = keys.map(strLabel);
        }
        rows.push({ id: i + ':' + qid, gold: strLabel(q.label), type: q.type || null,
          options: options, group: 'question', input: s.state, question: qid });
      });
      return;
    }
    if (Array.isArray(s && s.messages)) {
      const last = s.messages.filter(function (m) { return m && m.role === 'assistant'; }).pop();
      kind = kind || 'generation';
      rows.push({ id: String(i), gold: strLabel(last && (last.content != null ? last.content : last.text)), type: 'generation', input: s.messages });
      return;
    }
    if (Array.isArray(s && s.conversations)) {
      const last = s.conversations.filter(function (m) { return m && /^(gpt|assistant)$/i.test(m.from || m.role || ''); }).pop();
      kind = kind || 'generation';
      rows.push({ id: String(i), gold: strLabel(last && last.value), type: 'generation', input: s.conversations });
      return;
    }
    if (s && (s.label != null || s.target != null) && (s.text != null || s.input != null || s.question != null)) {
      kind = 'classification';
      rows.push({ id: String(i), gold: strLabel(s.label != null ? s.label : s.target),
        type: 'label', input: s.text != null ? s.text : (s.input != null ? s.input : s.question) });
      return;
    }
    const out = s && (s.output != null ? s.output : (s.answer != null ? s.answer : (s.completion != null ? s.completion : s.response)));
    kind = kind || 'generation';
    const inp = s && (s.instruction != null ? s.instruction : (s.question != null ? s.question : (s.prompt != null ? s.prompt : s.input)));
    rows.push({ id: String(i), gold: strLabel(out), type: 'generation', input: inp });
  });
  return { kind: kind || 'generation', rows: rows };
}

// ── 引擎输出解析 ────────────────────────────────────────────────────────

function parseJsonSafe(text, fallback) {
  try { return JSON.parse(text); } catch (e) { return fallback; }
}

function parseJsonLines(text) {
  return String(text || '').split(/\r?\n/).filter(function (l) { return l.trim(); })
    .map(function (l) { return parseJsonSafe(l, null); }).filter(Boolean);
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (Number(arr[i]) > Number(arr[best])) best = i;
  return best;
}

/** kev rows.json → 统一行；同时保留官方报告用于交叉校验与校准指标 */
function parseKevOutput(rowsText, reportText) {
  const raw = parseJsonSafe(rowsText, []);
  const list = Array.isArray(raw) ? raw : (raw.rows || []);
  const rows = list.map(function (r, i) {
    const keys = (r.keys || []).map(strLabel);
    const p = Array.isArray(r.p) ? r.p.map(Number) : [];
    const gold = Number.isInteger(r.label) && keys[r.label] != null ? keys[r.label] : strLabel(r.label);
    const pred = p.length === keys.length && p.length ? keys[argmax(p)] : null;
    return { id: String(r.id != null ? r.id : i), gold: gold, prediction: pred,
      type: r.type || null, task: r.task || null, confidence: p.length ? Number(p[argmax(p)]) : null };
  });
  const report = parseJsonSafe(reportText, null);
  const clean = report && report.clean && typeof report.clean === 'object' ? report.clean : null;
  return { rows: rows, official: clean ? {
    accuracy: clean.acc, n: clean.n, ece: clean.ece, brier: clean.brier,
    nll: clean.nll, meanConfidence: clean.mean_conf
  } : null, coverage: report && report.coverage || null, report: report };
}

/**
 * rlcd-lite rows.json（records_to_rows 输出：每条 = 一个字段一次预测）
 *   {example, field, gold_index, pred_index, correct, probs[], logits[], option_index[], conf_softmax, conf_stated}
 * → 统一行。taskType 保持 classification（多问题并行打分）。
 * 汇总数字（accuracy/ECE/Brier）来自 rlcd report.json 的 signals——与逐行自算交叉校验。
 *
 * 语义还原（对照实验可比性关键）：rows.json 只有数值索引，gold/prediction 必须还原成
 * 选项标签（与 kev 引擎的 keys[label] 同一语义空间），否则 perClass 分组会碎成
 * "例子:字段:索引" 独键（每键 n=1，F1 全 0/1），macro-F1 与 kev 臂完全不可比。
 * 还原数据源 = goldRows（extractGold 产物，行 id = "example:field"，带 question 名
 * 与样本引用顺序）。选项顺序：choice/noul 字段按样本 questions[field].options（或
 * criteria 键序）；kev 桥接时 noul 的选项即 [false, true]，label 是 0/1 下标。
 * type 对齐 kev 分类学（choice/noul/score，来自 goldRows 行），字段名放 question。
 */
function parseRlcdOutput(rowsText, reportText, goldRows) {
  const raw = parseJsonSafe(rowsText, []);
  const list = Array.isArray(raw) ? raw : (raw.rows || []);
  // goldRows 行 id = "example:field"（extractGold 的 decision 分支），借它拿选项标签与问题类型
  const goldByKey = {};
  (goldRows || []).forEach(function (g) { if (g && g.id != null) goldByKey[String(g.id)] = g; });
  const rows = list.map(function (r, i) {
    const p = Array.isArray(r.probs) ? r.probs.map(Number) : [];
    const key = r.example != null ? r.example + ':' + r.field : String(i);
    const gRow = goldByKey[key] || {};
    // 选项标签：goldRows 行带 options（评估编排时从样本 questions 提取，见 runEvaluation）；
    // 拿不到时退回索引字符串（保证 unique，不与别的样本串键）
    const opts = Array.isArray(gRow.options) && gRow.options.length ? gRow.options : null;
    const gold = opts ? (opts[r.gold_index] != null ? String(opts[r.gold_index]) : 'idx' + r.gold_index)
      : key + ':' + r.gold_index;
    const pred = r.pred_index == null ? null
      : (opts ? (opts[r.pred_index] != null ? String(opts[r.pred_index]) : 'idx' + r.pred_index) : key + ':' + r.pred_index);
    return { id: key,
      gold: gold, prediction: pred,
      // type 用 kev 分类学（choice/noul/score）；字段名放 question 与 kev 的 question 对齐
      type: gRow.type || null, question: r.field || gRow.question || null,
      task: gRow.task || null, correct: !!r.correct,
      confidence: p.length && r.pred_index != null ? Number(p[r.pred_index])
        : (r.conf_softmax != null ? Number(r.conf_softmax) : null),
      statedConfidence: r.conf_stated != null ? Number(r.conf_stated) : null };
  });
  const report = parseJsonSafe(reportText, null);
  return { rows: rows,
    official: report && Number.isFinite(report.accuracy) ? {
      accuracy: report.accuracy, n: report.n_records,
      // rlcd 的 signals 是多信号结构；softmax 为主信号
      ece: report.signals && report.signals.softmax ? report.signals.softmax.ece : null,
      brier: report.signals && report.signals.softmax ? report.signals.softmax.brier : null,
      temperatures: report.temperatures || null
    } : null,
    report: report };
}

/** LLaMA-Factory generated_predictions.jsonl（每行 prompt/label/predict）→ 统一行 */
function parseLfOutput(text, goldRows) {
  const lines = parseJsonLines(text);
  const rows = goldRows.map(function (g, i) {
    const p = lines[i] || null;
    return { id: g.id, gold: g.gold, prediction: p ? strLabel(p.predict != null ? p.predict : p.prediction) : null,
      type: 'generation', referenceLabel: p ? strLabel(p.label) : null };
  });
  return { rows: rows, official: null, parsedLines: lines.length };
}

/** 通用导入：JSONL / 数组，字段 id / prediction|predict / gold（可选，缺省用评估集 gold） */
function parseImported(text, goldRows) {
  // 整体是 JSON 数组时优先用它；否则才按 JSONL 逐行解析。
  // 历史缺陷：`[{...},{...}]` 被 parseJsonLines 当成一行，整体数组被当成单条预测 → 全部落空。
  const whole = parseJsonSafe(String(text == null ? '' : text), null);
  let list;
  if (Array.isArray(whole)) list = whole;
  else if (whole && Array.isArray(whole.predictions)) list = whole.predictions;
  else list = parseJsonLines(text);
  const byId = new Map();
  list.forEach(function (p, i) {
    const key = p && p.id != null ? String(p.id) : String(i);
    byId.set(key, p);
  });
  const rows = goldRows.map(function (g, i) {
    const p = byId.get(String(g.id)) || list[i] || null;
    return { id: g.id, gold: p && p.gold != null ? strLabel(p.gold) : g.gold,
      prediction: p ? strLabel(p.prediction != null ? p.prediction : p.predict) : null, type: g.type };
  });
  return { rows: rows, official: null, parsedLines: list.length };
}

// ── 结果组装（纯函数） ──────────────────────────────────────────────────

/**
 * 把「评估集 + 引擎预测」组装成可写入 job.evaluation 的结构化结果。
 * 所有数值由代码计算，任何一处口径都在 method / warnings 里留痕。
 */
function assembleResult(ctx) {
  const gold = ctx.gold || { kind: 'generation', rows: [] };
  const parsed = ctx.parsed || { rows: [], official: null };
  const warnings = [];
  const kind = ctx.taskType || gold.kind;
  const rows = (parsed.rows || []).filter(function (r) { return r && r.gold != null && String(r.gold) !== ''; });
  const nGold = gold.rows.filter(function (r) { return String(r.gold) !== ''; }).length;
  if (rows.length < nGold) warnings.push('评估集 ' + nGold + ' 个预测单元，只解析到 ' + rows.length + ' 条预测；缺失按弃权处理（计入分母，不算正确）。');
  if (!rows.length) throw new Error('评估集没有可用的标准答案（label/output 全为空），无法评估');

  let metrics, details;
  // record-level bootstrap CI：决策模型一条记录多问题（单元间相关，Wilson 偏窄）。
  // 行 id 带 "record:field" 分组时启用；不可用为 null（警告文案回退提示 Wilson）。
  const bootCI = kind === 'classification' && rows.length && String(rows[0].id || '').indexOf(':') >= 0
    ? M.bootstrapRecordCI(rows, 1000) : null;
  if (kind === 'classification') {
    const cls = M.classificationMetrics(rows);
    metrics = {
      taskType: 'classification', n: cls.n, correct: cls.correct, coverage: cls.coverage,
      accuracy: cls.accuracy, accuracyCovered: cls.accuracyCovered,
      accuracyCI95: cls.ci95, accuracyCI95Record: bootCI,
      macroPrecision: cls.macro.precision, macroRecall: cls.macro.recall,
      macroF1: cls.macro.f1, microF1: cls.micro.f1, weightedF1: cls.weighted.f1,
      exactMatch: cls.accuracy, tokenF1: null, rougeL: null,
      calibration: parsed.official || null
    };
    details = { perClass: cls.perClass, confusion: cls.confusion,
      // qwen 建议：置信度-准确率校准曲线数据（rlcd report 的 signals.*.bins 分桶）
      calibrationSignals: parsed.report && parsed.report.signals ? { signals: parsed.report.signals } : null,
      perSample: rows.slice(0, 200).map(function (r) { return { id: r.id, type: r.type, gold: r.gold, prediction: r.prediction }; }) };
    // 与 kev 官方 acc 交叉校验（同一批 rows 的 argmax 判定应逐位一致）
    if (parsed.official && Number.isFinite(parsed.official.accuracy)) {
      const delta = Math.abs(parsed.official.accuracy - cls.accuracy);
      if (delta > 1e-6) warnings.push('平台计算 accuracy=' + cls.accuracy.toFixed(6) + ' 与引擎官方 acc=' + Number(parsed.official.accuracy).toFixed(6) + ' 不一致（Δ=' + delta.toFixed(6) + '），请检查评估集/版本是否一致。');
    }
  } else {
    const gen = M.generationMetrics(rows);
    metrics = {
      taskType: 'generation', n: gen.n, correct: null, coverage: gen.coverage,
      accuracy: gen.exactMatch, accuracyCovered: null, accuracyCI95: gen.ci95,
      macroPrecision: null, macroRecall: null, macroF1: null, microF1: null, weightedF1: null,
      exactMatch: gen.exactMatch, tokenF1: gen.tokenF1, rougeL: gen.rougeL,
      calibration: parsed.official || null
    };
    details = { perClass: [], confusion: null,
      perSample: gen.perSample.slice(0, 200).map(function (p) {
        const g = rows[p.index] || {};
        return { id: g.id, type: 'generation', gold: g.gold, prediction: p.prediction,
          exactMatch: p.exactMatch, tokenF1: p.tokenF1, rougeL: p.rougeL };
      }) };
    warnings.push('生成类任务：exact_match / token-F1 / ROUGE-L 衡量的是文本重合度，不等于事实正确性；关键结论仍需人工或 LLM 评审。');
  }

  const leak = ctx.leakage || { evalCount: rows.length, duplicated: 0, ratio: 0 };
  if (leak.ratio > 0) warnings.push('评估集与训练集有 ' + leak.duplicated + '/' + leak.evalCount + ' 条完全相同的样本（重复率 ' + (leak.ratio * 100).toFixed(1) + '%），分数可能虚高；请改用独立留出集。');
  // 独立样本数 = 评估集记录数；逐问题展开后的行数不等于独立观测数。
  const independentN = Number(ctx.independentN) > 0 ? Number(ctx.independentN) : rows.length;
  if (independentN < 30) warnings.push('独立评估记录仅 ' + independentN + ' 条（共 ' + rows.length + ' 个预测单元），区间较宽；不同运行间的小差异不宜当作显著结论。');
  if (rows.length > independentN && !bootCI) {
    warnings.push('每条记录包含多个问题，逐问题结果不独立；Wilson CI 按 ' + rows.length + ' 个预测单元计算，可能偏窄。');
  }
  if (ctx.evalSet && ctx.evalSet.sha256 && ctx.previousHash && ctx.previousHash !== ctx.evalSet.sha256) {
    warnings.push('本次评估集指纹与上次不同，两次分数不可直接比较。');
  }

  // 按问题类型分组（kev 的 choice / noul / score 各有不同难度）
  const byType = {};
  if (kind === 'classification') {
    const groups = {};
    rows.forEach(function (r) { const t = r.type || 'unknown'; (groups[t] = groups[t] || []).push(r); });
    Object.keys(groups).forEach(function (t) {
      const c = M.classificationMetrics(groups[t]);
      byType[t] = { n: c.n, accuracy: c.accuracy, macroF1: c.macro.f1, ci95: c.ci95 };
    });
  }

  return {
    id: ctx.id || newId('eval'), jobId: ctx.jobId, engine: ctx.engine,
    status: 'succeeded', startedAt: ctx.startedAt, finishedAt: ctx.finishedAt || now(),
    durationMs: (ctx.finishedAt || now()) - ctx.startedAt,
    evalDataset: ctx.evalSet || null, trainDataset: ctx.trainSet || null,
    leakage: leak,
    metrics: Object.assign({ independentN: independentN }, metrics),
    byType: byType, details: details, warnings: warnings,
    artifacts: ctx.artifacts || {},
    method: {
      classification: 'accuracy = correct/n（n 含弃权）；P/R/F1 的 macro / micro / weighted 按 sklearn 定义；CI = Wilson 95%',
      generation: 'SQuAD 归一化 exact_match、bag-of-tokens F1、ROUGE-L(LCS)，逐样本 macro 平均',
      engine: ctx.engine === 'kev' ? 'kev.benchmark 官方逐问题概率 argmax，报告官方 acc/ECE/Brier/NLL'
        : ctx.engine === 'llamafactory' ? 'LLaMA-Factory do_predict + predict_with_generate 的 generated_predictions.jsonl'
          : '外部导入 predictions（平台只做对齐与评分，不参与推理）',
      sources: ['LLaMA-Factory Eval docs', 'lm-evaluation-harness Metrics API', 'HF evaluate F1 metric card', 'kev/metrics.py + kev/benchmark.py']
    }
  };
}

// ── 任务化执行 ──────────────────────────────────────────────────────────

function getTask(id) { return TASKS.get(id) || readEvalTaskFile(id); }
function listTasks() {
  pruneTasks();
  // 磁盘上的历史任务也纳入列表（内存 Map 只有最近 MAX_TASKS 个）
  const seen = new Set();
  const out = [];
  Array.from(TASKS.values()).forEach(function (t) { seen.add(t.id); out.push(Object.assign({}, t)); });
  try {
    const files = fs.readdirSync(EVAL_TASKS_DIR).filter(f => /^evaltask-[a-z0-9]+\.json$/.test(f));
    for (const f of files) {
      const id = f.slice(0, -5);
      if (seen.has(id)) continue;
      const o = readEvalTaskFile(id);
      if (o) out.push(o);
    }
  } catch (e) { /* 目录读取失败忽略 */ }
  return out;
}

function cancelTask(id) {
  const t = TASKS.get(id);
  if (!t) return false;
  if (t.state === 'running') { t.cancelled = true; t.state = 'cancelling'; persistEvalTask(t); }
  return true;
}

/**
 * 启动评估任务（立即返回 taskId，长任务通过轮询拿状态）。
 * io 需提供：readJob / saveJob / datasetById / readDatasetSamples / obtain(plan) / paths
 * obtain(plan) 返回 { predictions, engine, artifacts, taskType?, log? }
 */
function startEvaluation(job, opts, io) {
  pruneTasks();
  const taskId = newId('evaltask');
  const task = { id: taskId, jobId: job.id, state: 'queued', progress: '准备评估集',
    startedAt: now(), finishedAt: null, result: null, error: null, cancelled: false,
    auto: !!opts.auto };
  TASKS.set(taskId, task);
  persistEvalTask(task);
  setImmediate(function () { run(task, job, opts, io).catch(function (e) {
    task.state = 'failed'; task.error = String((e && e.message) || e); task.finishedAt = now();
    persistEvalTask(task);
  }); });
  return task;
}

async function run(task, job, opts, io) {
  task.state = 'running';
  persistEvalTask(task);
  const startedAt = now();
  let current = io.readJob(job.id) || job;
  const previous = current.evaluation || null;
  const previousSucceeded = previous && previous.status === 'succeeded' ? previous : null;
  const markRunning = Object.assign({}, current, {
    evaluation: Object.assign({}, previous, {
      status: 'running', engine: opts.engine || 'auto', evalDatasetId: opts.evalDatasetId || null,
      startedAt: startedAt, error: null, taskId: task.id,
      previousHash: previous && previous.evalDataset && previous.evalDataset.sha256 || null
    })
  });
  await io.saveJob(markRunning);

  try {
    const ds = io.datasetById(opts.evalDatasetId);
    if (!ds) throw new Error('评估集不存在：' + opts.evalDatasetId);
    task.progress = '读取评估集';
    const allSamples = await io.readDatasetSamples(ds);
    if (!allSamples.length) throw new Error('评估集为空');
    // 截断在指纹计算之前：评估集指纹必须描述真正被评估的那批样本，否则报告里的
    // 「评估集版本」与实际推理输入不一致（大评估集尤其容易踩坑）。
    const limit = Math.max(1, Math.min(parseInt(opts.maxSamples, 10) || allSamples.length, 5000));
    const samples = allSamples.slice(0, limit);
    const trainSamples = job.datasetId ? await io.readDatasetSamples({ id: job.datasetId, fileName: null }) : [];
    const fp = M.evalSetFingerprint(samples);
    const leak = M.leakage(trainSamples, samples);
    const gold = extractGold(samples);

    const plan = {
      jobId: job.id, engine: opts.engine || (job.trainMode === 'kev' || job.trainMode === 'rlcd' ? 'kev' : 'llamafactory'),
      evalDatasetId: ds.id, samples: samples, gold: gold,
      maxSamples: limit, independentN: samples.length,
      evalSet: Object.assign({ id: ds.id, name: ds.name, kind: ds.kind }, fp),
      trainSet: { id: job.datasetId, name: job.datasetName },
      artifacts: {}
    };

    if (task.cancelled) throw new Error('评估已取消');
    task.progress = '运行模型推理';
    const out = await io.obtain(plan);
    if (task.cancelled) throw new Error('评估已取消');
    task.progress = '计算指标';

    let parsed;
    if (out.engine === 'kev') parsed = out.rlcdFormat ? parseRlcdOutput(out.rowsText, out.reportText, gold.rows) : parseKevOutput(out.rowsText, out.reportText);
    else if (out.engine === 'llamafactory') {
      parsed = parseLfOutput(out.predictionsText, gold.rows);
      if (parsed.parsedLines < gold.rows.length) task.progress = '部分样本无预测';
    } else parsed = parseImported(out.predictionsText, gold.rows);

    const result = assembleResult({
      id: newId('eval'), jobId: job.id, engine: out.engine,
      startedAt: startedAt, finishedAt: now(),
      gold: gold, parsed: parsed, taskType: out.taskType, independentN: samples.length,
      leakage: leak, evalSet: plan.evalSet, trainSet: plan.trainSet,
      previousHash: previous && previous.evalDataset && previous.evalDataset.sha256,
      artifacts: out.artifacts || {}
    });
    // 评估期间任务被删除：不得把 job 文件重新创建出来
    if (!io.readJob(job.id)) {
      task.state = 'failed'; task.error = '任务记录已被删除，评估结果不落盘';
      task.finishedAt = now(); task.progress = '失败';
      persistEvalTask(task);
      return;
    }
    current = io.readJob(job.id);
    const history = (current.evaluationHistory || []).filter(function (h) { return h && h.status === 'succeeded'; }).slice(-4);
    if (previous && previous.status === 'succeeded') history.push(previous);
    await io.saveJob(Object.assign({}, current, { evaluation: result, evaluationHistory: history }));
    task.state = 'succeeded'; task.result = result; task.finishedAt = now();
    task.progress = '完成';
    persistEvalTask(task);
  } catch (e) {
    // 失败不覆盖训练状态，也不抹掉上一次成功的评估结果
    const cur = io.readJob(job.id) || job;
    if (cur === job) {
      task.state = 'failed'; task.error = '任务记录已被删除，评估结果不落盘';
      task.finishedAt = now(); task.progress = '失败';
      persistEvalTask(task);
      return;
    }
    // markRunning 已把状态改成 running，不能用它当基准；用启动前快照恢复上一次成功结果
    const prev = previousSucceeded || cur.evaluation || {};
    await io.saveJob(Object.assign({}, cur, {
      evaluation: Object.assign({}, prev, {
        status: (previousSucceeded ? 'succeeded' : 'failed'),
        lastError: String((e && e.message) || e), failedAt: now(), taskId: task.id,
        autoFailed: !!task.auto
      })
    }));
    task.state = 'failed'; task.error = String((e && e.message) || e); task.finishedAt = now();
    task.progress = '失败';
    persistEvalTask(task);
  }
}

module.exports = {
  parseSamples, extractGold, parseKevOutput, parseRlcdOutput, parseLfOutput, parseImported,
  assembleResult, startEvaluation, getTask, listTasks, cancelTask,
  newId, _tasks: TASKS
};
