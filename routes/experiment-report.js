/**
 * 实验报告 / 多实验对比分析 — 后端路由（/api/experiment-report/*）
 *
 * 数据来源与边界（必须在 UI 上如实标注）：
 *  - 训练曲线、配置、loss：直接读 data/model-training/jobs/*.json（平台自动采集）
 *  - 精度类指标（accuracy / F1 / EM / 人工评测 / 业务指标）：平台当前【不自动跑评估集】，
 *    因此由用户在报告页录入或从 CSV/JSON 导入，每条记录保留 source / seed / n / unit / task，
 *    绝不把手工值伪装成自动评测结果。
 *  - 外部实验结果（其他框架导出的报告）也能导入为 external run，和本机任务同表对比。
 *
 * 所有统计量由 routes/experiment-stats.js 计算（纯函数、有数值单测），
 * 本文件只负责数据规范化、分组、聚合编排与持久化。
 *
 * 端点：
 *   GET  /api/experiment-report/runs                 可选 run 列表（本机任务 + 外部导入）
 *   GET  /api/experiment-report/metrics             指标记录列表
 *   POST /api/experiment-report/metrics             新增指标记录
 *   POST /api/experiment-report/metrics/delete      删除指标记录
 *   POST /api/experiment-report/metrics/import      CSV / JSON 批量导入（长表 + 宽表）
 *   POST /api/experiment-report/analyze             生成对比分析结果（图表所需的全部数值）
 *   GET  /api/experiment-report/reports             已保存报告列表
 *   POST /api/experiment-report/reports             保存报告快照
 *   GET  /api/experiment-report/reports/:id         读取报告
 *   POST /api/experiment-report/reports/delete      删除报告
 *   POST /api/experiment-report/export              导出 CSV / Markdown
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const S = require('./experiment-stats');
const AI = require('./experiment-report-ai');
const STORE = require('./json-store');

const STORE_DIR = path.join(__dirname, '..', 'data', 'model-training');
const JOBS_DIR = path.join(STORE_DIR, 'jobs');
const METRICS_FILE = path.join(STORE_DIR, 'metrics.json');
const REPORTS_DIR = path.join(STORE_DIR, 'reports');

for (const d of [STORE_DIR, JOBS_DIR, REPORTS_DIR]) fs.mkdirSync(d, { recursive: true });

/** 原子写 + 损坏隔离：详见 routes/json-store.js */
function readJson(file, fallback) { return STORE.readJson(file, fallback); }
function writeJson(file, obj) { STORE.writeJsonSync(file, obj); }
function newId(prefix) {
  return prefix + '-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}
/**
 * 报告快照的安全路径。所有来自 URL / body 的 id 都必须过这里。
 * 历史缺陷：body.id='../metrics' 可越界删除 data/model-training/metrics.json，
 * repMatch 同理可读任意 .json。白名单 + 目录前缀双重校验。
 */
const SAFE_ID = /^[A-Za-z0-9_-]+$/;
function reportFile(id) {
  const s = String(id == null ? '' : id);
  if (!s || s.length > 128 || !SAFE_ID.test(s)) return null;
  const f = path.join(REPORTS_DIR, s + '.json');
  return path.dirname(f) === REPORTS_DIR ? f : null;
}
function badId(res) { return sendJson(res, 400, { ok: false, error: '无效的报告 ID（仅允许字母、数字、下划线、连字符）' }); }
function sendJson(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json;charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
}
function readBody(req) {
  return new Promise(function (resolve) {
    const chunks = [];
    let len = 0;
    req.on('data', function (c) {
      len += c.length;
      if (len > 20 * 1024 * 1024) { req.destroy(); resolve({ error: 'body too large' }); return; }
      chunks.push(c);
    });
    req.on('end', function () {
      try { resolve({ data: JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}') }); }
      catch (e) { resolve({ error: 'invalid JSON: ' + e.message }); }
    });
    req.on('error', function () { resolve({ error: 'request aborted' }); });
  });
}
function num(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }

function readMetrics() {
  const m = readJson(METRICS_FILE, null);
  if (m && Array.isArray(m.entries)) return m;
  return { entries: [], externalRuns: [] };
}
function saveMetrics(m) { writeJson(METRICS_FILE, m); }
/**
 * 指标库「读 → 改 → 写」事务：同一文件串行执行，避免并发录入/导入互相覆盖（lost update）。
 * mutator 直接改传入对象；返回 undefined 时写回该对象。
 */
function withMetrics(mutator) {
  return STORE.updateJsonSync(METRICS_FILE, { entries: [], externalRuns: [] }, function (cur) {
    if (!cur || typeof cur !== 'object') cur = { entries: [], externalRuns: [] };
    if (!Array.isArray(cur.entries)) cur.entries = [];
    if (!Array.isArray(cur.externalRuns)) cur.externalRuns = [];
    const out = mutator(cur);
    return out === undefined ? cur : out;
  });
}

// ── Run 规范化 ──────────────────────────────────────────────────────────────

/** 越小越好的常见指标名（可被用户显式 direction 覆盖） */
const LOWER_IS_BETTER = /(loss|error|perplexity|^ppl$|rmse|mae|wer|cer|latency|runtime|time|cost|memory|flops)/i;
function guessDirection(name) { return LOWER_IS_BETTER.test(String(name || '')) ? 'lower' : 'higher'; }

function listJobFiles() {
  if (!fs.existsSync(JOBS_DIR)) return [];
  return fs.readdirSync(JOBS_DIR).filter(function (f) { return f.endsWith('.json'); })
    .map(function (f) { return readJson(path.join(JOBS_DIR, f), null); })
    .filter(Boolean);
}

/** 从训练日志中解析可选的运行指标（不写死某一种框架） */
function parseRuntimeSeconds(text) {
  const t = String(text || '');
  let m = t.match(/train_runtime\s*=\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)/);
  if (m) return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]);
  m = t.match(/train_runtime\s*=\s*([\d.]+)/);
  if (m) return parseFloat(m[1]);
  return null;
}
function parseEvalAccuracy(text) {
  const t = String(text || '');
  const m = [...t.matchAll(/eval_accuracy\s*=\s*([\d.]+)/g)].pop() ||
            [...t.matchAll(/(?:accuracy|acc)\s*[:=]\s*([\d.]+)/gi)].pop();
  return m ? parseFloat(m[1]) : null;
}

/** 把一条 job 记录规范化成统一 run 结构（字段缺失时留空，不臆造） */
function normalizeJob(job) {
  const cfg = job.config || {};
  const hist = Array.isArray(job.lossHistory) ? job.lossHistory : [];
  const evHist = Array.isArray(job.evalLossHistory) ? job.evalLossHistory : [];
  const losses = hist.map(function (h) { return num(h && h.loss); }).filter(function (v) { return v != null; });
  const model = cfg.modelName || cfg.model_name_or_path || '';
  const modelShort = model.split(/[\\/]/).filter(Boolean).pop() || model || '—';
  const dataset = job.datasetName || cfg.dataset || cfg.datasetId || '';
  const task = cfg.task || dataset || '—';
  const engine = cfg.finetuningType || cfg.stage || (job.trainMode === 'kev' ? 'kev' : '') || '—';
  const steps = (job.progress && job.progress.total) ||
    (hist.length && num(hist[hist.length - 1].step)) || null;

  const auto = [];
  function pushAuto(key, name, value, unit, direction, note, opts) {
    if (value == null || !Number.isFinite(Number(value))) return;
    const o = opts || {};
    auto.push({
      id: 'auto:' + job.id + ':' + key, runId: job.id, key: key, name: name,
      value: Number(value), unit: unit || '', direction: direction,
      task: task, seed: cfg.seed == null ? null : num(cfg.seed), n: o.n == null ? null : num(o.n),
      source: o.source || 'auto', ci95: o.ci95 || null,
      note: note || '', createdAt: job.createdAt || Date.now()
    });
  }
  const finalLoss = losses.length ? losses[losses.length - 1] : num(job.loss);
  pushAuto('auto.loss.final', 'Final Train Loss', finalLoss, '', 'lower', '训练日志最后一个 loss（非留出集指标）');
  if (losses.length) {
    pushAuto('auto.loss.min', 'Min Train Loss', Math.min.apply(null, losses), '', 'lower', '训练过程中观测到的最小 loss');
    if (losses.length > 1) pushAuto('auto.loss.improvement', 'Loss Improvement', losses[0] - losses[losses.length - 1], '', 'higher', '首个记录点 loss − 最后记录点 loss');
  }
  if (evHist.length) pushAuto('auto.eval.final', 'Final Eval Loss', evHist[evHist.length - 1], '', 'lower', '评估集 loss');
  const acc = parseEvalAccuracy(job.logTail);
  if (acc != null) pushAuto('auto.eval.accuracy', '准确率（日志解析）', acc, '', 'higher', '从训练日志中解析到的精度（非留出集评估，仅供参考）');
  if (steps != null) pushAuto('auto.steps', 'Training Steps', steps, 'steps', 'higher', '总训练步数');
  const runtime = parseRuntimeSeconds(job.logTail);
  if (runtime != null) pushAuto('auto.runtime', 'Train Runtime', runtime, 's', 'lower', '训练耗时');

  // ── 训练后自动评估：job.evaluation 是唯一事实源，报告只读取不重算 ──
  // 与「训练日志代理指标」严格区分：source=auto-eval，带 n、评估集指纹、Wilson CI。
  const ev = job.evaluation;
  const evInfo = (ev && ev.status === 'succeeded' && ev.metrics) ? ev : null;
  if (evInfo) {
    const m = evInfo.metrics;
    const dsName = (evInfo.evalDataset && evInfo.evalDataset.name) || '—';
    const hash = (evInfo.evalDataset && evInfo.evalDataset.sha256) || '';
    const tag = evInfo.engine + ' · ' + dsName + (hash ? ' · sha256 ' + String(hash).slice(0, 8) : '') + ' · n=' + m.n;
    const ciTxt = m.accuracyCI95 && m.accuracyCI95.low != null
      ? '；Wilson 95% CI [' + m.accuracyCI95.low.toFixed(4) + ', ' + m.accuracyCI95.high.toFixed(4) + ']' : '';
    const meta = { source: 'auto-eval', n: m.n, ci95: m.accuracyCI95 || null };
    if (m.taskType === 'classification') {
      pushAuto('eval.accuracy', '准确率（留出集）', m.accuracy, '', 'higher',
        'Accuracy：预测正确的比例（留出集 correct/n，n 含弃权）' + ciTxt + ' · ' + tag, meta);
      pushAuto('eval.macroF1', '宏平均 F1（Macro-F1）', m.macroF1, '', 'higher', 'Macro-F1：每个类别的 F1 等权平均，少数类与多数类同等重要 · ' + tag, meta);
      pushAuto('eval.weightedF1', '加权平均 F1（Weighted-F1）', m.weightedF1, '', 'higher', 'Weighted-F1：按各类别样本数加权平均，多数类占主导 · ' + tag, meta);
    } else {
      pushAuto('eval.exactMatch', '完全匹配（Exact Match）', m.exactMatch, '', 'higher',
        '生成文本经 SQuAD 归一化后与标准答案完全一致的比例' + ciTxt + ' · ' + tag, meta);
      pushAuto('eval.tokenF1', '词元 F1（Token-F1）', m.tokenF1, '', 'higher', '预测与答案的词元重合 F1 逐样本平均 · ' + tag, meta);
      pushAuto('eval.rougeL', 'ROUGE-L', m.rougeL, '', 'higher', '最长公共子序列 F1 逐样本平均 · ' + tag, meta);
    }
    const cal = m.calibration;
    if (cal) {
      pushAuto('eval.ece', '期望校准误差（ECE）', cal.ece, '', 'lower', 'ECE：模型自报置信度与真实正确率的加权偏差，越小表示「越诚实」· ' + tag, meta);
      pushAuto('eval.brier', 'Brier 评分', cal.brier, '', 'lower', 'Brier score：预测概率与实际结果的均方差，综合精度+校准的概率质量指标 · ' + tag, meta);
      pushAuto('eval.nll', '负对数似然（NLL）', cal.nll, '', 'lower', 'NLL：预测概率下的对数损失 · ' + tag, meta);
    }
  }

  // 分组可用的 facet：顶层关键字段 + config 里的数值型超参，全部通用化，不写死某个训练框架
  const facets = {
    task: task, model: modelShort, dataset: dataset, engine: engine,
    evalSet: evInfo && evInfo.evalDataset ? String(evInfo.evalDataset.sha256 || '').slice(0, 8) || '—' : '—',
    evalEngine: evInfo ? evInfo.engine : '—',
    stage: cfg.stage || '—', status: job.status || '—',
    epochs: cfg.epochs, learningRate: cfg.learningRate, loraRank: cfg.loraRank,
    loraAlpha: cfg.loraAlpha, batchSize: cfg.batchSize, gradAccum: cfg.gradAccum,
    cutoffLen: cfg.cutoffLen, seed: cfg.seed, template: cfg.template || '—',
    server: job.serverLabel || '—'
  };
  Object.keys(cfg).forEach(function (k) {
    if (!(k in facets) && (typeof cfg[k] === 'number' || typeof cfg[k] === 'string') &&
        !/^(name|outputDir|serverId|datasetId)$/.test(k)) facets[k] = cfg[k];
  });

  return {
    id: job.id, name: job.name || job.id, origin: 'job', status: job.status || '—',
    createdAt: job.createdAt || null, model: modelShort, modelFull: model,
    dataset: dataset, task: task, engine: engine, facets: facets,
    lossHistory: hist, evalLossHistory: evHist,
    progress: job.progress || null, config: cfg,
    autoMetrics: auto, metricCount: 0,
    evaluation: evInfo ? {
      id: evInfo.id, engine: evInfo.engine, evalDataset: evInfo.evalDataset,
      n: evInfo.metrics.n, taskType: evInfo.metrics.taskType,
      leakage: evInfo.leakage, warnings: evInfo.warnings,
      metrics: evInfo.metrics, finishedAt: evInfo.finishedAt
    } : null
  };
}

function normalizeExternal(ext) {
  return {
    id: ext.id, name: ext.name || ext.id, origin: 'external', status: 'imported',
    createdAt: ext.createdAt || null,
    model: ext.model || '—', modelFull: ext.model || '', dataset: ext.dataset || '—',
    task: ext.task || ext.dataset || '—', engine: ext.engine || '—',
    facets: {
      task: ext.task || '—', model: ext.model || '—', dataset: ext.dataset || '—',
      engine: ext.engine || '—', status: 'imported', seed: ext.seed == null ? null : ext.seed
    },
    lossHistory: [], evalLossHistory: [], progress: null, config: {},
    autoMetrics: [], metricCount: 0, evaluation: null
  };
}

function listRuns() {
  const store = readMetrics();
  const jobs = listJobFiles().map(normalizeJob).sort(function (a, b) { return (b.createdAt || 0) - (a.createdAt || 0); });
  const ext = (store.externalRuns || []).map(normalizeExternal);
  const all = jobs.concat(ext);
  const counts = {};
  store.entries.forEach(function (e) { counts[e.runId] = (counts[e.runId] || 0) + 1; });
  all.forEach(function (r) { r.metricCount = counts[r.id] || 0; });
  return all;
}

/** 汇总一条 run 的全部指标行（自动 + 录入/导入） */
function metricRowsFor(runs) {
  const byId = {};
  runs.forEach(function (r) { byId[r.id] = r; });
  const store = readMetrics();
  const rows = [];
  runs.forEach(function (r) { (r.autoMetrics || []).forEach(function (m) { rows.push(m); }); });
  (store.entries || []).forEach(function (e) {
    if (!byId[e.runId]) return;
    rows.push({
      id: e.id, runId: e.runId, key: 'user.' + e.name, name: e.name,
      value: num(e.value), unit: e.unit || '',
      direction: e.direction || guessDirection(e.name),
      task: e.task || byId[e.runId].task, seed: e.seed == null ? null : num(e.seed),
      n: e.n == null ? null : num(e.n), source: e.source || 'manual',
      note: e.note || '', createdAt: e.createdAt || Date.now()
    });
  });
  return rows.filter(function (r) { return r.value != null && Number.isFinite(r.value); });
}

function metricCatalog(rows) {
  const map = {};
  rows.forEach(function (m) {
    const k = m.name;
    if (!map[k]) map[k] = { name: k, key: m.key, unit: m.unit, direction: m.direction, source: m.source, n: 0, runIds: [] };
    map[k].n++;
    if (map[k].runIds.indexOf(m.runId) < 0) map[k].runIds.push(m.runId);
    if (m.unit && map[k].unit && m.unit !== map[k].unit) map[k].mixedUnits = true;
  });
  return Object.keys(map).map(function (k) { return map[k]; })
    .sort(function (a, b) { return (a.source === 'auto' ? 0 : 1) - (b.source === 'auto' ? 0 : 1) || a.name.localeCompare(b.name); });
}

// ── 分析编排 ────────────────────────────────────────────────────────────────

const GROUP_LABELS = {
  none: '全部', task: '任务 (task)', model: '基座模型', dataset: '数据集',
  engine: '训练引擎', stage: '训练阶段', status: '状态', epochs: 'epochs',
  learningRate: 'learningRate', loraRank: 'loraRank', batchSize: 'batchSize',
  gradAccum: 'gradAccum', cutoffLen: 'cutoffLen', seed: 'seed', server: '服务器'
};

function groupKeyFor(run, groupBy) {
  if (!groupBy || groupBy === 'none') return '全部';
  let v = run.facets[groupBy];
  if (v == null && run.config) v = run.config[groupBy];
  if (v == null || v === '') return '—';
  if (typeof v === 'number') return String(v);
  return String(v);
}

function fmtGroupLabel(groupBy, key) {
  if (!groupBy || groupBy === 'none') return key;
  const prefix = (GROUP_LABELS[groupBy] || groupBy) + '=';
  return prefix + (key.length > 34 ? key.slice(0, 32) + '…' : key);
}

const PALETTE = ['#2060cc', '#c0392b', '#0e9f6e', '#8040cc', '#d97706', '#0e7490', '#be185d', '#4d7c0f'];

function analyze(body) {
  const allRuns = listRuns();
  const wanted = Array.isArray(body.runIds) ? body.runIds : [];
  const runs = allRuns.filter(function (r) { return wanted.indexOf(r.id) >= 0; });
  if (!runs.length) throw new Error('请至少选择一条实验运行');

  const rows = metricRowsFor(runs);
  const catalog = metricCatalog(rows);
  if (!catalog.length) throw new Error('所选运行没有任何指标：请先录入评估指标，或选择有 loss 曲线的训练任务');

  let metricNames = Array.isArray(body.metricNames) && body.metricNames.length ? body.metricNames : [catalog[0].name];
  metricNames = metricNames.filter(function (n) { return catalog.some(function (c) { return c.name === n; }); });
  if (!metricNames.length) metricNames = [catalog[0].name];
  const primary = metricNames[0];
  const groupBy = body.groupBy || 'none';
  const xMode = body.xMode || 'progress';
  const emaAlpha = body.emaAlpha == null ? 0.6 : Math.max(0, Math.min(0.99, Number(body.emaAlpha)));
  const histMethod = body.histogramMethod || 'auto';
  const bandType = body.bandType === 'sd' ? 'sd' : 'ci95';

  // 分组
  const groupsMap = {};
  runs.forEach(function (r) {
    const k = groupKeyFor(r, groupBy);
    if (!groupsMap[k]) groupsMap[k] = { key: k, label: fmtGroupLabel(groupBy, k), runIds: [], values: {} };
    groupsMap[k].runIds.push(r.id);
  });
  const groups = Object.keys(groupsMap).map(function (k, i) {
    return Object.assign(groupsMap[k], { color: PALETTE[i % PALETTE.length] });
  });
  const groupOfRun = {};
  groups.forEach(function (g) { g.runIds.forEach(function (id) { groupOfRun[id] = g.key; }); });
  rows.forEach(function (m) {
    const g = groupsMap[groupOfRun[m.runId]];
    if (!g) return;
    if (!g.values[m.name]) g.values[m.name] = [];
    g.values[m.name].push(m.value);
  });

  // 每个指标 × 每组的描述统计
  // codex 审核：Brier/ECE 等有界指标的 t-CI 可能越出定义域（负数或 >1），
  // 对这类指标把 CI 裁剪到定义域并在描述里标注
  // 有界指标：CI 裁剪到定义域（按中文名与旧英文名双键兼容，历史快照仍能命中）
  const BOUNDED_METRICS = {
    'Brier 评分': [0, 1], 'Eval Brier': [0, 1], 'Brier': [0, 1],
    '期望校准误差（ECE）': [0, 1], 'Eval ECE': [0, 1], 'ECE': [0, 1],
    '准确率（留出集）': [0, 1], 'Eval Accuracy': [0, 1],
    '宏平均 F1（Macro-F1）': [0, 1], 'Eval Macro-F1': [0, 1],
    '加权平均 F1（Weighted-F1）': [0, 1], 'Eval Weighted-F1': [0, 1], 'Eval Micro-F1': [0, 1],
  };
  const stats = {};
  metricNames.forEach(function (name) {
    stats[name] = groups.map(function (g) {
      const desc = S.describe(g.values[name] || []);
      const domain = BOUNDED_METRICS[name];
      if (domain && desc.ci95) {
        const clipped = [Math.max(domain[0], desc.ci95[0]), Math.min(domain[1], desc.ci95[1])];
        if (clipped[0] !== desc.ci95[0] || clipped[1] !== desc.ci95[1]) desc.ciClipped = true;
        desc.ci95 = clipped;
      }
      return { group: g.key, label: g.label, color: g.color, stats: desc, values: (g.values[name] || []).slice() };
    });
  });

  // 显著性检验：组两两比较；有可对齐 seed 时配对，否则 Welch，并做 Bonferroni 校正
  const pairs = [];
  for (let i = 0; i < groups.length; i++) {
    for (let j = i + 1; j < groups.length; j++) pairs.push([groups[i], groups[j]]);
  }
  const tests = pairs.map(function (p) {
    const [g1, g2] = p;
    const r1 = rows.filter(function (m) { return m.name === primary && g1.runIds.indexOf(m.runId) >= 0; });
    const r2 = rows.filter(function (m) { return m.name === primary && g2.runIds.indexOf(m.runId) >= 0; });
    const seeded = r1.length >= 2 && r2.length >= 2 &&
      r1.every(function (m) { return m.seed != null; }) && r2.every(function (m) { return m.seed != null; });
    let res, mode = 'unpaired';
    if (seeded) {
      const bySeed1 = {}, bySeed2 = {};
      r1.forEach(function (m) { (bySeed1[m.seed] = bySeed1[m.seed] || []).push(m.value); });
      r2.forEach(function (m) { (bySeed2[m.seed] = bySeed2[m.seed] || []).push(m.value); });
      const common = Object.keys(bySeed1).filter(function (s) { return bySeed2[s]; }).sort(function (a, b) { return Number(a) - Number(b); });
      if (common.length >= 2) {
        res = S.pairedTTest(common.map(function (s) { return S.mean(bySeed1[s]); }),
          common.map(function (s) { return S.mean(bySeed2[s]); }));
        mode = 'paired';
      }
    }
    if (!res || !res.applicable) {
      res = S.welchTTest(r1.map(function (m) { return m.value; }), r2.map(function (m) { return m.value; }));
      mode = 'unpaired';
    }
    const corr = S.bonferroni(res.p, Math.max(1, pairs.length));
    return {
      metric: primary, groupA: g1.label, groupB: g2.label, mode: mode,
      nA: r1.length, nB: r2.length, result: res,
      alphaCorrected: corr.alphaCorrected, significant: corr.significant,
      note: mode === 'paired'
        ? '按相同 seed 配对；假设配对差值近似正态且配对相互独立。'
        : '两组 seed 无法对齐，使用 Welch t 检验（不假设方差齐性），统计功效低于配对检验。'
    };
  });

  // 直方图（共享 bins）
  const histogram = S.sharedHistogram(groups.map(function (g) {
    return { label: g.label, color: g.color, values: (g.values[primary] || []).slice() };
  }), histMethod);

  // 叠加 loss 曲线：逐 run 的 raw + EMA，以及按组的 mean±区间聚合带
  const curves = runs.map(function (r, idx) {
    const total = (r.progress && r.progress.total) || null;
    const s = S.buildSeries(r.lossHistory, xMode, total, r.config && r.config.epochs);
    if (s.available) s.ema = S.emaSmooth(s.points.map(function (p) { return p.y; }), emaAlpha);
    return {
      runId: r.id, name: r.name, group: groupOfRun[r.id],
      color: (groups.find(function (g) { return g.key === groupOfRun[r.id]; }) || {}).color || PALETTE[idx % PALETTE.length],
      series: s
    };
  }).filter(function (c) { return c.series.available && c.series.points.length; });

  const aggregates = groups.map(function (g) {
    const list = curves.filter(function (c) { return c.group === g.key; }).map(function (c) { return c.series; });
    const agg = S.aggregateSeries(list, 60);
    return { group: g.key, label: g.label, color: g.color, grid: agg.grid, note: agg.note };
  });

  // 雷达图：跨组、跨指标 min-max 归一化；至少 3 个指标才有意义
  const radarMetrics = metricNames.slice(0, 7).map(function (name) {
    const meta = catalog.find(function (c) { return c.name === name; }) || {};
    const means = groups.map(function (g) { return S.mean(g.values[name] || []); });
    const finite = means.filter(function (v) { return v != null; });
    const norm = S.normalizeMetric(finite, meta.direction || 'higher');
    const scores = [], raw = means.slice();
    let cursor = 0;
    means.forEach(function (v) { scores.push(v == null ? 0 : norm.scores[cursor++]); });
    return {
      name: name, unit: meta.unit || '', direction: meta.direction || 'higher',
      min: norm.min, max: norm.max, degenerate: norm.degenerate,
      note: norm.note || '', raw: raw, scores: scores
    };
  });
  const radarEnabled = radarMetrics.length >= 3 && groups.length >= 2 && groups.length <= 3;
  const radar = {
    enabled: radarEnabled,
    reason: radarMetrics.length < 3 ? '雷达图至少需要 3 个指标（当前 ' + radarMetrics.length + ' 个），请多选指标；少于 3 维时柱状图更精确'
      : groups.length > 3 ? '叠加组数超过 3，图形不可读；请减少分组或改用柱状图' : '',
    axes: radarMetrics, groups: groups.map(function (g, gi) {
      return { label: g.label, color: g.color, scores: radarMetrics.map(function (a) { return a.scores[gi]; }) };
    }),
    convention: '所有轴统一 min-max 归一化到 [0,1]，lower-is-better 指标取反；轴序固定；多边形面积不代表综合得分。'
  };

  // 平行坐标：数值型配置 facet + 主指标，逐轴归一化；分类字段仅作标签
  // 自动发现所有「数值型且在所选 run 间有差异」的配置轴，不写死某个训练框架的超参名
  const allFacetKeys = Array.from(new Set(runs.reduce(function (acc, r) {
    return acc.concat(Object.keys(r.facets || {}));
  }, [])));
  const numericFacets = [];
  allFacetKeys.forEach(function (k) {
    const vals = runs.map(function (r) { return num(r.facets[k]); }).filter(function (v) { return v != null; });
    if (vals.length >= 2 && new Set(vals).size > 1) numericFacets.push({ key: k, values: vals });
  });
  numericFacets.length = Math.min(numericFacets.length, 6); // 轴过多会不可读
  const primaryByRun = {};
  rows.filter(function (m) { return m.name === primary; }).forEach(function (m) {
    (primaryByRun[m.runId] = primaryByRun[m.runId] || []).push(m.value);
  });
  const parallel = {
    axes: numericFacets.map(function (f) { return { key: f.key, label: f.key }; }).concat([{ key: primary, label: primary }]),
    runs: runs.map(function (r) {
      const vals = numericFacets.map(function (f) { return num(r.facets[f.key]); });
      const pm = S.mean(primaryByRun[r.id] || []);
      vals.push(pm);
      return { runId: r.id, name: r.name, group: groupOfRun[r.id], color: (groups.find(function (g) { return g.key === groupOfRun[r.id]; }) || {}).color, values: vals };
    })
  };
  // 逐 run × 逐轴归一化（缺失值保持 null，避免与有效值错位）
  parallel.normalized = parallel.runs.map(function (r) {
    return r.values.map(function (v, ai) {
      if (v == null || !Number.isFinite(v)) return null;
      const vals = parallel.runs.map(function (x) { return x.values[ai]; }).filter(function (u) { return u != null && Number.isFinite(u); });
      if (vals.length < 2) return 0.5;
      const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      return hi === lo ? 0.5 : (v - lo) / (hi - lo);
    });
  });

  // Diff-only 运行对比表（对标 W&B Run Comparer / ClearML 标量表）
  // 常用字段优先展示，其余配置字段自动补齐（换框架 / 换任务时不会漏字段）
  const coreKeys = ['status', 'model', 'engine', 'dataset', 'task', 'stage', 'epochs', 'learningRate',
    'loraRank', 'loraAlpha', 'batchSize', 'gradAccum', 'cutoffLen', 'template', 'server', 'seed'];
  const diffKeys = coreKeys.filter(function (k) { return allFacetKeys.indexOf(k) >= 0; })
    .concat(allFacetKeys.filter(function (k) { return coreKeys.indexOf(k) < 0; }))
    .slice(0, 18);
  const tableRows = [];
  diffKeys.forEach(function (k) {
    const vals = runs.map(function (r) { return r.facets[k] == null ? '—' : String(r.facets[k]); });
    tableRows.push({ label: k, values: vals, differs: new Set(vals).size > 1, kind: 'config' });
  });
  catalog.forEach(function (c) {
    const vals = runs.map(function (r) {
      const v = rows.filter(function (m) { return m.runId === r.id && m.name === c.name; }).map(function (m) { return m.value; });
      const m = S.mean(v);
      return m == null ? '—' : (v.length > 1 ? 'mean=' + m.toFixed(4) + ' (n=' + v.length + ')' : String(v[0]));
    });
    tableRows.push({ label: c.name + (c.unit ? ' (' + c.unit + ')' : ''), values: vals, differs: new Set(vals).size > 1, kind: 'metric', source: c.source });
  });

  // 可比性与数据质量警告（W&B Run Comparer 明确指出不同 job_type 不可直接比较）
  const warnings = [];
  ['model', 'dataset', 'task', 'engine', 'stage'].forEach(function (k) {
    const distinct = Array.from(new Set(runs.map(function (r) { return String(r.facets[k] || '—'); })));
    if (distinct.length > 1) warnings.push('所选运行的 ' + k + ' 不一致（' + distinct.slice(0, 4).join(' / ') + (distinct.length > 4 ? ' …' : '') + '）：跨任务/跨模型的指标不能作为同一基准直接比较，建议按其分组。');
  });
  const unitMeta = catalog.find(function (c) { return c.name === primary; });
  if (unitMeta && unitMeta.mixedUnits) warnings.push('主指标「' + primary + '」存在不同单位，数值对比无意义，请先统一单位。');
  groups.forEach(function (g) {
    const desc = S.describe(g.values[primary] || []);
    if (desc.n > 0 && desc.n < 3) warnings.push('分组「' + g.label + '」仅有 n=' + desc.n + ' 个观测，标准差与置信区间不稳定；学术报告建议 n≥3（小改进建议 n≥5）。');
  });
  metricNames.forEach(function (n) {
    const dup = rows.filter(function (m) { return m.name === n && m.source !== 'auto'; }).length;
    if (dup) warnings.push('指标「' + n + '」同时存在训练日志自动值与录入/导入值（' + dup + ' 条），已合并统计；请确认二者单位与统计口径一致。');
  });
  const autoOnly = metricNames.every(function (n) {
    const c = catalog.find(function (x) { return x.name === n; });
    return c && c.source === 'auto';
  });
  if (autoOnly) warnings.push('当前只选择了训练日志自动代理指标（loss / steps / runtime），它们不是留出集精度；请在训练任务上配置自动评估，或录入评估指标。');
  // 自动评估的可比性：不同评估集内容 / 不同引擎 / 与训练集重复，都会让数字不可直接比较
  const evalHashes = Array.from(new Set(runs.map(function (r) {
    return r.evaluation && r.evaluation.evalDataset ? String(r.evaluation.evalDataset.sha256 || '') : '';
  }).filter(Boolean)));
  if (evalHashes.length > 1) warnings.push('所选运行的自动评估使用了 ' + evalHashes.length + ' 个不同内容的评估集（指纹不一致），跨评估集的 accuracy/F1 不可直接比较。');
  const evalEngines = Array.from(new Set(runs.map(function (r) { return r.evaluation ? r.evaluation.engine : ''; }).filter(Boolean)));
  if (evalEngines.length > 1) warnings.push('自动评估引擎不一致（' + evalEngines.join(' / ') + '）：不同推理实现/生成参数下的分数需分别解读。');
  runs.forEach(function (r) {
    const e = r.evaluation;
    if (e && e.leakage && e.leakage.ratio > 0) {
      warnings.push('运行「' + r.name + '」的评估集与训练集有 ' + e.leakage.duplicated + '/' + e.leakage.evalCount + ' 条重复样本，分数可能虚高。');
    }
    if (e && e.warnings && e.warnings.length) warnings.push('运行「' + r.name + '」评估口径提示：' + e.warnings[0]);
  });
  const noEval = runs.filter(function (r) { return r.origin === 'job' && !r.evaluation; });
  if (noEval.length) warnings.push('有 ' + noEval.length + ' 条训练任务没有自动评估结果（仅 loss 曲线）；如需对比精度，请在任务详情里运行评估。');
  if (curves.length < runs.length) warnings.push('有 ' + (runs.length - curves.length) + ' 条运行没有 loss 曲线（外部导入或尚未产生日志），叠加曲线中已跳过。');
  const degraded = curves.filter(function (c) { return c.series.degraded; });
  if (degraded.length) warnings.push('x 轴口径降级：' + degraded.length + ' 条运行缺少 step 字段，已按 Logging Point Index 绘制（' + degraded[0].series.degraded + '）。');
  if (tests.length && tests.some(function (t) { return t.mode === 'unpaired'; })) warnings.push('部分组间比较因 seed 无法对齐而降级为 Welch 非配对检验；若要做严格的配对检验，请为每条指标录入相同的 seed。');

  return {
    ok: true,
    meta: {
      generatedAt: Date.now(), runCount: runs.length, groupBy: groupBy,
      groupLabel: GROUP_LABELS[groupBy] || groupBy, xMode: xMode,
      xLabel: curves.length ? curves[0].series.xLabel : '',
      emaAlpha: emaAlpha, histogramMethod: histogram.method,
      primary: primary, metricNames: metricNames, bandType: bandType,
      palette: PALETTE
    },
    runs: runs.map(function (r) {
      return {
        id: r.id, name: r.name, origin: r.origin, status: r.status, createdAt: r.createdAt,
        model: r.model, dataset: r.dataset, task: r.task, engine: r.engine,
        facets: r.facets, group: groupOfRun[r.id], metricCount: r.metricCount,
        evaluation: r.evaluation ? {
          engine: r.evaluation.engine, n: r.evaluation.n, taskType: r.evaluation.taskType,
          evalDataset: r.evaluation.evalDataset, leakage: r.evaluation.leakage
        } : null,
        hasCurve: curves.some(function (c) { return c.runId === r.id; })
      };
    }),
    catalog: catalog, groups: groups.map(function (g) { return { key: g.key, label: g.label, color: g.color, runIds: g.runIds }; }),
    stats: stats, tests: tests, histogram: histogram,
    curves: curves, aggregates: aggregates,
    radar: radar, parallel: parallel, tableRows: tableRows,
    warnings: warnings,
    methods: {
      ci: '95% CI = mean ± t(0.975, n−1) · s/√n，s 为样本标准差 (ddof=1)；n<2 不输出 CI。',
      sd: '标准差描述观测散布，置信区间描述均值不确定性，二者不可混用。',
      ema: 'TensorBoard 风格去偏置 EMA：s=α·s+(1−α)·x，除以权重 w 消除冷启动偏差。',
      histogram: 'numpy.histogram_bin_edges 口径：auto = min(h_fd, h_sturges)，h_fd=2·IQR/n^(1/3)，h_sturges=range/(log2 n+1)；所有分组共享同一组 bin 边界；最后一个 bin 右闭。',
      test: '同 seed 可配对时用 paired t-test，否则 Welch t-test；多组两两比较以 Bonferroni 校正 α；同时报告 Cohen\'s d 效应量。',
      radar: '雷达图各轴统一 min-max 归一化，lower-is-better 指标取反；面积不代表综合得分；精确数值请看柱状图与汇总表。',
      aggregation: '分组曲线在公共 x 网格上逐 run 线性插值，禁止外推；网格点 n 为参与聚合的 run 数，n<2 不给置信带。',
      autoEval: '训练后自动评估：kev.benchmark 逐问题概率 argmax，或 LLaMA-Factory do_predict + predict_with_generate 产出预测文件。分类任务 accuracy=correct/n（n 含弃权样本，另报 coverage），macro/micro/weighted P·R·F1 按 sklearn 定义，二项比例的 95% 区间用 Wilson score（不是 t 分布）；生成任务用 SQuAD 归一化 exact_match、bag-of-tokens F1、ROUGE-L(LCS)，均为逐样本 macro 平均。评估集内容 SHA-256 指纹与训练/评估重复率一并记录，逐样本预测留痕。',
      sources: [
        'LLaMA-Factory Eval (llamafactory.readthedocs.io/getting_started/eval.html): 训练后独立评估；do_predict + predict_with_generate 产出 generated_predictions.jsonl 再计算 NLG 指标',
        'lm-evaluation-harness Metrics API (lm-evaluation-harness.readthedocs.io/api/metrics): acc = mean(逐样本 0/1)、exact_match aggregation=mean、corpus 级指标先收集中间结果；逐样本结果可留痕',
        'HF evaluate F1 metric card: 多分类必须显式指定 average（macro/micro/weighted），默认 binary 在多分类下会给出误导性结果',
        'W&B Workspace (docs.wandb.ai/models/track/project-page): Group by → "a line along the mean value, and a shaded region for the variance"; Runs sidebar 支持 filter/group/sort/visibility',
        'W&B Run Comparer (docs.wandb.ai/models/app/features/panels/run-comparer): Diff only 与不可比 job_type 警告；W&B Reports Gallery: 章节化叙述 + 结论前置',
        'MLflow Quickstart 2 Compare the results (mlflow.org/docs/2.9.2/.../quickstart-2): Table view 按指标排序 + Chart view parallel coordinates 按目标指标着色',
        'ClearML Comparing Models (clear.ml/docs/.../webapp_model_comparing): Last/Min/Max 标量表、叠加曲线、单值标量 clustered bar chart、Hide Identical Fields、CSV 导出；内嵌图使用 Plotly',
        'Aim (github.com/aimhubio/aim README): "tracked params are first class citizens... search, group, aggregate via params"；明确指出 TensorBoard/MLflow 缺少分组聚合',
        'TensorBoard (github.com/tensorflow/tensorboard README + plugins/hparams/api.py): run 目录对比、HParams parallel coordinates；无分组聚合与统计检验',
        'Plotly.js (plotly.com/javascript/error-bars, continuous-error-bars, parallel-coordinates-plot): error_y 误差棒、fill=tonexty 置信带、显式 xbins、scatterpolar、parcoords；MIT 且官方明确可离线',
        'numpy.histogram_bin_edges: auto = min(h_fd, h_sturges)，h_fd=2·IQR/n^(1/3)，共享 bin 边界；matplotlib fill_between: mean ± 区间半透明带',
        'ML 多随机种子报告规范: n / mean / sd (ddof=1) / 95% CI (t 分布) / 配对 t 检验 / Bonferroni / Cohen\'s d',
        'arXiv:2611.04515 Learnings and Pitfalls from AI-Assisted A/B Experimentation at Scale: 结构化报告优于聊天记录、模板化章节、图表引用确定性产物、"code produces every number"',
        'Nature Formatting Guide: figure legend 以短标题开头、误差棒与统计量必须在 legend 定义、方法细节另立章节'
      ]
    }
  };
}

// ── CSV / JSON 导入 ─────────────────────────────────────────────────────────

function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', quoted = false;
  const t = String(text).replace(/^\uFEFF/, '');
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (quoted) {
      if (ch === '"' && t[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(cell.trim()); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && t[i + 1] === '\n') i++;
      row.push(cell.trim()); cell = '';
      if (row.some(function (c) { return c !== ''; })) rows.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell.trim());
  if (row.some(function (c) { return c !== ''; })) rows.push(row);
  return rows;
}
function lower(s) { return String(s || '').trim().toLowerCase().replace(/[\s\-]+/g, '_'); }
const ALIASES = {
  run: ['run', 'run_id', 'runid', 'experiment', 'exp', 'name', 'job', 'job_id', 'method', '配置', '实验'],
  metric: ['metric', 'metric_name', '指标', '指标名'],
  value: ['value', 'score', 'result', '数值', '值'],
  unit: ['unit', '单位'],
  task: ['task', 'dataset', '任务', '数据集'],
  seed: ['seed', '随机种子'],
  n: ['n', 'samples', 'sample_size', '样本量'],
  model: ['model', '模型'],
  engine: ['engine', 'method_type', '训练引擎'],
  direction: ['direction', '方向', 'higher_is_better']
};
function findCol(header, key) {
  return header.findIndex(function (h) { return ALIASES[key].indexOf(lower(h)) >= 0; });
}

/** 把 JSON 或 CSV（长表 / 宽表）转成统一指标记录 */
function parseImport(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) throw new Error('导入内容为空');
  let records = [];
  if (trimmed[0] === '[' || trimmed[0] === '{') {
    let obj;
    try { obj = JSON.parse(trimmed); } catch (e) { throw new Error('JSON 解析失败：' + e.message); }
    const arr = Array.isArray(obj) ? obj : (obj.runs || obj.records || obj.data || []);
    if (!Array.isArray(arr)) throw new Error('JSON 需要是数组，或包含 runs / records / data 数组');
    const WIDE_META = ['run', 'run_id', 'runid', 'name', 'experiment', 'exp', 'job', 'job_id', 'task', 'dataset', 'seed', 'n', 'samples', 'unit', 'model', 'engine', 'note', 'direction'];
    arr.forEach(function (o) {
      const out = {};
      Object.keys(o).forEach(function (k) { out[lower(k)] = o[k]; });
      if (out.metric == null) {
        // 宽表对象：除元数据字段外，其余数值字段视为指标
        Object.keys(out).forEach(function (k) {
          if (WIDE_META.indexOf(k) >= 0) return;
          const v = num(out[k]);
          if (v == null) return;
          records.push({
            run: out.run || out.run_id || out.name || out.experiment, metric: k, value: v,
            unit: out.unit || '', task: out.task || out.dataset, seed: out.seed,
            n: out.n || out.samples, model: out.model, engine: out.engine,
            direction: out.direction, note: out.note
          });
        });
        return;
      }
      records.push({
        run: out.run || out.run_id || out.name || out.experiment,
        metric: out.metric || out.metric_name, value: num(out.value != null ? out.value : out.score),
        unit: out.unit || '', task: out.task || out.dataset, seed: out.seed,
        n: out.n || out.samples, model: out.model, engine: out.engine,
        direction: out.direction, note: out.note
      });
    });
  } else {
    const rows = parseCsv(trimmed);
    if (rows.length < 2) throw new Error('CSV 至少需要表头 + 1 行数据');
    const header = rows[0];
    const iMetric = findCol(header, 'metric'), iValue = findCol(header, 'value'), iRun = findCol(header, 'run');
    if (iMetric >= 0 && iValue >= 0) {
      // 长表：一行一个 (run, metric, value)
      const idx = {};
      Object.keys(ALIASES).forEach(function (k) { idx[k] = findCol(header, k); });
      records = rows.slice(1).map(function (r) {
        return {
          run: r[iRun >= 0 ? iRun : 0], metric: r[iMetric], value: num(r[iValue]),
          unit: idx.unit >= 0 ? r[idx.unit] : '', task: idx.task >= 0 ? r[idx.task] : '',
          seed: idx.seed >= 0 ? r[idx.seed] : null, n: idx.n >= 0 ? r[idx.n] : null,
          model: idx.model >= 0 ? r[idx.model] : '', engine: idx.engine >= 0 ? r[idx.engine] : '',
          direction: idx.direction >= 0 ? r[idx.direction] : ''
        };
      });
    } else {
      // 宽表：一行一个 run，列名即指标名
      const metaIdx = {};
      Object.keys(ALIASES).forEach(function (k) { if (k !== 'metric' && k !== 'value') metaIdx[k] = findCol(header, k); });
      const metaCols = Object.keys(metaIdx).filter(function (k) { return metaIdx[k] >= 0; }).map(function (k) { return metaIdx[k]; });
      rows.slice(1).forEach(function (r) {
        header.forEach(function (h, ci) {
          if (metaCols.indexOf(ci) >= 0) return;
          const v = num(r[ci]);
          if (v == null) return;
          records.push({
            run: r[metaIdx.run >= 0 ? metaIdx.run : 0], metric: h, value: v,
            unit: metaIdx.unit >= 0 ? r[metaIdx.unit] : '',
            task: metaIdx.task >= 0 ? r[metaIdx.task] : '',
            seed: metaIdx.seed >= 0 ? r[metaIdx.seed] : null,
            n: metaIdx.n >= 0 ? r[metaIdx.n] : null,
            model: metaIdx.model >= 0 ? r[metaIdx.model] : '',
            engine: metaIdx.engine >= 0 ? r[metaIdx.engine] : ''
          });
        });
      });
    }
  }
  const cleaned = records.filter(function (r) {
    return r && r.metric && String(r.metric).trim() && r.value != null && Number.isFinite(r.value);
  });
  if (!cleaned.length) throw new Error('未解析到任何有效指标行（需要 metric + 数值 value，或宽表中的数值列）');
  return cleaned;
}

function resolveRunRef(ref, store) {
  const all = listRuns();
  const s = String(ref == null ? '' : ref).trim();
  if (!s) return null;
  const hit = all.find(function (r) { return r.id === s || r.name === s; });
  if (hit) return { runId: hit.id, created: false, task: hit.task };
  let ext = (store.externalRuns || []).find(function (e) { return e.id === s || e.name === s; });
  if (!ext) {
    ext = { id: 'ext-' + s.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-').slice(0, 40) + '-' + Math.random().toString(36).slice(2, 5), name: s, createdAt: Date.now() };
    store.externalRuns.push(ext);
  }
  return { runId: ext.id, created: true, task: '' };
}

// ── 导出 ────────────────────────────────────────────────────────────────────

function csvCell(v) {
  let s = String(v == null ? '' : v);
  // 防 CSV / Excel 公式注入：以 = + @ 或控制字符开头的单元格会被表格软件当公式执行。
  // 合法负数（-0.5）不转义，避免把正常数值变成文本、破坏后续分析。
  if (/^[=+@\t\r]/.test(s) || (/^-/.test(s) && !/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(s))) s = "'" + s;
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function exportData(result, format) {
  if (!result || !result.tableRows) throw new Error('没有可导出的分析结果');
  const headers = ['Field'].concat(result.runs.map(function (r) { return r.name; }));
  if (format === 'csv') {
    const lines = [headers.map(csvCell).join(',')];
    result.tableRows.forEach(function (row) { lines.push([row.label].concat(row.values).map(csvCell).join(',')); });
    lines.push('');
    lines.push(['Group', 'Metric', 'n', 'mean', 'sd', '95% CI low', '95% CI high'].map(csvCell).join(','));
    Object.keys(result.stats).forEach(function (metric) {
      result.stats[metric].forEach(function (g) {
        const s = g.stats;
        lines.push([g.label, metric, s.n, s.mean, s.sd, s.ci95 ? s.ci95[0] : '', s.ci95 ? s.ci95[1] : ''].map(csvCell).join(','));
      });
    });
    return lines.join('\n');
  }
  const md = [];
  md.push('# 多实验对比报告');
  md.push('');
  md.push('- 生成时间：' + new Date(result.meta.generatedAt).toLocaleString('zh-CN'));
  md.push('- 运行数：' + result.meta.runCount + ' · 分组维度：' + result.meta.groupLabel + ' · 主指标：' + result.meta.primary);
  md.push('');
  if (result.warnings.length) {
    md.push('## ⚠ 可比性与数据质量');
    md.push('');
    result.warnings.forEach(function (w) { md.push('- ' + w); });
    md.push('');
  }
  md.push('## 汇总统计');
  md.push('');
  md.push('| 分组 | 指标 | n | mean | sd | 95% CI |');
  md.push('|---|---|---:|---:|---:|---|');
  Object.keys(result.stats).forEach(function (metric) {
    result.stats[metric].forEach(function (g) {
      const s = g.stats;
      md.push('| ' + g.label + ' | ' + metric + ' | ' + s.n + ' | ' +
        (s.mean == null ? '—' : s.mean.toFixed(4)) + ' | ' + (s.sd == null ? '—' : s.sd.toFixed(4)) + ' | ' +
        (s.ci95 ? '[' + s.ci95[0].toFixed(4) + ', ' + s.ci95[1].toFixed(4) + ']' : '—') + ' |');
    });
  });
  md.push('');
  if (result.tests.length) {
    md.push('## 显著性检验');
    md.push('');
    md.push('| 对比 | 检验 | p | Bonferroni α | 显著 | 效应量 |');
    md.push('|---|---|---:|---:|---|---:|');
    result.tests.forEach(function (t) {
      md.push('| ' + t.groupA + ' vs ' + t.groupB + ' | ' + (t.result.test || '—') + ' | ' +
        (t.result.p == null ? '—' : t.result.p.toFixed(5)) + ' | ' + t.alphaCorrected.toFixed(5) + ' | ' +
        (t.significant ? '是' : '否') + ' | ' + (t.result.cohensD == null ? '—' : t.result.cohensD.toFixed(3)) + ' |');
    });
    md.push('');
  }
  md.push('## 运行配置对比');
  md.push('');
  md.push('| 字段 | ' + result.runs.map(function (r) { return r.name; }).join(' | ') + ' |');
  md.push('|---' + result.runs.map(function () { return '|---'; }).join('') + '|');
  result.tableRows.forEach(function (row) { md.push('| ' + row.label + ' | ' + row.values.join(' | ') + ' |'); });
  md.push('');
  md.push('## 方法与口径');
  md.push('');
  Object.keys(result.methods).forEach(function (k) {
    if (k === 'sources') return;
    md.push('- **' + k + '**：' + result.methods[k]);
  });
  md.push('');
  md.push('## 已知限制');
  md.push('');
  md.push('- 训练日志自动指标（loss / steps / runtime）不是留出集精度；精度类结论必须基于录入或导入的评估指标。');
  md.push('- 配对 t 检验假设差值近似正态、配对独立；n<3 时 p 值与置信区间仅作参考。');
  md.push('- 雷达图为归一化轮廓，面积不代表综合得分；精确比较请看柱状图与汇总表。');
  md.push('');
  md.push('<small>参考：' + result.methods.sources.join(' · ') + '</small>');
  return md.join('\n');
}

// ── 路由 ────────────────────────────────────────────────────────────────────

async function handleExperimentReportRequest(req, res) {
  const pathname = (req.pathname || req.url.split('?')[0]);
  if (pathname !== '/api/experiment-report' && !pathname.startsWith('/api/experiment-report/')) return false;
  const sub = pathname.slice('/api/experiment-report'.length).replace(/^\//, '') || '';
  let body = {};
  if (req.method === 'POST') {
    const parsed = await readBody(req);
    if (parsed.error) return sendJson(res, 400, { ok: false, error: parsed.error });
    body = parsed.data || {};
  }
  try {
    if (sub === 'runs' && req.method === 'GET') {
      const runs = listRuns();
      const rows = metricRowsFor(runs);
      return sendJson(res, 200, { ok: true, runs: runs, catalog: metricCatalog(rows), metricCount: readMetrics().entries.length });
    }
    if (sub === 'metrics' && req.method === 'GET') {
      const store = readMetrics();
      return sendJson(res, 200, { ok: true, entries: store.entries, externalRuns: store.externalRuns, runs: listRuns().map(function (r) { return { id: r.id, name: r.name, task: r.task }; }) });
    }
    if (sub === 'metrics' && req.method === 'POST') {
      const items = Array.isArray(body.entries) ? body.entries : [body];
      const store = withMetrics(function (store) {
      const added = [];
      items.forEach(function (it) {
        if (!it.runId || !it.name || num(it.value) == null) return;
        const rec = {
          id: newId('met'), runId: String(it.runId), name: String(it.name).trim(),
          value: num(it.value), unit: String(it.unit || ''),
          task: String(it.task || ''), seed: it.seed === '' || it.seed == null ? null : num(it.seed),
          n: it.n === '' || it.n == null ? null : num(it.n),
          direction: it.direction || guessDirection(it.name),
          source: it.source || 'manual', note: String(it.note || ''), createdAt: Date.now()
        };
        store.entries.push(rec); added.push(rec);
      });
      return store;
      });
      return sendJson(res, 200, { ok: true, added: added.length, entries: store.entries });
    }
    if (sub === 'metrics/delete' && req.method === 'POST') {
      const ids = Array.isArray(body.ids) ? body.ids : [body.id];
      const store = withMetrics(function (store) {
        store.entries = store.entries.filter(function (e) { return ids.indexOf(e.id) < 0; });
        return store;
      });
      return sendJson(res, 200, { ok: true, entries: store.entries });
    }
    if (sub === 'metrics/import' && req.method === 'POST') {
      const records = parseImport(body.text);
      const added = [];
      const store = withMetrics(function (store) {
      const extMeta = {};
      records.forEach(function (r) {
        const ref = resolveRunRef(r.run, store);
        if (!ref) return;
        if (r.model || r.engine || r.task) {
          const ext = (store.externalRuns || []).find(function (e) { return e.id === ref.runId; });
          if (ext) {
            if (r.model) ext.model = String(r.model);
            if (r.engine) ext.engine = String(r.engine);
            if (r.task) { ext.task = String(r.task); ext.dataset = String(r.task); }
          }
          extMeta[ref.runId] = true;
        }
        added.push({
          id: newId('met'), runId: ref.runId, name: String(r.metric).trim(),
          value: num(r.value), unit: String(r.unit || ''), task: String(r.task || ''),
          seed: r.seed === '' || r.seed == null ? null : num(r.seed),
          n: r.n === '' || r.n == null ? null : num(r.n),
          direction: String(r.direction || '').toLowerCase() === 'lower' ? 'lower' : guessDirection(r.metric),
          source: 'import', note: String(r.note || ''), createdAt: Date.now()
        });
      });
      store.entries = store.entries.concat(added);
      return store;
      });
      return sendJson(res, 200, { ok: true, added: added.length, externalRuns: store.externalRuns.length, parsed: records.length });
    }
    if (sub === 'analyze' && req.method === 'POST') {
      const result = analyze(body);
      return sendJson(res, 200, result);
    }
    // ── AI 叙述报告：数值仍由 analyze() 确定性计算，模型只写叙述 ──
    if (sub === 'ai/engines' && req.method === 'GET') {
      return sendJson(res, 200, { ok: true, engines: AI.listEngines() });
    }
    if (sub === 'ai/generate' && req.method === 'POST') {
      const aiFile = body.reportId ? reportFile(body.reportId) : null;
      if (body.reportId && !aiFile) return badId(res);
      const source = body.result || (aiFile ? (readJson(aiFile, {}) || {}).result : null);
      if (!source) return sendJson(res, 400, { ok: false, error: '缺少分析结果（result 或 reportId）' });
      const task = AI.startAiTask(source, { model: body.model, style: body.style });
      return sendJson(res, 200, Object.assign({ ok: true }, task));
    }
    const aiTaskMatch = sub.match(/^ai\/task\/([^/]+)$/);
    if (aiTaskMatch && req.method === 'GET') {
      const t = AI.getAiTask(aiTaskMatch[1]);
      if (!t) return sendJson(res, 404, { ok: false, error: 'AI task not found' });
      return sendJson(res, 200, { ok: true, task: t });
    }
    if (sub === 'reports' && req.method === 'GET') {
      const list = fs.readdirSync(REPORTS_DIR).filter(function (f) { return f.endsWith('.json'); })
        .map(function (f) { return readJson(path.join(REPORTS_DIR, f), null); })
        .filter(Boolean)
        .map(function (r) { return { id: r.id, name: r.name, createdAt: r.createdAt, runCount: (r.config && r.config.runIds || []).length, groupBy: r.config && r.config.groupBy, hasAi: !!(r.ai && r.ai.markdown) }; })
        .sort(function (a, b) { return b.createdAt - a.createdAt; });
      return sendJson(res, 200, { ok: true, reports: list });
    }
    if (sub === 'reports' && req.method === 'POST') {
      const id = newId('rep');
      const report = {
        id: id, name: String(body.name || ('实验报告 ' + new Date().toLocaleString('zh-CN'))),
        createdAt: Date.now(), config: body.config || {}, note: String(body.note || ''),
        result: body.result || null,
        ai: body.ai ? { markdown: String(body.ai.markdown || ''), model: String(body.ai.model || ''), generatedAt: body.ai.generatedAt || null } : null
      };
      writeJson(path.join(REPORTS_DIR, id + '.json'), report);
      return sendJson(res, 200, { ok: true, report: { id: report.id, name: report.name, createdAt: report.createdAt } });
    }
    const repMatch = sub.match(/^reports\/([^/]+)$/);
    if (repMatch && req.method === 'GET') {
      const f = reportFile(repMatch[1]);
      if (!f) return badId(res);
      const r = readJson(f, null);
      if (!r) return sendJson(res, 404, { ok: false, error: 'report not found' });
      return sendJson(res, 200, { ok: true, report: r });
    }
    if (sub === 'reports/delete' && req.method === 'POST') {
      const f = reportFile(body.id);
      if (!f) return badId(res);
      let deleted = false;
      try { if (fs.existsSync(f)) { fs.unlinkSync(f); deleted = true; } } catch (e) { /* 权限等，交由 deleted=false 表达 */ }
      return sendJson(res, 200, { ok: true, deleted: deleted });
    }
    if (sub === 'export' && req.method === 'POST') {
      const expFile = body.reportId ? reportFile(body.reportId) : null;
      if (body.reportId && !expFile) return badId(res);
      const text = exportData(body.result || (expFile ? (readJson(expFile, {}) || {}).result : null), body.format || 'csv');
      return sendJson(res, 200, { ok: true, text: text, filename: 'experiment-report.' + (body.format === 'markdown' ? 'md' : 'csv') });
    }
    return sendJson(res, 404, { ok: false, error: 'unknown endpoint: ' + sub });
  } catch (e) {
    return sendJson(res, 500, { ok: false, error: e.message });
  }
}

module.exports = { handleExperimentReportRequest, analyze, parseImport, exportData, AI, reportFile };
