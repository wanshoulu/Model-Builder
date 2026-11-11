/**
 * 评估指标内核（纯函数，无 IO / 无外部依赖，可离线单测）。
 *
 * 口径来源（均已实调原文，见 data/ui-review/eval-methods.md）：
 *  - LLaMA-Factory Eval：训练后用独立评估流程打分；NLG 评估用 do_predict +
 *    predict_with_generate 产出 generated_predictions.jsonl，再算 BLEU/ROUGE。
 *  - lm-evaluation-harness Metrics：acc = mean(per-sample 0/1)，exact_match 的
 *    aggregation = mean，corpus 级指标先收集逐样本中间结果再聚合；--log_samples 保留逐样本结果。
 *  - HF evaluate F1 metric card：多分类必须显式指定 average；macro 不加权，
 *    weighted 按 support 加权，micro 全局计 TP/FP/FN；默认 average='binary' 是多分类的陷阱。
 *  - SQuAD / HF evaluate exact_match：归一化（小写、去标点、去冠词、折叠空白）后字符串相等；
 *    token F1 为 bag-of-tokens 的 F1，多参考答案取最大值，再对样本求 macro 平均。
 *
 * 平台自己的约束（不伪装精度）：
 *  - 无效/空预测不被静默丢弃：accuracy 分母始终是评估集全部样本，另给 coverage。
 *  - 所有比例都给 Wilson 95% CI（二项比例的合适口径，不是对连续值用 t 分布）。
 *  - 生成类指标与分类类指标分开报告，不混成一个「精度」。
 */

'use strict';

const crypto = require('node:crypto');

const Z95 = 1.959963984540054; // norm.ppf(0.975)

/** SQuAD 风格归一化：NFKC → 小写 → 去标点/冠词 → 折叠空白 */
function normalizeAnswer(s) {
  let t = String(s == null ? '' : s).normalize('NFKC').toLowerCase();
  t = t.replace(/[\p{P}\p{S}]/gu, ' ');          // 中英文标点、符号一并去掉
  t = t.replace(/\b(a|an|the)\b/g, ' ');
  return t.replace(/\s+/g, ' ').trim();
}

/** token 化：CJK 逐字切分 + 其余按空白切词，保证中英混排的 token F1 有定义 */
function tokenize(s) {
  const t = normalizeAnswer(s);
  if (!t) return [];
  return t.replace(/([\u3400-\u9fff\u3040-\u30ff])/g, ' $1 ')
    .trim().split(/\s+/).filter(Boolean);
}

function bagF1(goldTokens, predTokens) {
  if (!goldTokens.length && !predTokens.length) return 1;
  if (!goldTokens.length || !predTokens.length) return 0;
  const counts = new Map();
  goldTokens.forEach(function (t) { counts.set(t, (counts.get(t) || 0) + 1); });
  let common = 0;
  predTokens.forEach(function (t) {
    const c = counts.get(t) || 0;
    if (c > 0) { common += 1; counts.set(t, c - 1); }
  });
  const precision = common / predTokens.length;
  const recall = common / goldTokens.length;
  return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
}

/** ROUGE-L：LCS 长度-based F1（句子级，不做摘要级截断） */
function rougeL(goldTokens, predTokens) {
  if (!goldTokens.length && !predTokens.length) return 1;
  if (!goldTokens.length || !predTokens.length) return 0;
  const n = goldTokens.length, m = predTokens.length;
  // 滚动数组，避免 O(n*m) 内存；评估集通常几十~几千条，逐样本调用
  let prev = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    const cur = new Array(m + 1).fill(0);
    for (let j = 1; j <= m; j++) {
      cur[j] = goldTokens[i - 1] === predTokens[j - 1]
        ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    prev = cur;
  }
  const lcs = prev[m];
  const p = lcs / m, r = lcs / n;
  return p + r > 0 ? 2 * p * r / (p + r) : 0;
}

/** Wilson score interval：二项比例的正确小样本区间（p=0/1 时区间仍非退化） */
function wilsonInterval(k, n, z) {
  z = z || Z95;
  if (!n) return { low: null, high: null, method: 'Wilson score, 95%' };
  const p = k / n, z2 = z * z, denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
  return {
    low: Math.max(0, center - half), high: Math.min(1, center + half),
    method: 'Wilson score interval, z=1.95996 (95%)'
  };
}

/**
 * record-level bootstrap CI（2026-11-11）：决策模型一条记录含多个问题（n=预测单元数
 * > 独立记录数），逐单元 Wilson 区间会偏窄——同一记录的问题高度相关。按【记录】重采样
 * （默认 1000 次），每次对采样记录的全部预测单元聚合正确率，取 2.5/97.5 分位。
 * rows 带 groupId（如 "0:topic" 的 "0"）时启用；无分组信息返回 null（调用方回退 Wilson）。
 */
function bootstrapRecordCI(rows, iterations, seed) {
  iterations = Math.min(Math.max(iterations || 1000, 200), 10000);
  // 分组：id 形如 "<record>:<field>"；兼容原始 id（无冒号时每行自成一组，等价单元 bootstrap）
  const groups = new Map();
  for (const r of rows) {
    const gid = String(r && r.id != null ? r.id : '').split(':')[0] || 'anon';
    if (!groups.has(gid)) groups.set(gid, []);
    groups.get(gid).push(r && r.correct === true ? 1 : 0);
  }
  const garr = Array.from(groups.values());
  if (garr.length < 2) return null;
  // 确定性伪随机（mulberry32）——同一数据每次算出的 CI 一致（可复现）
  let s = (seed == null ? 42 : seed) >>> 0;
  const rnd = function () { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const accs = [];
  for (let i = 0; i < iterations; i++) {
    let sum = 0, cnt = 0;
    for (let g = 0; g < garr.length; g++) {
      const pick = garr[Math.floor(rnd() * garr.length)];
      for (let k = 0; k < pick.length; k++) { sum += pick[k]; cnt++; }
    }
    if (cnt) accs.push(sum / cnt);
  }
  accs.sort(function (a, b) { return a - b; });
  const q = function (p) { return accs[Math.min(accs.length - 1, Math.max(0, Math.floor(p * accs.length)))]; };
  return { low: q(0.025), high: q(0.975), method: 'record-level bootstrap, ' + iterations + ' resamples (95%)', nRecords: garr.length };
}

function mean(arr) { return arr.length ? arr.reduce(function (a, b) { return a + b; }, 0) / arr.length : null; }

/**
 * 分类指标。rows: [{ gold: string, prediction: string|null, weight?: number }]
 * prediction 为 null/'' 视为弃权（计入 coverage，不算正确）。
 */
function classificationMetrics(rows) {
  const n = rows.length;
  const labelsSet = new Set();
  const support = new Map(), tp = new Map(), fp = new Map(), fn = new Map();
  let correct = 0, covered = 0;
  rows.forEach(function (r) {
    const g = String(r.gold == null ? '' : r.gold);
    const p = r.prediction == null ? null : String(r.prediction);
    labelsSet.add(g);
    support.set(g, (support.get(g) || 0) + 1);
    if (p !== null && p !== '') {
      covered += 1;
      labelsSet.add(p);
      if (p === g) { correct += 1; tp.set(g, (tp.get(g) || 0) + 1); }
      else {
        fp.set(p, (fp.get(p) || 0) + 1);
        fn.set(g, (fn.get(g) || 0) + 1);
      }
    } else {
      fn.set(g, (fn.get(g) || 0) + 1);   // 弃权 = 漏判，保证 micro recall 口径一致
    }
  });
  const labels = Array.from(labelsSet).sort();
  const perClass = labels.map(function (L) {
    const t = tp.get(L) || 0, f = fp.get(L) || 0, fnn = fn.get(L) || 0;
    const precision = t + f > 0 ? t / (t + f) : 0;
    const recall = t + fnn > 0 ? t / (t + fnn) : 0;
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    return { label: L, precision: precision, recall: recall, f1: f1, support: support.get(L) || 0, tp: t, fp: f, fn: fnn };
  });
  const totalSupport = Array.from(support.values()).reduce(function (a, b) { return a + b; }, 0) || n;
  const macro = { precision: mean(perClass.map(c => c.precision)), recall: mean(perClass.map(c => c.recall)), f1: mean(perClass.map(c => c.f1)) };
  const tSum = Array.from(tp.values()).reduce(function (a, b) { return a + b; }, 0);
  const fSum = Array.from(fp.values()).reduce(function (a, b) { return a + b; }, 0);
  const fnSum = Array.from(fn.values()).reduce(function (a, b) { return a + b; }, 0);
  const microP = tSum + fSum > 0 ? tSum / (tSum + fSum) : 0;
  const microR = tSum + fnSum > 0 ? tSum / (tSum + fnSum) : 0;
  const micro = { precision: microP, recall: microR, f1: microP + microR > 0 ? 2 * microP * microR / (microP + microR) : 0 };
  const weighted = {};
  ['precision', 'recall', 'f1'].forEach(function (k) {
    weighted[k] = totalSupport
      ? perClass.reduce(function (a, c) { return a + c[k] * c.support; }, 0) / totalSupport : null;
  });
  // 混淆矩阵：行 = gold，列 = prediction（弃权单列 __ABSTAIN__）
  const index = new Map(labels.map(function (L, i) { return [L, i]; }));
  const matrix = labels.map(function () { return new Array(labels.length + 1).fill(0); });
  rows.forEach(function (r) {
    const g = String(r.gold == null ? '' : r.gold);
    const p = r.prediction == null ? null : String(r.prediction);
    const ri = index.get(g);
    const ci = p === null || p === '' ? labels.length : index.get(p);
    if (ri != null && ci != null) matrix[ri][ci] += 1;
  });
  const accuracy = n ? correct / n : null;
  return {
    n: n, covered: covered, coverage: n ? covered / n : null, correct: correct,
    accuracy: accuracy, accuracyCovered: covered ? correct / covered : null,
    ci95: wilsonInterval(correct, n),
    macro: macro, micro: micro, weighted: weighted,
    perClass: perClass, confusion: { labels: labels.concat(['__ABSTAIN__']), matrix: matrix },
    method: 'accuracy = correct/n（n 含弃权样本）；macro/micro/weighted P·R·F1 按 sklearn 定义；CI 为 Wilson 95%'
  };
}

/**
 * 生成类指标。rows: [{ gold: string|string[], prediction: string|null }]
 * 多参考答案取逐样本最大值（SQuAD / HF evaluate 口径），再对样本求 macro 平均。
 */
function generationMetrics(rows) {
  const perSample = rows.map(function (r, i) {
    const refs = (Array.isArray(r.gold) ? r.gold : [r.gold]).map(function (g) { return String(g == null ? '' : g); });
    const pred = r.prediction == null ? '' : String(r.prediction);
    const pt = tokenize(pred);
    const normPred = normalizeAnswer(pred);
    let em = 0, bestF1 = 0, bestRouge = 0;
    refs.forEach(function (ref) {
      const rt = tokenize(ref);
      if (normPred && normalizeAnswer(ref) === normPred) em = 1;
      bestF1 = Math.max(bestF1, bagF1(rt, pt));
      bestRouge = Math.max(bestRouge, rougeL(rt, pt));
    });
    return { index: i, exactMatch: em, tokenF1: bestF1, rougeL: bestRouge, prediction: pred };
  });
  const covered = perSample.filter(function (p) { return p.prediction.trim() !== ''; }).length;
  const accuracy = mean(perSample.map(p => p.exactMatch));
  return {
    n: rows.length, covered: covered, coverage: rows.length ? covered / rows.length : null,
    exactMatch: accuracy, tokenF1: mean(perSample.map(p => p.tokenF1)),
    rougeL: mean(perSample.map(p => p.rougeL)),
    ci95: wilsonInterval(perSample.reduce(function (a, p) { return a + p.exactMatch; }, 0), rows.length),
    perSample: perSample,
    method: 'SQuAD normalize 后 exact_match；token F1 = bag-of-tokens F1（多参考答案取 max）；ROUGE-L = LCS F1；均为逐样本 macro 平均'
  };
}

/** 稳定序列化：对象键排序，保证字段书写顺序不同不影响内容指纹 */
function stableStringify(v) {
  if (v === null || v === undefined || typeof v !== 'object') return JSON.stringify(v == null ? null : v);
  if (Array.isArray(v)) return '[' + v.map(stableStringify).join(',') + ']';
  return '{' + Object.keys(v).sort().map(function (k) {
    return JSON.stringify(k) + ':' + stableStringify(v[k]);
  }).join(',') + '}';
}

/** 内容指纹：用于评估集版本固定与训练/评估集泄漏检测 */
function sampleFingerprint(sample) {
  let v;
  const s = sample || {};
  if (s.state && s.questions) {
    // kev decision：state + 每个问题的类型/选项/label 参与指纹（描述文案变化也算改动）
    v = stableStringify({ state: s.state, questions: s.questions });
  } else if (Array.isArray(s.messages)) {
    v = stableStringify(s.messages);
  } else if (Array.isArray(s.conversations)) {
    v = stableStringify(s.conversations);
  } else {
    v = stableStringify(s);
  }
  return crypto.createHash('sha256').update(v.normalize('NFKC')).digest('hex');
}
function evalSetFingerprint(samples) {
  const hashes = (samples || []).map(sampleFingerprint).sort();
  return {
    count: hashes.length,
    sha256: crypto.createHash('sha256').update(hashes.join('\n')).digest('hex')
  };
}
/** 训练/评估集重复率：交集 / 评估集样本数 */
function leakage(trainSamples, evalSamples) {
  const train = new Set((trainSamples || []).map(sampleFingerprint));
  const ev = (evalSamples || []).map(sampleFingerprint);
  const dup = ev.filter(function (h) { return train.has(h); }).length;
  return { evalCount: ev.length, duplicated: dup, ratio: ev.length ? dup / ev.length : null };
}

/** 把任意来源的预测结果对齐到统一行：支持按 index 或 id 对齐 */
function alignPredictions(samples, predictions, goldOf, predOf) {
  const byId = new Map();
  (predictions || []).forEach(function (p, i) {
    const key = p && p.id != null ? String(p.id) : String(i);
    byId.set(key, p);
  });
  return (samples || []).map(function (s, i) {
    const p = byId.get(String(s && s.id != null ? s.id : i)) || predictions[i] || null;
    return {
      index: i, gold: goldOf(s, p), prediction: predOf(s, p),
      type: (s && s.type) || (p && p.type) || null
    };
  });
}

module.exports = {
  Z95, normalizeAnswer, tokenize, bagF1, rougeL, wilsonInterval, bootstrapRecordCI,
  classificationMetrics, generationMetrics,   sampleFingerprint, evalSetFingerprint, stableStringify,
  leakage, alignPredictions, mean
};
