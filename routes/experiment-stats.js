/**
 * 实验报告统计内核 — 纯函数，无 IO，可独立单测
 *
 * 设计（2026-11-10 对标调研后的取舍）：
 *  1. 多实验对比对标 W&B Workspace「Group by → mean 线 + variance 阴影」、
 *     ClearML「Last/Min/Max 汇总表 + 叠加曲线 + CSV 导出」、
 *     MLflow「Table view 排序 + Parallel coordinates」、
 *     W&B Run Comparer「Diff only + 不可比任务警告」。
 *  2. 统计口径对标学术报告规范：跨 seed 必须给 n / mean / sample sd / 95% CI，
 *     小样本用 t 分布临界值而不是 1.96；配对检验 + Bonferroni 校正 + Cohen's d。
 *  3. 直方图 bin 规则严格照 numpy.histogram_bin_edges：
 *       h_fd = 2·IQR / n^(1/3)，h_sturges = range / (log2(n)+1)，
 *       'auto' = min(h_fd, h_sturges)；分组比较时所有组共享同一组 bin 边界。
 *  4. 雷达图不做「面积代表综合实力」的暗示：所有轴统一 min-max 归一化到 [0,1]、
 *     固定轴序、保留原始值与单位、≤3 组叠加，并同时提供更精确的柱状图。
 *  5. 曲线对齐：不同 run 的 step 数不同，直接叠加会误导；提供 step / epoch /
 *     logging index / normalized progress 四种 x 口径，聚合时禁止外推。
 *
 * 已知限制（诚实列出）：
 *  - 配对 t 检验假设差值近似正态、配对相互独立；样本 n<3 时结论不可靠，界面会标注。
 *  - 95% CI 是均值的置信区间（t 分布），不是个体观测的散布；标准差单独给出。
 *  - 所有数值均为双精度浮点，样本量极小时 tCritical 采用二分求逆（精度 1e-10）。
 */

'use strict';

// ── 基础描述统计 ────────────────────────────────────────────────────────────

function toNumbers(a) {
  return (a || []).map(Number).filter(function (v) { return Number.isFinite(v); });
}

function mean(a) {
  const x = toNumbers(a);
  if (!x.length) return null;
  let s = 0;
  for (const v of x) s += v;
  return s / x.length;
}

/** 样本标准差（ddof = 1），n<2 返回 null —— 不把 0 冒充「无波动」 */
function sampleSd(a) {
  const x = toNumbers(a);
  if (x.length < 2) return null;
  const m = mean(x);
  let ss = 0;
  for (const v of x) ss += (v - m) * (v - m);
  return Math.sqrt(ss / (x.length - 1));
}

/** 线性插值分位数（与 numpy quantile(method='linear') 一致） */
function quantile(a, p) {
  const x = toNumbers(a).slice().sort(function (u, v) { return u - v; });
  if (!x.length) return null;
  if (p <= 0) return x[0];
  if (p >= 1) return x[x.length - 1];
  const idx = (x.length - 1) * p;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return x[lo];
  return x[lo] + (x[hi] - x[lo]) * (idx - lo);
}

// ── Gamma / 不完全 Beta / Student-t ──────────────────────────────────────────
// 参考 Numerical Recipes in C 的 Lanczos + continued fraction 实现，
// 用于精确计算 t 分布 CDF，而不是用 1.96 这类大样本近似。

function logGamma(z) {
  const G = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.984369578019572e-6, 1.5056327351493116e-7
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < G.length; i++) x += G[i] / (z + i + 1);
  const t = z + G.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function betacf(a, b, x) {
  const MAXIT = 300, EPS = 3e-14, FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** 正则化不完全 Beta 函数 I_x(a,b) */
function betaI(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) +
    a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
  return 1 - bt * betacf(b, a, 1 - x) / b;
}

/** Student-t 累积分布，df 可为非整数（Welch 检验需要） */
function studentTCdf(t, df) {
  if (!(df > 0)) return NaN;
  const x = df / (df + t * t);
  const half = 0.5 * betaI(df / 2, 0.5, x);
  return t > 0 ? 1 - half : half;
}

/** 双侧 (1-alpha) 的 t 临界值，二分求逆 */
function studentTCritical(df, alpha) {
  alpha = alpha == null ? 0.05 : alpha;
  if (!(df > 0)) return null;
  const target = 1 - alpha / 2;
  let lo = 0, hi = 1;
  while (studentTCdf(hi, df) < target && hi < 1e6) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (studentTCdf(mid, df) < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/** 一组观测的完整描述（含 95% CI） */
function describe(a) {
  const x = toNumbers(a);
  const n = x.length;
  if (!n) return { n: 0, mean: null, sd: null, sem: null, min: null, max: null, ci95: null, tCrit: null };
  const m = mean(x), sd = sampleSd(x), sem = sd == null ? null : sd / Math.sqrt(n);
  const tCrit = n >= 2 ? studentTCritical(n - 1, 0.05) : null;
  return {
    n: n, mean: m, sd: sd, sem: sem,
    min: Math.min.apply(null, x), max: Math.max.apply(null, x),
    median: quantile(x, 0.5), q1: quantile(x, 0.25), q3: quantile(x, 0.75),
    tCrit: tCrit,
    ci95: (sem != null && tCrit != null) ? [m - tCrit * sem, m + tCrit * sem] : null
  };
}

// ── 假设检验 ────────────────────────────────────────────────────────────────

/** 双尾 p 值（t 统计量 + 自由度） */
function twoSidedP(t, df) {
  if (!(df > 0) || !Number.isFinite(t)) return null;
  const p = 2 * (1 - studentTCdf(Math.abs(t), df));
  return Math.max(0, Math.min(1, p));
}

/**
 * 配对 t 检验（同一 seed / 同一任务下两组结果一一配对）
 * 返回 Cohen's d_z = mean(D)/sd(D)，差值以自身标准差为单位。
 */
function pairedTTest(x, y) {
  const a = toNumbers(x), b = toNumbers(y);
  const n = Math.min(a.length, b.length);
  if (n < 2) return { applicable: false, reason: '需至少 2 组可配对的观测', n: n };
  const d = [];
  for (let i = 0; i < n; i++) d.push(a[i] - b[i]);
  const mD = mean(d), sD = sampleSd(d);
  if (!(sD > 0)) return { applicable: false, reason: '配对差值标准差为 0，无法计算 t 统计量', n: n, meanDiff: mD };
  const t = mD / (sD / Math.sqrt(n));
  const df = n - 1;
  const tCrit = studentTCritical(df, 0.05);
  return {
    applicable: true, test: 'paired t-test', n: n, df: df, t: t,
    p: twoSidedP(t, df), meanDiff: mD, sdDiff: sD,
    ci95Diff: [mD - tCrit * sD / Math.sqrt(n), mD + tCrit * sD / Math.sqrt(n)],
    cohensD: mD / sD, effectSizeLabel: "Cohen's d_z"
  };
}

/**
 * Welch t 检验（无法配对时的默认；不假设方差齐性）
 * 效应量用合并样本标准差的 Cohen's d。
 */
function welchTTest(x, y) {
  const a = toNumbers(x), b = toNumbers(y);
  if (a.length < 2 || b.length < 2) return { applicable: false, reason: '两组各需至少 2 个观测', n: a.length + b.length };
  const m1 = mean(a), m2 = mean(b), s1 = sampleSd(a), s2 = sampleSd(b);
  const v1 = s1 * s1 / a.length, v2 = s2 * s2 / b.length;
  const denom = Math.sqrt(v1 + v2);
  if (!(denom > 0)) return { applicable: false, reason: '合并标准误为 0', n: a.length + b.length };
  const t = (m1 - m2) / denom;
  const df = Math.pow(v1 + v2, 2) / (v1 * v1 / (a.length - 1) + v2 * v2 / (b.length - 1));
  const pooled = Math.sqrt(((a.length - 1) * s1 * s1 + (b.length - 1) * s2 * s2) / (a.length + b.length - 2));
  return {
    applicable: true, test: 'Welch t-test (unpaired)', n: a.length + b.length,
    n1: a.length, n2: b.length, df: df, t: t, p: twoSidedP(t, df),
    meanDiff: m1 - m2, cohensD: pooled > 0 ? (m1 - m2) / pooled : null,
    effectSizeLabel: "Cohen's d (pooled)"
  };
}

/** 多重比较 Bonferroni 校正：返回校正后 α 与是否显著 */
function bonferroni(p, m, alpha) {
  alpha = alpha || 0.05;
  const mEff = Math.max(1, m || 1);
  return { alphaCorrected: alpha / mEff, significant: p != null && p < alpha / mEff, comparisons: mEff };
}

// ── 直方图 ──────────────────────────────────────────────────────────────────

/**
 * 计算 bin 边界（对齐 numpy.histogram_bin_edges）
 * method: 'auto'(min(h_fd,h_sturges)) | 'fd' | 'sturges' | 'sqrt' | 正整数
 */
function histogramEdges(allValues, method) {
  const x = toNumbers(allValues);
  if (!x.length) return { edges: [], method: 'empty', binWidth: null, n: 0 };
  const lo = Math.min.apply(null, x), hi = Math.max.apply(null, x);
  const range = hi - lo;
  if (range === 0) return { edges: [lo - 0.5, hi + 0.5], method: 'degenerate (single value)', binWidth: 1, n: x.length };
  const n = x.length;
  let bins, used;
  if (typeof method === 'number' && method > 0) {
    bins = Math.max(1, Math.min(80, Math.round(method)));
    used = 'manual (' + bins + ' bins)';
  } else {
    const iqr = quantile(x, 0.75) - quantile(x, 0.25);
    const hFd = iqr > 0 ? 2 * iqr / Math.cbrt(n) : Infinity;
    const hSturges = range / (Math.log2(n) + 1);
    const hSqrt = range / Math.sqrt(n);
    let h;
    const m = method || 'auto';
    if (m === 'fd') { h = hFd; used = 'Freedman–Diaconis (h = 2·IQR/n^(1/3))'; }
    else if (m === 'sturges') { h = hSturges; used = "Sturges (log2(n)+1)"; }
    else if (m === 'sqrt') { h = hSqrt; used = 'sqrt(n)'; }
    else { h = Math.min(hFd, hSturges); used = (h === hFd ? 'auto → Freedman–Diaconis' : 'auto → Sturges') + ' (min bin width)'; }
    if (!Number.isFinite(h) || h <= 0) { h = hSqrt > 0 ? hSqrt : range / 10; used += ' · IQR=0 fallback'; }
    bins = Math.ceil(range / h);
  }
  bins = Math.max(1, Math.min(60, Math.round(bins)));
  const w = range / bins;
  const edges = [];
  for (let i = 0; i <= bins; i++) edges.push(lo + i * w);
  return { edges: edges, method: used, binWidth: w, n: n, range: [lo, hi] };
}

/** 按给定边界计数；最后一个 bin 为闭区间（含最大值），与 numpy 一致 */
function histogramCounts(values, edges) {
  const x = toNumbers(values);
  const k = edges.length - 1;
  if (k < 1) return { counts: [], n: 0, outliers: 0 };
  const counts = new Array(k).fill(0);
  const w = edges[1] - edges[0];
  let outliers = 0;
  for (const v of x) {
    if (v < edges[0] || v > edges[k]) { outliers++; continue; }
    let idx = Math.floor((v - edges[0]) / w);
    if (idx >= k) idx = k - 1;   // 右端闭合
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  return { counts: counts, n: x.length - outliers, outliers: outliers };
}

/** 分组共享 bin 边界（numpy 官方示例：组间可比性依赖相同 edges） */
function sharedHistogram(groups, method) {
  const all = [];
  (groups || []).forEach(function (g) { all.push.apply(all, toNumbers(g.values)); });
  const meta = histogramEdges(all, method);
  const series = (groups || []).map(function (g) {
    const c = histogramCounts(g.values, meta.edges);
    return { label: g.label, color: g.color || null, counts: c.counts, n: c.n, outliers: c.outliers };
  });
  return {
    edges: meta.edges, binWidth: meta.binWidth, method: meta.method,
    totalN: meta.n, series: series,
    maxCount: Math.max.apply(null, [1].concat(series.reduce(function (acc, s) { return acc.concat(s.counts); }, [])))
  };
}

// ── 曲线 ────────────────────────────────────────────────────────────────────

/** TensorBoard / W&B 同款去偏置 EMA：s = α·s + (1-α)·x，除以权重消除冷启动偏差 */
function emaSmooth(values, alpha) {
  const x = toNumbers(values);
  if (!x.length || !(alpha > 0)) return x.slice();
  const out = [x[0]];
  let s = x[0], w = 1;
  for (let i = 1; i < x.length; i++) {
    s = s * alpha + x[i] * (1 - alpha);
    w = w * alpha + (1 - alpha);
    out.push(s / w);
  }
  return out;
}

/**
 * 把一条 lossHistory 转成指定 x 口径的点列。
 * mode: 'index' | 'step' | 'epoch' | 'progress'
 * 不伪造 x：缺 step 时明确返回 unavailable，由调用方决定降级。
 */
function buildSeries(hist, mode, totalSteps, totalEpochs) {
  const h = toNumbers(hist && hist.length ? hist.map(function (p) { return p.loss; }) : []);
  const pts = (hist || []).filter(function (p) { return Number.isFinite(Number(p.loss)); });
  if (!pts.length) return { points: [], available: false, reason: 'no loss points' };
  let xs, label, unavailable = null;
  if (mode === 'step') {
    const hasStep = pts.every(function (p) { return Number.isFinite(Number(p.step)); });
    if (!hasStep) {
      unavailable = '部分记录缺少 step 字段，已降级为 Logging Point Index';
      mode = 'index';
    } else {
      xs = pts.map(function (p) { return Number(p.step); });
      label = 'Training Step';
    }
  }
  if (mode === 'epoch') {
    xs = pts.map(function (p) { return Number(p.epoch) || 0; });
    label = 'Epoch';
  }
  if (mode === 'progress') {
    const maxStep = pts.reduce(function (m, p) { return Math.max(m, Number(p.step) || 0); }, 0);
    const denom = totalSteps || maxStep || (pts.length - 1) || 1;
    xs = pts.map(function (p, i) {
      const v = Number.isFinite(Number(p.step)) ? Number(p.step) : i;
      return Math.min(1, v / denom);
    });
    label = 'Normalized Training Progress (0–1)';
  }
  if (mode === 'index' || !xs) {
    xs = pts.map(function (p, i) { return i; });
    label = 'Logging Point Index';
    if (!unavailable) mode = 'index';
  }
  return {
    points: pts.map(function (p, i) { return { x: xs[i], y: Number(p.loss) }; }),
    xLabel: label, mode: mode, available: true, degraded: unavailable
  };
}

/** 线性插值；超出该 run 自身范围返回 null（禁止外推，避免伪造数据点） */
function interpolate(points, xq) {
  if (!points.length) return null;
  const sorted = points.slice().sort(function (a, b) { return a.x - b.x; });
  if (xq < sorted[0].x - 1e-9 || xq > sorted[sorted.length - 1].x + 1e-9) return null;
  for (let i = 0; i < sorted.length; i++) {
    if (Math.abs(sorted[i].x - xq) < 1e-9) return sorted[i].y;
    if (sorted[i].x > xq) {
      const p0 = sorted[i - 1], p1 = sorted[i];
      if (!p0) return null;
      const w = (xq - p0.x) / (p1.x - p0.x);
      return p0.y + (p1.y - p0.y) * w;
    }
  }
  return sorted[sorted.length - 1].y;
}

/**
 * 分组聚合曲线（W&B group-by 范式）：在公共 x 网格上逐 run 插值 → mean / sd / 95% CI。
 * n=1 的网格点不给置信带（只有一条 run 时不存在「组间波动」）。
 */
function aggregateSeries(seriesList, gridCount) {
  const all = (seriesList || []).filter(function (s) { return s && s.points && s.points.length; });
  if (!all.length) return { grid: [], xLabel: '', note: '无可聚合曲线' };
  let xMin = Infinity, xMax = -Infinity;
  all.forEach(function (s) {
    s.points.forEach(function (p) { xMin = Math.min(xMin, p.x); xMax = Math.max(xMax, p.x); });
  });
  if (!Number.isFinite(xMin)) return { grid: [], xLabel: all[0].xLabel || '', note: '' };
  if (xMin === xMax) xMax = xMin + 1;
  const K = Math.max(10, Math.min(200, gridCount || 60));
  const grid = [];
  for (let i = 0; i <= K; i++) {
    const x = xMin + (xMax - xMin) * i / K;
    const vals = [];
    all.forEach(function (s) {
      const v = interpolate(s.points, x);
      if (v != null && Number.isFinite(v)) vals.push(v);
    });
    if (!vals.length) continue;
    const st = describe(vals);
    grid.push({
      x: x, n: st.n, mean: st.mean, sd: st.sd,
      ciLo: st.ci95 ? st.ci95[0] : null,
      ciHi: st.ci95 ? st.ci95[1] : null,
      sdLo: st.sd != null ? st.mean - st.sd : null,
      sdHi: st.sd != null ? st.mean + st.sd : null
    });
  }
  return {
    grid: grid, xLabel: all[0].xLabel || '',
    note: '网格点在每条 run 自身 x 范围内线性插值，不做外推；n 为该点参与聚合的 run 数。'
  };
}

// ── 归一化（雷达图 / 平行坐标）──────────────────────────────────────────────

/**
 * min-max 归一化到 [0,1]；direction='lower' 时取反（loss 等越小越好）。
 * 全组相同值 → 0.5 中性，并标记 degenerate，避免除零造出「满分」假象。
 */
function normalizeMetric(values, direction) {
  const x = toNumbers(values);
  if (!x.length) return { scores: [], min: null, max: null, degenerate: true };
  const lo = Math.min.apply(null, x), hi = Math.max.apply(null, x);
  if (hi === lo) return {
    scores: x.map(function () { return 0.5; }), min: lo, max: hi, degenerate: true,
    note: '该指标所有组取值相同，统一记为中性 0.5'
  };
  const scores = x.map(function (v) {
    const s = (v - lo) / (hi - lo);
    return direction === 'lower' ? 1 - s : s;
  });
  return { scores: scores, min: lo, max: hi, degenerate: false };
}

module.exports = {
  // 描述统计
  mean: mean, sampleSd: sampleSd, quantile: quantile, describe: describe, toNumbers: toNumbers,
  // 分布函数
  logGamma: logGamma, betaI: betaI, studentTCdf: studentTCdf, studentTCritical: studentTCritical,
  // 检验
  pairedTTest: pairedTTest, welchTTest: welchTTest, bonferroni: bonferroni, twoSidedP: twoSidedP,
  // 直方图
  histogramEdges: histogramEdges, histogramCounts: histogramCounts, sharedHistogram: sharedHistogram,
  // 曲线
  emaSmooth: emaSmooth, buildSeries: buildSeries, interpolate: interpolate, aggregateSeries: aggregateSeries,
  // 归一化
  normalizeMetric: normalizeMetric
};
