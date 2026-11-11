/**
 * 实验报告统计内核数值校验
 * 运行： node tests/experiment-stats.test.js
 *
 * 期望值来源：
 *  - t 临界值 / t 检验：标准 t 分布表（df=4 → 2.776445，df=9 → 2.262157，df→∞ → 1.959964）
 *  - 直方图：numpy.histogram_bin_edges 官方示例 arr=[0,0,0,1,2,3,3,4,5] auto → [0,1,2,3,4,5]
 *  - Welch t：手算 [1..5] vs [3..7] → t=-2, df=8, p≈0.0805
 */
'use strict';
const assert = require('node:assert/strict');
const S = require('../routes/experiment-stats');

let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); console.log('  PASS  ' + name); pass++; }
  catch (e) { console.log('  FAIL  ' + name + ' → ' + e.message); fail++; }
}
function close(a, b, eps, msg) {
  assert.ok(Math.abs(a - b) <= (eps || 1e-6), (msg || '') + ' got ' + a + ' expected ' + b);
}

console.log('\n=== 描述统计 ===');
check('mean / sample sd (ddof=1)', function () {
  close(S.mean([1, 2, 3, 4, 5]), 3, 1e-12);
  close(S.sampleSd([1, 2, 3, 4, 5]), Math.SQRT2 * 1.11803398875, 1e-9); // =1.58113883
  assert.equal(S.sampleSd([7]), null, 'n=1 时标准差必须是 null，不能伪装成 0');
});
check('quantile 线性插值与 numpy 一致', function () {
  close(S.quantile([0, 0, 0, 1, 2, 3, 3, 4, 5], 0.25), 0, 1e-12);
  close(S.quantile([0, 0, 0, 1, 2, 3, 3, 4, 5], 0.75), 3, 1e-12);
  close(S.quantile([1, 2, 3, 4], 0.5), 2.5, 1e-12);
});
check('describe 95% CI 用 t 临界值（非 1.96）', function () {
  const d = S.describe([1, 2, 3, 4, 5]);
  close(d.tCrit, 2.7764451051977987, 1e-6);
  close(d.ci95[0], 3 - 2.7764451051977987 * Math.SQRT1_2, 1e-6);
  close(d.ci95[1], 3 + 2.7764451051977987 * Math.SQRT1_2, 1e-6);
  assert.ok(d.ci95[1] - d.ci95[0] > (3 + 1.96 * Math.SQRT1_2) - (3 - 1.96 * Math.SQRT1_2));
});

console.log('\n=== t 分布 / 假设检验 ===');
check('t 临界值对照统计表', function () {
  close(S.studentTCritical(4), 2.776445105, 1e-5);
  close(S.studentTCritical(9), 2.262157163, 1e-5);
  close(S.studentTCritical(1e6), 1.959963985, 1e-4);
});
check('t CDF 对称性 / 单调性', function () {
  close(S.studentTCdf(0, 5), 0.5, 1e-10);
  close(S.studentTCdf(2.776445105, 4), 0.975, 1e-6);
  close(S.studentTCdf(-2.776445105, 4), 0.025, 1e-6);
});
check('betaI 边界', function () {
  assert.equal(S.betaI(2, 3, 0), 0);
  assert.equal(S.betaI(2, 3, 1), 1);
  close(S.betaI(2, 3, 0.5), 0.6875, 1e-10);
});
check('paired t-test 手算', function () {
  const r = S.pairedTTest([5, 6, 7, 8, 9], [4, 5, 5, 6, 7]); // d=[1,1,2,2,2]
  assert.equal(r.applicable, true);
  close(r.meanDiff, 1.6, 1e-12);
  close(r.sdDiff, 0.5477225575, 1e-8);
  close(r.t, 1.6 / (0.5477225575 / Math.sqrt(5)), 1e-8);
  assert.ok(r.p > 0 && r.p < 0.01, 'p 应显著小于 0.01，got ' + r.p);
  close(r.cohensD, 1.6 / 0.5477225575, 1e-8);
});
check('paired t-test 差值全同时拒绝出结论', function () {
  const r = S.pairedTTest([2, 3, 4], [1, 2, 3]);
  assert.equal(r.applicable, false);
  assert.equal(r.p, undefined);
});
check('Welch t-test 手算（不假设等方差）', function () {
  const r = S.welchTTest([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]);
  close(r.t, -2, 1e-10);
  close(r.df, 8, 1e-10);
  close(r.p, 0.080516237, 1e-6);
});
check('Bonferroni 校正', function () {
  const b = S.bonferroni(0.02, 5);
  close(b.alphaCorrected, 0.01, 1e-12);
  assert.equal(b.significant, false);
  assert.equal(S.bonferroni(0.005, 5).significant, true);
});

console.log('\n=== 直方图 ===');
check('numpy 官方示例 auto bins', function () {
  const e = S.histogramEdges([0, 0, 0, 1, 2, 3, 3, 4, 5], 'auto');
  assert.deepEqual(e.edges.map(function (v) { return Math.round(v * 1e9) / 1e9; }), [0, 1, 2, 3, 4, 5]);
});
check('最后一个 bin 右闭（最大值不丢失）', function () {
  const e = S.histogramEdges([1, 2, 3, 4], 2);
  const c = S.histogramCounts([1, 2, 3, 4], e.edges);
  assert.equal(c.n, 4);
  assert.equal(c.counts.reduce(function (a, b) { return a + b; }, 0), 4);
  assert.equal(c.counts[c.counts.length - 1], 2, '最大值必须计入最后一个 bin');
});
check('退化数据（全相同）不除零', function () {
  const e = S.histogramEdges([0.5, 0.5, 0.5], 'auto');
  assert.equal(e.method.indexOf('degenerate'), 0);
  assert.equal(e.edges.length, 2);
  assert.equal(S.histogramCounts([0.5, 0.5], e.edges).n, 2);
});
check('分组共享 bin 边界', function () {
  const h = S.sharedHistogram([
    { label: 'A', values: [1, 2, 3, 4, 5] },
    { label: 'B', values: [3, 4, 5, 6, 7] }
  ], 'auto');
  assert.equal(h.series.length, 2);
  assert.equal(h.series[0].counts.length, h.series[1].counts.length);
  assert.equal(h.edges.length, h.series[0].counts.length + 1);
  assert.equal(h.totalN, 10);
});

console.log('\n=== 曲线 / 归一化 ===');
check('EMA 去偏置（TensorBoard 风格）', function () {
  const out = S.emaSmooth([1, 2, 3], 0.6);
  close(out[0], 1, 1e-12);
  close(out[1], 1.4, 1e-12);
  close(out[2], 2.04, 1e-12);
  assert.deepEqual(S.emaSmooth([], 0.6), []);
});
check('buildSeries 缺 step 时降级而非伪造', function () {
  const r = S.buildSeries([{ loss: 1, epoch: 0.5 }, { loss: 0.8, epoch: 1 }], 'step');
  assert.equal(r.mode, 'index');
  assert.ok(r.degraded && r.degraded.indexOf('step') >= 0);
  assert.deepEqual(r.points.map(function (p) { return p.x; }), [0, 1]);
});
check('progress 归一化不超过 1', function () {
  const r = S.buildSeries([{ loss: 2, step: 10 }, { loss: 1, step: 20 }], 'progress', 20);
  assert.deepEqual(r.points.map(function (p) { return p.x; }), [0.5, 1]);
});
check('插值禁止外推', function () {
  assert.equal(S.interpolate([{ x: 0, y: 1 }, { x: 1, y: 2 }], 2), null);
  close(S.interpolate([{ x: 0, y: 1 }, { x: 1, y: 3 }], 0.5), 2, 1e-12);
});
check('aggregateSeries 只在覆盖区给置信带', function () {
  const a = S.buildSeries([{ loss: 3, step: 0 }, { loss: 1, step: 10 }], 'step');
  const b = S.buildSeries([{ loss: 4, step: 0 }, { loss: 2, step: 10 }], 'step');
  const c = S.buildSeries([{ loss: 5, step: 0 }, { loss: 3, step: 4 }], 'step');
  const g = S.aggregateSeries([a, b, c], 10);
  assert.ok(g.grid.length > 5);
  assert.equal(g.grid[0].n, 3);
  assert.ok(g.grid[g.grid.length - 1].n <= 2, '长 run 之外不应再计入已结束的 run');
  assert.equal(g.grid[0].ciLo != null, true);
  assert.equal(g.grid[g.grid.length - 1].ciLo == null || g.grid[g.grid.length - 1].n >= 2, true);
});
check('雷达归一化：lower-is-better 取反 + 退化值中性', function () {
  const r = S.normalizeMetric([2, 1, 0.5], 'lower');
  close(r.scores[0], 0, 1e-12);
  close(r.scores[1], 2 / 3, 1e-12);
  close(r.scores[2], 1, 1e-12);
  const d = S.normalizeMetric([0.8, 0.8, 0.8]);
  assert.equal(d.degenerate, true);
  assert.deepEqual(d.scores, [0.5, 0.5, 0.5]);
});

console.log('\n=== 结果：' + pass + ' passed, ' + fail + ' failed ===\n');
process.exit(fail ? 1 : 0);
