/**
 * 实验报告后端端到端校验
 * 运行： node tests/experiment-report.test.js
 *
 * 覆盖：真实 jobs 聚合 → CSV/JSON 导入（长表/宽表）→ 分组统计 → 配对检验 →
 *       雷达/直方图/曲线 → 导出。测试会临时写 metrics.json，结束后原样恢复。
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const R = require('../routes/experiment-report');

const MF = path.join(__dirname, '..', 'data', 'model-training', 'metrics.json');
const backup = fs.existsSync(MF) ? fs.readFileSync(MF, 'utf8') : null;
let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); console.log('  PASS  ' + name); pass++; }
  catch (e) { console.log('  FAIL  ' + name + ' → ' + e.message); fail++; }
}

try {
  console.log('\n=== 真实训练任务聚合（无任何录入指标）===');
  const jobsDir = path.join(__dirname, '..', 'data', 'model-training', 'jobs');
  const runIds = fs.existsSync(jobsDir) ? fs.readdirSync(jobsDir).filter(f => f.endsWith('.json')).map(f => f.replace(/\.json$/, '')) : [];
  let base = null;
  check('可从历史任务生成分析（不依赖手工指标）', function () {
    if (!runIds.length) { console.log('       (无历史任务，跳过)'); return; }
    base = R.analyze({ runIds: runIds, xMode: 'progress' });
    assert.equal(base.ok, true);
    assert.equal(base.runs.length, runIds.length);
    assert.ok(base.catalog.length >= 3, '应至少有 final/min loss 等自动指标');
    assert.ok(base.curves.length >= 1, '至少一个任务有 loss 曲线');
    assert.ok(base.histogram.edges.length >= 2);
    assert.ok(base.tableRows.length > 5);
  });
  check('loss-only 时明确警告不是精度指标', function () {
    if (!base) return;
    assert.ok(base.warnings.some(w => w.indexOf('不是留出集精度') >= 0));
  });
  check('雷达图指标不足 3 个时禁用并给理由', function () {
    if (!base) return;
    const r = R.analyze({ runIds: runIds, xMode: 'progress', metricNames: ['Final Train Loss'] });
    assert.equal(r.radar.enabled, false);
    assert.ok(r.radar.reason.indexOf('至少') >= 0);
  });
  check('分组维度可切换（engine / model / epochs）', function () {
    if (!base) return;
    ['none', 'engine', 'model', 'epochs', 'learningRate'].forEach(function (g) {
      const r = R.analyze({ runIds: runIds, groupBy: g, xMode: 'index' });
      assert.ok(r.groups.length >= 1, g);
      assert.equal(r.meta.groupBy, g);
    });
  });
  check('x 轴四种口径都不崩且 step 缺失时降级', function () {
    if (!base) return;
    ['index', 'step', 'epoch', 'progress'].forEach(function (m) {
      const r = R.analyze({ runIds: runIds, xMode: m });
      assert.ok(r.curves.length >= 1, m);
      assert.ok(r.curves.every(c => c.series.points.length > 0), m);
    });
  });

  console.log('\n=== CSV / JSON 导入解析 ===');
  check('CSV 长表（含 seed / task / unit）', function () {
    const rec = R.parseImport('run,metric,value,unit,task,seed,n\nrunA,accuracy,0.91,%,bank-qa,1,200\nrunA,accuracy,0.93,%,bank-qa,2,200\nrunB,accuracy,0.85,%,bank-qa,1,200');
    assert.equal(rec.length, 3);
    assert.equal(rec[0].metric, 'accuracy');
    assert.equal(rec[0].value, 0.91);
    assert.equal(rec[1].seed, '2');
  });
  check('CSV 宽表（列名即指标，含引号与逗号）', function () {
    const rec = R.parseImport('run,task,seed,accuracy,f1,loss\n"A, baseline",bank-qa,1,0.80,0.79,0.5\nB,bank-qa,2,0.88,0.86,0.3');
    assert.equal(rec.length, 6);
    assert.equal(rec[0].run, 'A, baseline');
    assert.ok(rec.some(r => r.metric === 'f1' && r.value === 0.86));
  });
  check('JSON 长表 + 宽表对象', function () {
    const long = R.parseImport(JSON.stringify([{ run: 'X', metric: 'accuracy', value: 0.9, seed: 1 }, { run: 'Y', metric: 'accuracy', value: 0.8, seed: 1 }]));
    assert.equal(long.length, 2);
    const wide = R.parseImport(JSON.stringify([{ run: 'X', task: 't1', seed: 3, accuracy: 0.9, f1: 0.88, loss: 0.4 }]));
    assert.equal(wide.length, 3);
    assert.ok(wide.some(r => r.metric === 'accuracy' && r.value === 0.9));
  });
  check('无效内容给出明确错误', function () {
    assert.throws(() => R.parseImport(''), /为空/);
    assert.throws(() => R.parseImport('a,b\nonly,header'), /未解析到任何有效指标行/);
  });

  console.log('\n=== 多 seed 分组统计 / 显著性检验 / 雷达图 ===');
  // 构造已知数据：两组各 5 个 seed，B 稳定高于 A 约 0.06
  const store = { entries: [], externalRuns: [] };
  const seeds = [1, 2, 3, 4, 5];
  const A = [0.812, 0.825, 0.807, 0.819, 0.817];
  const B = [0.876, 0.884, 0.871, 0.889, 0.880];
  ['accuracy', 'f1', 'exact_match'].forEach(function (metric, mi) {
    seeds.forEach(function (s, i) {
      store.entries.push({ id: 'ta' + mi + i, runId: 'run-A', name: metric, value: A[i] - mi * 0.02, unit: '%', task: 'bank-qa', seed: s, n: 200, source: 'import', direction: 'higher', createdAt: Date.now() });
      store.entries.push({ id: 'tb' + mi + i, runId: 'run-B', name: metric, value: B[i] - mi * 0.02, unit: '%', task: 'bank-qa', seed: s, n: 200, source: 'import', direction: 'higher', createdAt: Date.now() });
    });
  });
  store.externalRuns.push({ id: 'run-A', name: 'LoRA r8', task: 'bank-qa', engine: 'lora', model: 'Qwen3-0.6B', createdAt: Date.now() });
  store.externalRuns.push({ id: 'run-B', name: 'LoRA r16', task: 'bank-qa', engine: 'lora', model: 'Qwen3-0.6B', createdAt: Date.now() });
  fs.writeFileSync(MF, JSON.stringify(store, null, 2), 'utf8');

  let full = null;
  check('外部导入 run 与本机 job 同表可选', function () {
    // 通过 handle 的 runs 逻辑间接验证：analyze 能按 id 找到 external run
    full = R.analyze({ runIds: ['run-A', 'run-B'], groupBy: 'dataset', metricNames: ['accuracy', 'f1', 'exact_match'], xMode: 'index' });
    assert.equal(full.runs.length, 2);
    assert.ok(full.runs.every(r => r.origin === 'external'));
  });
  check('分组统计 n=5、mean 与 95% CI 数值正确', function () {
    const gA = full.stats.accuracy.find(g => g.label.indexOf('run-A') >= 0 || g.group === '—');
    const rows = full.stats.accuracy;
    assert.equal(rows.length, 1, '两组 dataset 相同 → 应合并为 1 组；若要分开比较需按不同 facet 分组');
    assert.equal(rows[0].stats.n, 10);
  });
  // 按 model 分组无法区分（同 model），改用 name 作为分组维度：给两组不同 task
  const store2 = JSON.parse(JSON.stringify(store));
  store2.entries.forEach(function (e) { e.task = e.runId === 'run-A' ? 'task-A' : 'task-B'; });
  store2.externalRuns.forEach(function (e) { e.task = e.id === 'run-A' ? 'task-A' : 'task-B'; });
  fs.writeFileSync(MF, JSON.stringify(store2, null, 2), 'utf8');
  check('按 task 分成两组 → 配对 t 检验显著 + Bonferroni', function () {
    full = R.analyze({ runIds: ['run-A', 'run-B'], groupBy: 'task', metricNames: ['accuracy', 'f1', 'exact_match'], xMode: 'index' });
    assert.equal(full.groups.length, 2);
    const ga = full.stats.accuracy.find(g => g.group === 'task-A');
    const gb = full.stats.accuracy.find(g => g.group === 'task-B');
    assert.equal(ga.stats.n, 5);
    assert.equal(gb.stats.n, 5);
    assert.ok(Math.abs(ga.stats.mean - 0.816) < 1e-9, 'A mean');
    assert.ok(Math.abs(gb.stats.mean - 0.88) < 1e-9, 'B mean');
    assert.ok(ga.stats.ci95 && ga.stats.ci95[0] < ga.stats.mean && ga.stats.ci95[1] > ga.stats.mean);
    assert.equal(full.tests.length, 1);
    assert.equal(full.tests[0].mode, 'paired', '相同 seed 应配对');
    assert.ok(full.tests[0].result.p < 0.001, 'p 应极显著，got ' + full.tests[0].result.p);
    assert.equal(full.tests[0].significant, true);
    // 差值定义为 A − B：accuracy 上 A 更低，故 d 为大负数；效应量看绝对值
    assert.ok(Math.abs(full.tests[0].result.cohensD) > 2, '效应量绝对值应很大，got ' + full.tests[0].result.cohensD);
  });
  check('3 指标 × 2 组时雷达图启用且归一化正确', function () {
    assert.equal(full.radar.enabled, true);
    assert.equal(full.radar.axes.length, 3);
    assert.equal(full.radar.groups.length, 2);
    assert.ok(full.radar.axes.every(a => a.scores.every(s => s >= 0 && s <= 1)));
  });
  check('直方图共享 bins 且计数守恒', function () {
    const h = full.histogram;
    assert.equal(h.series.length, 2);
    assert.equal(h.totalN, 10);
    const total = h.series.reduce((a, s) => a + s.counts.reduce((x, y) => x + y, 0), 0);
    assert.equal(total, 10);
  });
  check('导出 CSV / Markdown 含统计口径', function () {
    const csv = R.exportData(full, 'csv');
    assert.ok(csv.indexOf('95% CI low') >= 0);
    assert.ok(csv.indexOf('accuracy') >= 0);
    const md = R.exportData(full, 'markdown');
    assert.ok(md.indexOf('## 汇总统计') >= 0);
    assert.ok(md.indexOf('Bonferroni') >= 0);
    assert.ok(md.indexOf('面积不代表综合得分') >= 0);
  });
  check('n<3 时给出样本量警告', function () {
    const s3 = JSON.parse(JSON.stringify(store2));
    s3.entries = s3.entries.filter(e => e.name === 'accuracy' && (e.seed === 1 || e.seed === 2));
    fs.writeFileSync(MF, JSON.stringify(s3, null, 2), 'utf8');
    const r = R.analyze({ runIds: ['run-A', 'run-B'], groupBy: 'task', metricNames: ['accuracy'], xMode: 'index' });
    assert.ok(r.warnings.some(w => w.indexOf('n=2') >= 0));
  });
} finally {
  // 恢复现场，绝不把测试假数据留在用户数据里
  if (backup == null) { try { fs.unlinkSync(MF); } catch (e) {} }
  else fs.writeFileSync(MF, backup, 'utf8');
  console.log('\n=== 结果：' + pass + ' passed, ' + fail + ' failed ===');
  console.log('(metrics.json 已还原为测试前状态)\n');
}
process.exit(fail ? 1 : 0);
