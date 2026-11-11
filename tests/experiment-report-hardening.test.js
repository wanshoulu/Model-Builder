/**
 * 安全加固回归测试（2026-11-10 审核发现问题）
 *  - AI 证据回锚：中文数量短语 / 科学计数法 / 数字紧贴字母 不得绕过校验
 *  - CSV 导出：公式注入字符必须转义，合法负数必须保持数值
 *  - 报告 ID：路径穿越必须被拒绝（HTTP 层，见 experiment-report-http.test.js）
 */
'use strict';
const assert = require('assert');
const AI = require('../routes/experiment-report-ai');
const R = require('../routes/experiment-report');

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}

console.log('=== 1. AI 数值回锚：旧正则漏检的四种写法现在必须被捕获 ===');
const PROBES = [
  ['中文数量短语「三成」', '准确率提升了三成', '三成'],
  ['中文百分比「百分之二十」', '比基线高百分之二十', '百分之二十'],
  ['科学计数法「3e-4」', '学习率 3e-4 更优', '3e-4'],
  ['数字紧贴字母「acc0.99」', 'acc0.99 的表现', '0.99'],
  ['年份「2024」', '基于 2024 年的实验', '2024'],
  ['百分点', '提升了五个百分点', '五个百分点']
];
PROBES.forEach(function (p) {
  check(p[0], function () {
    const got = AI.extractNumbers(p[1]);
    assert.ok(got.indexOf(p[2]) >= 0, '未捕获 "' + p[2] + '"，实际：' + JSON.stringify(got));
  });
});

console.log('\n=== 2. 降级清理：可疑数值不进入渲染层，合法数值保留 ===');
check('sanitizeSections 剔除编造数值并保留合法值', function () {
  const allowed = new Set(['0.81', '3', '95']);
  const built = { refs: new Set(['meta.run_count']), numbers: allowed };
  const sec = {
    headline: '准确率 0.81，学习率 3e-4，比基线高三成，基于 2024 年',
    executive_summary: '置信度 95；acc0.99 为编造',
    scope_and_method: '共 3 次运行',
    findings: [{ title: '结论', text: '提升了三成', evidence_refs: ['meta.run_count'] }],
    figure_notes: [{ fig: 1, text: '图 1 显示 0.81' }],
    limitations: ['样本量有限'], next_steps: ['补做 5 个种子']
  };
  const out = AI.sanitizeSections(sec, built);
  const blob = JSON.stringify(out.sections);
  assert.ok(blob.indexOf('0.81') >= 0, '合法数值 0.81 不应被移除');
  assert.ok(blob.indexOf('3e-4') < 0, '无出处的科学计数法应被剔除');
  assert.ok(blob.indexOf('三成') < 0, '中文数量短语应被剔除');
  assert.ok(blob.indexOf('2024') < 0, '无出处年份应被剔除');
  assert.ok(blob.indexOf('0.99') < 0, 'acc0.99 里的数值应被剔除');
  assert.ok(out.removed.length >= 4, '应记录被移除的 token，实际 ' + JSON.stringify(out.removed));
});

console.log('\n=== 3. CSV 公式注入防护 ===');
function fakeResult() {
  return {
    runs: [{ name: '=HYPERLINK("http://evil","click")', id: 'r1' }],
    tableRows: [{ label: '@SUM(1+1)', values: ['+cmd|/c calc', '-1+1', '=2+2', '-0.5', '普通文本'] }],
    stats: { accuracy: [{ label: 'A', stats: { n: 3, mean: 0.81, sd: 0.01, ci95: [0.79, 0.83] } }] },
    tests: [], groups: [], warnings: [],
    meta: { generatedAt: Date.now(), runCount: 1, groupLabel: 'task', primary: 'accuracy' }
  };
}
check('导出 CSV 中公式字符被前置单引号转义', function () {
  let csv;
  try { csv = R.exportData(fakeResult(), 'csv'); }
  catch (e) { throw new Error('exportData 抛错：' + e.message); }
  ['=HYPERLINK', '@SUM', '+cmd', '=2+2'].forEach(function (s) {
    assert.ok(csv.indexOf("'" + s) >= 0, '"' + s + '" 未被转义');
  });
  assert.ok(csv.indexOf(',=2+2') < 0 && csv.indexOf(',=HYPERLINK') < 0, '仍存在裸公式单元格');
});
check('合法负数 -0.5 不被转义（保持数值）', function () {
  const csv = R.exportData(fakeResult(), 'csv');
  assert.ok(/,\s*-0\.5\b/.test(csv) || csv.indexOf(',-0.5') >= 0, '-0.5 应原样输出');
  assert.ok(csv.indexOf("'-0.5") < 0, '合法负数不应变文本');
});
check('普通文本与含逗号文本仍正确引用', function () {
  const r = fakeResult();
  r.tableRows[0].values.push('a,b');
  const csv = R.exportData(r, 'csv');
  assert.ok(csv.indexOf('"a,b"') >= 0, '含逗号字段应加引号');
});

console.log('\n=== 4. 资源 ID 白名单（路径穿越的模块级回归）===');
const path = require('path');
const MT = require('../routes/model-training');
check('报告 id 拒绝 ../、反斜杠、点号与超长串', function () {
  ['../metrics', '..\\metrics', 'rep/../../x', 'a.b', '', 'x'.repeat(200)].forEach(function (id) {
    assert.strictEqual(R.reportFile(id), null, '"' + id + '" 不应通过');
  });
});
check('合法报告 id 落在 REPORTS_DIR 内', function () {
  const f = R.reportFile('rep-abc_123');
  assert.ok(f && path.dirname(f).endsWith(path.join('model-training', 'reports')), '实际：' + f);
});
check('job / dataset id 同样拒绝穿越', function () {
  ['../servers', '..\\servers', 'job/x', 'a.json', ''].forEach(function (id) {
    assert.strictEqual(MT.isSafeId(id), false, '"' + id + '" 不应通过');
  });
  assert.strictEqual(MT.isSafeId('job-mwa20qawa7zh'), true);
  assert.strictEqual(MT.isSafeId('ds-mwa20q9yoo7f'), true);
});

console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
process.exit(failed ? 1 : 0);
