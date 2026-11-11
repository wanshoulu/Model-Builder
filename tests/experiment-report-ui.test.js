/**
 * 实验报告前端渲染校验（无浏览器依赖）
 * 运行： node tests/experiment-report-ui.test.js
 *
 * 做法：用 vm 执行 model-training.html 内联脚本，注入极简 DOM stub 与 Plotly mock，
 * 再把后端 analyze 的真实结果交给 renderRepAll()，逐个校验交给图表库的 spec
 * （trace / layout / config），而不是校验手写 SVG 字符串：
 *  - 脚本不抛异常，所有图表都产出合法 spec
 *  - spec 中不含 NaN / undefined / Infinity
 *  - 学术口径落到库配置上：serif、轴标题带量纲、柱从 0 起、error_y 等于后端 CI 半宽、
 *    共享 bin、雷达各轴统一 [0,1]、平行坐标逐轴独立 range
 *  - AI 叙述章节的 Markdown 渲染与 XSS 转义
 * 测试会临时写 metrics.json，结束后原样恢复。
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const R = require('../routes/experiment-report');

const ROOT = path.join(__dirname, '..');
const MF = path.join(ROOT, 'data', 'model-training', 'metrics.json');
const backup = fs.existsSync(MF) ? fs.readFileSync(MF, 'utf8') : null;
let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); console.log('  PASS  ' + name); pass++; }
  catch (e) { console.log('  FAIL  ' + name + ' → ' + e.message); fail++; }
}

function makeEl(id) {
  return {
    id: id, innerHTML: '', textContent: '', value: '', checked: false, dataset: {}, style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, remove() {}, click() {}, addEventListener() {}, focus() {},
    querySelectorAll() { return []; }, getBoundingClientRect() { return { width: 720 }; }
  };
}
function makeDocument() {
  const els = {};
  return {
    els: els,
    getElementById(id) { if (!els[id]) els[id] = makeEl(id); return els[id]; },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    addEventListener() {}, removeEventListener() {},
    createElement(tag) { return makeEl(tag); },
    body: { appendChild() {}, addEventListener() {} }
  };
}

try {
  // 构造 2 组 × 5 seed × 3 指标，让雷达图 / 配对检验 / 置信带全部启用
  const store = { entries: [], externalRuns: [] };
  const A = [0.812, 0.825, 0.807, 0.819, 0.817];
  const B = [0.876, 0.884, 0.871, 0.889, 0.880];
  ['accuracy', 'f1', 'exact_match'].forEach(function (metric, mi) {
    [1, 2, 3, 4, 5].forEach(function (s, i) {
      store.entries.push({ id: 'ua' + mi + i, runId: 'ui-run-A', name: metric, value: A[i] - mi * 0.02, unit: 'ratio', task: 'bank-intent', seed: s, n: 200, source: 'import', direction: 'higher', createdAt: Date.now() });
      store.entries.push({ id: 'ub' + mi + i, runId: 'ui-run-B', name: metric, value: B[i] - mi * 0.02, unit: 'ratio', task: 'bank-intent', seed: s, n: 200, source: 'import', direction: 'higher', createdAt: Date.now() });
    });
  });
  store.externalRuns.push(
    { id: 'ui-run-A', name: 'UI-A r8', task: 'task-A', dataset: 'bank-intent', engine: 'lora', model: 'Qwen3-0.6B', createdAt: Date.now() },
    { id: 'ui-run-B', name: 'UI-B r16', task: 'task-B', dataset: 'bank-intent', engine: 'lora', model: 'Qwen3-0.6B', createdAt: Date.now() }
  );
  fs.writeFileSync(MF, JSON.stringify(store, null, 2), 'utf8');

  const jobsDir = path.join(ROOT, 'data', 'model-training', 'jobs');
  const jobIds = fs.existsSync(jobsDir) ? fs.readdirSync(jobsDir).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)) : [];
  const runIds = jobIds.concat(['ui-run-A', 'ui-run-B']);
  const result = R.analyze({
    runIds: runIds, groupBy: 'engine', xMode: 'progress',
    metricNames: ['accuracy', 'f1', 'exact_match'], emaAlpha: 0.6,
    histogramMethod: 'auto', bandType: 'ci95'
  });
  assert.ok(result.curves.length >= 1, '需要至少一条真实 loss 曲线');

  const html = fs.readFileSync(path.join(ROOT, 'tools', 'model-training.html'), 'utf8');
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  assert.ok(m && m[1], '未找到内联脚本');

  const doc = makeDocument();
  // Plotly mock：捕获传入的 data / layout / config，模拟库渲染后写入 SVG 占位
  const plotlyCalls = [];
  const PlotlyMock = {
    react(el, data, layout, config) {
      plotlyCalls.push({ id: el.id, data: data, layout: layout, config: config });
      el.__plotly = { data: data, layout: layout, config: config };
      el.innerHTML = '<svg data-plotly="1"></svg>';
      return Promise.resolve();
    },
    purge(el) { el.__plotly = null; el.innerHTML = ''; }
  };
  const ctx = {
    console: console, document: doc,
    location: { search: '' },
    window: { scrollTo() {}, addEventListener() {}, removeEventListener() {}, location: { search: '' }, Plotly: PlotlyMock },
    setTimeout() { return 0; }, clearTimeout() {}, setInterval() { return 0; }, clearInterval() {},
    alert() {}, confirm() { return true; }, prompt() { return 'test'; },
    Blob: function () {}, URL: { createObjectURL() { return 'blob:x'; }, revokeObjectURL() {} },
    fetch: async function (url) {
      return { ok: true, status: 200, json: async function () {
        if (String(url).indexOf('/api/model-training') === 0) return { ok: true, servers: [], datasets: [], jobs: [] };
        return { ok: true, runs: [], entries: [], reports: [], engines: [] };
      } };
    }
  };
  ctx.__REPORT_RESULT__ = result;
  vm.createContext(ctx);

  check('页面内联脚本在 DOM stub 中可完整执行（含 init）', function () {
    try { vm.runInContext(m[1], ctx, { filename: 'model-training.inline.js' }); }
    catch (e) { throw new Error('脚本执行异常：' + e.message); }
  });
  check('renderRepAll 全部图表执行不抛异常', function () {
    vm.runInContext('rep.result = __REPORT_RESULT__; renderRepAll();', ctx);
  });

  const specs = vm.runInContext('REP_SPECS', ctx);
  const chartIds = ['repCurveChart', 'repBarChart', 'repHistChart', 'repRadarChart', 'repParallelChart'];
  const textIds = ['repSummary', 'repTests', 'repDiffTable', 'repMethods', 'repWarnings'];

  check('五张图都把 spec 交给了 Plotly（不再手写 SVG）', function () {
    chartIds.forEach(function (id) {
      const s = specs[id];
      assert.ok(s, id + ' 缺少 spec');
      assert.ok(Array.isArray(s.data) && s.data.length > 0, id + ' data 为空');
      assert.ok(s.layout && typeof s.layout === 'object', id + ' layout 缺失');
      assert.ok(s.config && s.config.responsive === true, id + ' 未启用响应式');
      assert.ok(s.config.toImageButtonOptions.format === 'svg', id + ' 未配置矢量 SVG 导出');
      assert.ok(doc.els[id].innerHTML.indexOf('<svg') >= 0, id + ' 容器无内容');
    });
    assert.ok(plotlyCalls.length >= chartIds.length, 'Plotly.react 调用次数不足');
  });
  check('所有 spec 中不含 NaN / undefined / Infinity', function () {
    chartIds.forEach(function (id) {
      const s = JSON.stringify(specs[id]);
      assert.ok(!/NaN|Infinity/.test(s), id + ' 含非法数值：' + (s.match(/.{0,60}(NaN|Infinity).{0,60}/) || [''])[0]);
      assert.ok(s.indexOf('"undefined"') < 0, id + ' 含 undefined');
    });
  });
  check('学术版式：serif 字体 + 轴标题带量纲', function () {
    ['repCurveChart', 'repBarChart', 'repHistChart'].forEach(function (id) {
      const L = specs[id].layout;
      assert.ok(/Times New Roman|Georgia/.test(L.font.family), id + ' 未使用 serif');
      assert.ok(L.xaxis.title.text && L.yaxis.title.text, id + ' 缺轴标题');
      assert.ok(L.xaxis.showline && !L.xaxis.zeroline, id + ' spines 设置不正确');
    });
    assert.ok(/Loss \(dimensionless\)/.test(specs.repCurveChart.layout.yaxis.title.text), 'loss 轴未标量纲');
  });
  check('曲线图：raw + EMA + fill=tonexty 分组置信带', function () {
    const d = specs.repCurveChart.data;
    assert.ok(d.filter(function (t) { return t.fill === 'tonexty'; }).length >= 1, '缺少 fill=tonexty 置信带');
    assert.ok(d.filter(function (t) { return t.line && t.line.width === 2; }).length >= result.curves.length, 'EMA 粗线不足');
    assert.ok(d.filter(function (t) { return t.hoverinfo === 'skip' && t.line && t.line.width === 1; }).length >= result.curves.length, 'raw 细线不足');
    assert.ok(doc.els.repCurveCap.innerHTML.indexOf('禁止外推') >= 0, '图注未声明禁止外推');
  });
  check('柱状图：从 0 起画，error_y 等于后端 95% CI 半宽', function () {
    const t = specs.repBarChart.data[0];
    assert.equal(t.type, 'bar');
    assert.equal(specs.repBarChart.layout.yaxis.rangemode, 'tozero');
    const list = result.stats[result.meta.primary] || [];
    assert.equal(t.error_y.array.length, list.length, '误差棒数量与分组数不一致');
    list.forEach(function (g, i) {
      const expect = g.stats.ci95 ? (g.stats.ci95[1] - g.stats.ci95[0]) / 2 : 0;
      assert.ok(Math.abs(t.error_y.array[i] - expect) < 1e-12, 'CI 半宽与后端不一致');
    });
    assert.ok(doc.els.repBarCap.innerHTML.indexOf('95% 置信区间') >= 0);
  });
  check('直方图：所有分组共享 bin 中心与 bin 宽度', function () {
    const h = result.histogram, d = specs.repHistChart.data;
    assert.equal(d.length, h.series.length);
    const x0 = JSON.stringify(d[0].x);
    d.forEach(function (t) {
      assert.equal(JSON.stringify(t.x), x0, '各组 bin 中心不一致');
      assert.ok(Math.abs(t.width - h.binWidth) < 1e-12, 'bin 宽度不一致');
      assert.equal(t.y.length, h.edges.length - 1, 'bin 数量不一致');
    });
    assert.ok(doc.els.repHistCap.innerHTML.indexOf('共享') >= 0);
  });
  check('雷达图：scatterpolar、闭合多边形、各轴统一 [0,1]', function () {
    assert.equal(result.radar.enabled, true);
    const d = specs.repRadarChart.data, L = specs.repRadarChart.layout;
    assert.ok(d.length >= 2 && d.every(function (t) { return t.type === 'scatterpolar'; }));
    d.forEach(function (t) {
      assert.equal(t.r[0], t.r[t.r.length - 1], '多边形未闭合');
      assert.equal(t.theta[0], t.theta[t.theta.length - 1]);
      assert.ok(t.fill === 'toself');
    });
    assert.deepEqual(Array.from(L.polar.radialaxis.range), [0, 1]);
    assert.ok(doc.els.repRadarCap.innerHTML.indexOf('面积不代表综合得分') >= 0);
    assert.ok(doc.els.repRadarRaw.innerHTML.indexOf('原始均值') >= 0);
  });
  check('平行坐标：parcoords 逐轴独立 range，缺失值保留 null', function () {
    const t = specs.repParallelChart.data[0];
    assert.equal(t.type, 'parcoords');
    assert.equal(t.dimensions.length, result.parallel.axes.length);
    t.dimensions.forEach(function (dim) {
      assert.ok(Array.isArray(dim.range) && dim.range.length === 2, '缺少独立 range');
      assert.equal(dim.values.length, result.parallel.runs.length);
    });
    assert.ok(doc.els.repParallelCap.innerHTML.indexOf('独立') >= 0);
  });
  check('文本型结果：汇总表 / 检验表 / 方法面板齐全', function () {
    textIds.forEach(function (id) {
      assert.ok(doc.els[id].innerHTML.length > 40, id + ' 内容为空');
    });
    assert.ok(doc.els.repSummary.innerHTML.indexOf('95% CI (t)') >= 0);
    assert.ok(doc.els.repTests.innerHTML.indexOf('Bonferroni') >= 0);
    assert.ok(doc.els.repTests.innerHTML.indexOf('Cohen') >= 0);
    assert.ok(doc.els.repMethods.innerHTML.indexOf('numpy.histogram_bin_edges') >= 0);
    assert.ok(doc.els.repMethods.innerHTML.indexOf('arXiv:2611.04515') >= 0, '方法面板应列出 AI 报告设计依据');
  });
  check('切到单指标时雷达图自动禁用（不产出 spec）并给出理由', function () {
    const single = R.analyze({ runIds: runIds, groupBy: 'engine', metricNames: ['accuracy'], xMode: 'progress' });
    ctx.__REPORT_RESULT__ = single;
    vm.runInContext('rep.result = __REPORT_RESULT__; renderRepAll();', ctx);
    const s2 = vm.runInContext('REP_SPECS', ctx);
    assert.ok(!s2.repRadarChart, '禁用时不应产出 chart spec');
    assert.ok(doc.els.repRadarChart.innerHTML.indexOf('至少') >= 0);
    assert.ok(doc.els.repRadarChart.innerHTML.indexOf('<svg') < 0, '禁用时不应画多边形');
  });
  check('Diff only 开关过滤相同字段', function () {
    ctx.__REPORT_RESULT__ = result;
    vm.runInContext('rep.result = __REPORT_RESULT__; renderRepAll();', ctx);
    const before = (doc.els.repDiffTable.innerHTML.match(/<tr/g) || []).length;
    doc.els.repDiffOnly.checked = true;
    vm.runInContext('renderRepTable();', ctx);
    const after = (doc.els.repDiffTable.innerHTML.match(/<tr/g) || []).length;
    assert.ok(after <= before, 'diff only 后行数不应增加');
  });

  console.log('\n=== AI 叙述章节 ===');
  check('AI 相关函数齐备且默认显示使用约束', function () {
    ['generateAiReport', 'pollAiTask', 'exportFullReport', 'renderAiReport', 'repMdToHtml', 'renderAiEngines']
      .forEach(function (fn) { assert.equal(vm.runInContext('typeof ' + fn, ctx), 'function', fn + ' 缺失'); });
    assert.ok(doc.els.repAiBody.innerHTML.indexOf('未经人工复核') >= 0 || doc.els.repAiBody.innerHTML.indexOf('不参与') >= 0);
  });
  check('Markdown 渲染：标题 / 表格 / 引用 / 列表，且转义 HTML', function () {
    const out = vm.runInContext('repMdToHtml(' + JSON.stringify(
      '# 标题\n\n## 小节\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n> 引用 **重点**\n\n- 项目一\n- 项目二\n\n<script>alert(1)</script>'
    ) + ')', ctx);
    assert.ok(out.indexOf('<h2>标题</h2>') >= 0);
    assert.ok(out.indexOf('<h3>小节</h3>') >= 0);
    assert.ok(out.indexOf('<table') >= 0);
    assert.ok(out.indexOf('<blockquote>') >= 0);
    assert.ok(out.indexOf('<b>重点</b>') >= 0);
    assert.ok(out.indexOf('<li>项目一</li>') >= 0);
    assert.ok(out.indexOf('<script>') < 0, '存在 XSS 风险');
  });
  check('渲染 AI 报告时标注模型、校验状态与复核声明', function () {
    vm.runInContext('rep.ai = {markdown:"# 报告\\n\\n## 1. 分析范围\\n\\n结论文本。", model:"qwen3.7-flash", validation:{refsChecked:42,numbersChecked:88}, elapsedMs:1234}; rep.aiStale=false; renderAiReport();', ctx);
    assert.ok(doc.els.repAiMeta.innerHTML.indexOf('未经人工复核') >= 0);
    assert.ok(doc.els.repAiMeta.innerHTML.indexOf('qwen3.7-flash') >= 0);
    assert.ok(doc.els.repAiMeta.innerHTML.indexOf('42') >= 0);
    assert.ok(doc.els.repAiBody.innerHTML.indexOf('结论文本') >= 0);
    assert.ok(doc.els.repAiActions.innerHTML.indexOf('复制 Markdown') >= 0);
  });
  check('配置变更后 AI 叙述被标记为过期', function () {
    vm.runInContext('rep.aiStale=true; renderAiReport();', ctx);
    assert.ok(doc.els.repAiMeta.innerHTML.indexOf('配置已变更') >= 0);
  });
} finally {
  if (backup == null) { try { fs.unlinkSync(MF); } catch (e) {} }
  else fs.writeFileSync(MF, backup, 'utf8');
  console.log('\n=== 结果：' + pass + ' passed, ' + fail + ' failed ===');
  console.log('(metrics.json 已还原)\n');
}
process.exit(fail ? 1 : 0);
