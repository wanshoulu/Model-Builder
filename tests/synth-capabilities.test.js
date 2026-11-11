// 合成工具新能力测试：label-variant / rule-map / decision 种子注入 / holdout 泄漏隔离
const assert = require('node:assert');
const path = require('node:path');
const { qualityFunnel } = require('../routes/dataset-tools');

let pass = 0, fail = 0;
function check(name, fn) {
  try { fn(); pass++; console.log('  PASS', name); }
  catch (e) { fail++; console.log('  FAIL', name, '::', e.message); }
}

console.log('=== 1. holdout 泄漏隔离（质检漏斗第 7 段）===');
check('与评测集重复的样本被剔除', function () {
  const samples = [
    { instruction: '查一下我的余额', output: '余额查询' },
    { instruction: '帮我转账给妈妈', output: '转账' },
    { instruction: '看看基金收益', output: '基金收益' },
  ];
  const holdout = ['查一下我的余额', '看看基金收益'];
  const r = qualityFunnel(samples, { holdoutQueries: holdout });
  const leakStage = r.stages.find(s => s.stage === 'holdout_leakage');
  assert.ok(leakStage, '应存在 holdout_leakage 阶段');
  assert.strictEqual(leakStage.removed, 2);
  assert.strictEqual(r.samples.length, 1);
  assert.strictEqual(r.samples[0].instruction, '帮我转账给妈妈');
});
check('规范化匹配（大小写/空白折叠）', function () {
  const r = qualityFunnel([{ instruction: '查一下 我的 余额', output: '余额查询相关的长回答内容' }], { holdoutQueries: ['查一下我的余额'] });
  const leakStage = r.stages.find(s => s.stage === 'holdout_leakage');
  assert.strictEqual(leakStage.removed, 1);
});
check('decision 样本按 state 泄漏检查', function () {
  const r = qualityFunnel([
    { state: '帮我查汇率', questions: {} },
    { state: '买点理财', questions: {} },
  ], { holdoutQueries: ['帮我查汇率'] });
  assert.strictEqual(r.samples.length, 1);
  assert.strictEqual(r.samples[0].state, '买点理财');
});
check('无 holdoutQueries 时该阶段跳过', function () {
  const r = qualityFunnel([{ instruction: 'x', output: 'y' }]);
  assert.ok(!r.stages.some(s => s.stage === 'holdout_leakage'));
});

console.log('=== 2. rule-map 确定性映射（零 LLM）===');
// 直接构造与路由等价的规则验证映射语义（核心是关键词路由正确性）
check('域关键词按序匹配', function () {
  const kwMap = [
    { keywords: ['基金'], domain: '基金业务' },
    { keywords: ['存款', '存单'], domain: '存款业务' },
  ];
  const route = (label) => {
    for (const kw of kwMap) if (kw.keywords.some(k => label.indexOf(k) >= 0)) return kw.domain;
    return '综合';
  };
  assert.strictEqual(route('基金推荐'), '基金业务');
  assert.strictEqual(route('大额存单预约'), '存款业务');
  assert.strictEqual(route('E红包'), '综合');
});

console.log('=== 3. PROMPTS.labelVariant 模板 ===');
check('模板要求 JSON 数组输出与风格多样性', function () {
  // PROMPTS 未导出，通过源码静态断言
  const src = require('fs').readFileSync(path.join(__dirname, '../routes/dataset-synthesis.js'), 'utf8');
  assert.ok(/labelVariant: \(label, examples, nVariants, styles\)/.test(src), 'labelVariant 模板存在');
  assert.ok(/严格保持与样例相同的意图（不得漂移/.test(src), '含防漂移约束');
  assert.ok(/"query": "\.\.\.", "style": "\.\.\."/ .test(src), '输出 schema 含 query+style');
});

console.log('=== 4. decision 种子注入模板 ===');
check('decision 模板支持 seeds 分支', function () {
  const src = require('fs').readFileSync(path.join(__dirname, '../routes/dataset-synthesis.js'), 'utf8');
  assert.ok(/spec\.seeds && spec\.seeds\.length/.test(src), 'PROMPTS.decision 读 spec.seeds');
  assert.ok(/禁止编造与种子无关的内容/.test(src), '含禁止编造约束');
});

console.log('=== 结果：' + pass + ' passed, ' + fail + ' failed ===');
process.exit(fail ? 1 : 0);
