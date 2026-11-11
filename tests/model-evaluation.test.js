/**
 * 训练后自动评估编排回归测试：解析 → 对齐 → 评分 → 回写 job（含失败降级）。
 * 用内存 IO 注入，不触碰真实数据集 / GPU / 服务进程。
 */
'use strict';
const assert = require('assert');
const E = require('../routes/model-evaluation');
const M = require('../routes/eval-metrics');

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}
function close(a, b, eps) { assert.ok(Math.abs(a - b) < (eps || 1e-6), a + ' ≠ ' + b); }
function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

console.log('=== 1. 评估集解析与标准答案抽取 ===');
check('.jsonl 逐行解析，非法行带行号报错', function () {
  const s = E.parseSamples('eval.jsonl', '{"a":1}\n\n{"a":2}\n');
  assert.strictEqual(s.length, 2);
  assert.throws(function () { E.parseSamples('x.jsonl', '{"a":1}\nNOT_JSON\n'); }, /第 2 行/);
});
check('.json 数组与 {samples:[]} 都支持', function () {
  assert.strictEqual(E.parseSamples('x.json', '[{"a":1}]').length, 1);
  assert.strictEqual(E.parseSamples('x.json', '{"samples":[{"a":1}]}').length, 1);
});
check('kev decision 展开为逐问题行，label 统一成字符串', function () {
  const g = E.extractGold([{ state: 's', questions: {
    intent: { type: 'choice', label: 'billing' },
    urgent: { type: 'noul', label: true },
    level: { type: 'score', label: 2 }
  } }]);
  assert.strictEqual(g.kind, 'classification');
  assert.strictEqual(g.rows.length, 3);
  assert.deepStrictEqual(g.rows.map(r => r.gold), ['billing', 'true', '2']);
});
check('alpaca / OpenAI messages / sharegpt 都能抽到 output', function () {
  assert.strictEqual(E.extractGold([{ instruction: 'q', output: 'a' }]).rows[0].gold, 'a');
  assert.strictEqual(E.extractGold([{ messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }] }]).rows[0].gold, 'a');
  assert.strictEqual(E.extractGold([{ conversations: [{ from: 'human', value: 'q' }, { from: 'gpt', value: 'a' }] }]).rows[0].gold, 'a');
});

console.log('\n=== 2. 三个引擎的输出解析 ===');
const kevRows = JSON.stringify([
  { id: 'r0', type: 'choice', keys: ['a', 'b'], label: 1, p: [0.2, 0.8] },
  { id: 'r1', type: 'noul', keys: ['false', 'true'], label: 0, p: [0.9, 0.1] },
  { id: 'r2', type: 'score', keys: ['lo', 'mid', 'hi'], label: 2, p: [0.1, 0.2, 0.7] }
]);
const kevReport = JSON.stringify({ clean: { acc: 1, n: 3, ece: 0.12, brier: 0.3, nll: 0.4, mean_conf: 0.8 } });
check('kev rows.json argmax → 统一预测，官方报告指标保留', function () {
  const p = E.parseKevOutput(kevRows, kevReport);
  assert.deepStrictEqual(p.rows.map(r => r.prediction), ['b', 'false', 'hi']);
  assert.strictEqual(p.official.accuracy, 1); close(p.official.ece, 0.12);
});
check('kev 官方 acc 与平台不一致时给出显式警告', function () {
  const wrong = JSON.stringify({ clean: { acc: 0.1, n: 3 } });
  const p = E.parseKevOutput(kevRows, wrong);
  const r = E.assembleResult({ id: 'e1', jobId: 'j1', engine: 'kev', startedAt: 1, finishedAt: 2,
    gold: { kind: 'classification', rows: [] }, parsed: p, leakage: { evalCount: 3, duplicated: 0, ratio: 0 } });
  assert.ok(r.warnings.some(w => /不一致/.test(w)));
});
check('LLaMA-Factory generated_predictions 按顺序对齐，缺失按弃权', function () {
  const gold = [{ id: '0', gold: 'x' }, { id: '1', gold: 'y' }];
  const p = E.parseLfOutput('{"prompt":"q1","label":"x","predict":"x"}\n', gold);
  assert.strictEqual(p.rows[0].prediction, 'x');
  assert.strictEqual(p.rows[1].prediction, null);
});
check('import 支持按 id 对齐与自带 gold', function () {
  const gold = [{ id: 'a', gold: 'x' }, { id: 'b', gold: 'y' }];
  const p = E.parseImported('{"id":"b","prediction":"Y"}\n{"id":"a","prediction":"x"}', gold);
  assert.strictEqual(p.rows[0].prediction, 'x');
  assert.strictEqual(p.rows[1].prediction, 'Y');
});

console.log('\n=== 3. 结果组装：口径、CI、泄漏、可比性 ===');
check('生成任务指标与 Wilson CI 齐全', function () {
  const parsed = E.parseImported('{"prediction":"cat sat"}\n{"prediction":"dog"}',
    [{ id: '0', gold: 'the cat sat' }, { id: '1', gold: 'cat sat' }]);
  const r = E.assembleResult({ id: 'e', jobId: 'j', engine: 'import', startedAt: 1, finishedAt: 2,
    gold: { kind: 'generation', rows: [] }, parsed: parsed,
    leakage: { evalCount: 2, duplicated: 0, ratio: 0 },
    evalSet: { id: 'd', name: 'd', sha256: 'abc', count: 2 } });
  assert.strictEqual(r.metrics.taskType, 'generation');
  assert.strictEqual(r.metrics.n, 2);
  close(r.metrics.exactMatch, 0.5);
  assert.ok(r.metrics.accuracyCI95.low != null);
  assert.ok(r.warnings.length >= 2, '应包含生成类口径提示与小样本提示');
  assert.ok(r.details.perSample.length === 2);
});
check('byType 给出 choice/noul/score 分组精度', function () {
  const p = E.parseKevOutput(kevRows, kevReport);
  const r = E.assembleResult({ id: 'e', jobId: 'j', engine: 'kev', startedAt: 1, finishedAt: 2,
    gold: { kind: 'classification', rows: [] }, parsed: p, leakage: { evalCount: 3, duplicated: 0, ratio: 0 } });
  assert.deepStrictEqual(Object.keys(r.byType).sort(), ['choice', 'noul', 'score']);
  close(r.byType.choice.accuracy, 1);
});
check('训练/评估重复与评估集指纹变化都进 warnings', function () {
  const p = E.parseImported('{"prediction":"x"}', [{ id: '0', gold: 'x' }]);
  const r = E.assembleResult({ id: 'e', jobId: 'j', engine: 'import', startedAt: 1, finishedAt: 2,
    gold: { kind: 'generation', rows: [] }, parsed: p,
    leakage: { evalCount: 1, duplicated: 1, ratio: 1 },
    evalSet: { id: 'd', name: 'd', sha256: 'new' }, previousHash: 'old' });
  assert.ok(r.warnings.some(w => /虚高/.test(w)));
  assert.ok(r.warnings.some(w => /指纹/.test(w)));
});
check('空标准答案直接抛错，不产出假分数', function () {
  assert.throws(function () {
    E.assembleResult({ id: 'e', jobId: 'j', engine: 'import', startedAt: 1,
      gold: { kind: 'generation', rows: [] }, parsed: { rows: [] } });
  }, /没有可用的标准答案/);
});

console.log('\n=== 4. 端到端闭环（内存 IO）：回写 job / 历史 / 失败降级 ===');
async function main() {
  const store = {};
  const datasets = {
    'ds-train': [{ instruction: 'q1', output: 'x' }],
    'ds-eval': [{ instruction: 'q1', output: 'x' }, { instruction: 'q2', output: 'y' }]
  };
  const job = { id: 'job-eval-test', datasetId: 'ds-train', datasetName: 'train',
    status: 'completed', config: {}, trainMode: 'llamafactory' };
  store[job.id] = Object.assign({}, job);
  let failNext = false;
  const io = {
    readJob: function (id) { return store[id]; },
    saveJob: async function (j) { store[j.id] = j; return j; },
    datasetById: function (id) { return { id: id, name: id, kind: 'sft', fileName: id + '.json' }; },
    readDatasetSamples: async function (ds) { return datasets[ds.id] || []; },
    obtain: async function (plan) {
      if (failNext) throw new Error('GPU OOM（模拟）');
      return { engine: 'import', taskType: 'generation',
        predictionsText: JSON.stringify([{ prediction: 'x' }, { prediction: 'z' }]), artifacts: { source: 'test' } };
    }
  };
  async function wait(taskId) {
    for (let i = 0; i < 100; i++) {
      const t = E.getTask(taskId);
      if (t && (t.state === 'succeeded' || t.state === 'failed')) return t;
      await sleep(15);
    }
    throw new Error('评估任务超时');
  }

  const t1 = E.startEvaluation(job, { evalDatasetId: 'ds-eval', engine: 'import' }, io);
  const done1 = await wait(t1.id);
  await check('评估成功且 job.evaluation 写入结构化结果', function () {
    assert.strictEqual(done1.state, 'succeeded');
    const ev = store[job.id].evaluation;
    assert.strictEqual(ev.status, 'succeeded');
    assert.strictEqual(ev.engine, 'import');
    assert.strictEqual(ev.metrics.n, 2);
    close(ev.metrics.exactMatch, 0.5);
    assert.strictEqual(ev.evalDataset.sha256.length, 64);
    assert.ok(ev.leakage.ratio > 0, '评估集与训练集有 1 条重复，应被记录');
    assert.strictEqual(store[job.id].status, 'completed', '训练状态不能被评估改写');
  });

  const t2 = E.startEvaluation(store[job.id], { evalDatasetId: 'ds-eval', engine: 'import' }, io);
  const done2 = await wait(t2.id);
  await check('重跑成功后上一版结果进入 evaluationHistory', function () {
    assert.strictEqual(done2.state, 'succeeded');
    const h = store[job.id].evaluationHistory || [];
    assert.ok(h.length >= 1, '历史为空');
    assert.strictEqual(h[0].status, 'succeeded');
  });

  failNext = true;
  const t3 = E.startEvaluation(store[job.id], { evalDatasetId: 'ds-eval', engine: 'import' }, io);
  const done3 = await wait(t3.id);
  await check('重跑失败不覆盖上一次成功结果，只记 lastError', function () {
    assert.strictEqual(done3.state, 'failed');
    const ev = store[job.id].evaluation;
    assert.strictEqual(ev.status, 'succeeded');
    assert.ok(/GPU OOM/.test(ev.lastError));
    assert.strictEqual(store[job.id].status, 'completed');
  });

  console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
  process.exit(failed ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
