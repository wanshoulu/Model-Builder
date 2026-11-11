/**
 * 「训练 → 自动评估 → 实验报告」闭环回归测试。
 * 在真实 jobs 目录里放一个临时任务（带 job.evaluation），调用报告 analyze()，
 * 验证留出集精度自动进入指标目录 / 统计 / 警告 / 方法口径，结束后还原。
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert');
const R = require('../routes/experiment-report');

const JOBS_DIR = path.join(__dirname, '..', 'data', 'model-training', 'jobs');
fs.mkdirSync(JOBS_DIR, { recursive: true });
const jobId = 'job-eval-loop-test';
const file = path.join(JOBS_DIR, jobId + '.json');
const backup = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : null;

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}

const job = {
  id: jobId, name: '闭环测试任务', status: 'completed', createdAt: Date.now(),
  datasetId: 'ds-train', datasetName: 'train-set', config: { modelName: 'Test/Model-0.6B', stage: 'sft' },
  lossHistory: [{ loss: 2.0, epoch: 1 }, { loss: 1.0, epoch: 2 }],
  logTail: "{'train_runtime': '120.0'}",
  evaluation: {
    id: 'eval-x', jobId: jobId, engine: 'kev', status: 'succeeded',
    startedAt: 1, finishedAt: 2,
    evalDataset: { id: 'ds-eval', name: 'heldout-v1', kind: 'decision', count: 50, sha256: 'a'.repeat(64) },
    trainDataset: { id: 'ds-train', name: 'train-set' },
    leakage: { evalCount: 50, duplicated: 0, ratio: 0 },
    metrics: {
      taskType: 'classification', n: 50, correct: 40, coverage: 1,
      accuracy: 0.8, accuracyCI95: { low: 0.67, high: 0.89, method: 'Wilson' },
      macroF1: 0.77, weightedF1: 0.79, calibration: { ece: 0.05, brier: 0.3, nll: 0.5, accuracy: 0.8 }
    },
    warnings: ['生成类任务：文本重合度不等于事实正确性'],
    method: { classification: 'sklearn 口径' }
  }
};
fs.writeFileSync(file, JSON.stringify(job, null, 2), 'utf8');

try {
  console.log('=== 1. 自动评估指标进入报告 catalog（不是手工录入）===');
  const res = R.analyze({ runIds: [jobId], metricNames: ['准确率（留出集）'] });
  check('catalog 含留出集 accuracy，source=auto-eval', function () {
    const c = res.catalog.find(x => x.name === '准确率（留出集）');
    assert.ok(c, '未找到自动评估指标');
    assert.strictEqual(c.source, 'auto-eval');
  });
  check('统计值 0.8 来自 job.evaluation，未被重新计算或臆造', function () {
    const s = res.stats['准确率（留出集）'][0].stats;
    assert.strictEqual(s.n, 1);
    assert.strictEqual(s.mean, 0.8);
  });
  check('run 暴露评估集指纹 / 引擎作为可比 facet', function () {
    const r = res.runs.find(x => x.id === jobId);
    assert.ok(r && r.facets);
    assert.strictEqual(r.facets.evalSet, 'aaaaaaaa');
    assert.strictEqual(r.facets.evalEngine, 'kev');
  });
  check('methods 新增自动评估口径（Wilson / sklearn / SQuAD）', function () {
    assert.ok(/Wilson/.test(res.methods.autoEval));
    assert.ok(/sklearn/.test(res.methods.autoEval));
    assert.ok(res.methods.sources.some(s => /llamafactory\.readthedocs/i.test(s)));
  });
  check('不再误报「只有训练日志代理指标」', function () {
    assert.ok(!res.warnings.some(w => /只选择了训练日志自动代理指标/.test(w)));
  });

  console.log('\n=== 2. 可比性警告：评估集不同 / 泄漏 / 无评估 ===');
  const job2Id = 'job-eval-loop-test2';
  const file2 = path.join(JOBS_DIR, job2Id + '.json');
  const backup2 = fs.existsSync(file2) ? fs.readFileSync(file2, 'utf8') : null;
  const job2 = JSON.parse(JSON.stringify(job));
  job2.id = job2Id;
  job2.evaluation.evalDataset.sha256 = 'b'.repeat(64);
  job2.evaluation.engine = 'llamafactory';
  job2.evaluation.leakage = { evalCount: 50, duplicated: 10, ratio: 0.2 };
  fs.writeFileSync(file2, JSON.stringify(job2, null, 2), 'utf8');
  try {
    const res2 = R.analyze({ runIds: [jobId, job2Id], metricNames: ['准确率（留出集）'] });
    check('不同评估集指纹给出不可比警告', function () {
      assert.ok(res2.warnings.some(w => /指纹不一致/.test(w)));
    });
    check('引擎不一致给出分别解读警告', function () {
      assert.ok(res2.warnings.some(w => /引擎不一致/.test(w)));
    });
    check('训练/评估重复样本给出虚高警告', function () {
      assert.ok(res2.warnings.some(w => /重复样本/.test(w)));
    });
  } finally {
    if (backup2 == null) { try { fs.unlinkSync(file2); } catch (e) {} }
    else fs.writeFileSync(file2, backup2, 'utf8');
  }

  console.log('\n=== 3. 没有评估的任务仍给明确提示 ===');
  const job3Id = 'job-eval-loop-test3';
  const file3 = path.join(JOBS_DIR, job3Id + '.json');
  const job3 = JSON.parse(JSON.stringify(job));
  job3.id = job3Id; delete job3.evaluation;
  fs.writeFileSync(file3, JSON.stringify(job3, null, 2), 'utf8');
  try {
    const res3 = R.analyze({ runIds: [job3Id], metricNames: ['Final Train Loss'] });
    check('提示该任务没有自动评估结果（仅 loss）', function () {
      assert.ok(res3.warnings.some(w => /没有自动评估结果/.test(w)));
    });
  } finally {
    try { fs.unlinkSync(file3); } catch (e) {}
  }
} catch (e) {
  failed++;
  console.log('  FAIL 报告分析抛错 → ' + (e && e.stack || e));
} finally {
  if (backup == null) { try { fs.unlinkSync(file); } catch (e) {} }
  else fs.writeFileSync(file, backup, 'utf8');
  console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
  process.exit(failed ? 1 : 0);
}
