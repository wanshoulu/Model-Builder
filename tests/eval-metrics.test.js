/**
 * 评估指标内核回归测试：口径对拍 sklearn / SQuAD / Wilson 明确算例。
 */
'use strict';
const assert = require('assert');
const M = require('../routes/eval-metrics');

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}
function close(a, b, eps) { assert.ok(Math.abs(a - b) < (eps || 1e-6), a + ' ≠ ' + b); }

console.log('=== 1. 分类指标：对拍 sklearn / HF evaluate 文档算例 ===');
// HF evaluate f1 card 示例：references=[0,1,0,1,0], predictions=[0,0,1,1,0]
const rows = [
  { gold: '0', prediction: '0' }, { gold: '1', prediction: '0' },
  { gold: '0', prediction: '1' }, { gold: '1', prediction: '1' },
  { gold: '0', prediction: '0' }
];
const cls = M.classificationMetrics(rows);
check('accuracy = correct/n = 0.6', function () { close(cls.accuracy, 0.6); assert.strictEqual(cls.correct, 3); });
check('micro-F1 单标签多类下等于 accuracy', function () { close(cls.micro.f1, 0.6); });
check('macro-F1 = 7/12（逐类等权，sklearn 口径）', function () {
  // class0: TP=2,FP=1,FN=1 → P=R=2/3,F1=2/3；class1: TP=1,FP=1,FN=1 → F1=0.5
  close(cls.macro.f1, (2 / 3 + 0.5) / 2);
  close(cls.perClass[0].f1, 2 / 3); close(cls.perClass[1].f1, 0.5);
});
check('weighted-F1 按 support 加权 = 0.6', function () { close(cls.weighted.f1, ((2 / 3) * 3 + 0.5 * 2) / 5); });
check('混淆矩阵按 gold 行 / prediction 列，含弃权列', function () {
  assert.strictEqual(cls.confusion.labels.length, 3);
  const sum = cls.confusion.matrix.reduce((a, r) => a + r.reduce((x, y) => x + y, 0), 0);
  assert.strictEqual(sum, 5);
});
check('Wilson 95% CI 包住点估计，且 p=0 时上界仍 > 0', function () {
  assert.ok(cls.ci95.low < 0.6 && cls.ci95.high > 0.6);
  close(cls.ci95.low, 0.2307, 0.002); close(cls.ci95.high, 0.8824, 0.002);
  const zero = M.wilsonInterval(0, 5);
  assert.strictEqual(zero.low, 0); assert.ok(zero.high > 0);
  const one = M.wilsonInterval(5, 5);
  assert.ok(one.low < 1 && one.high === 1);
});
check('弃权样本计入分母、不算正确，coverage 单列', function () {
  const c = M.classificationMetrics([{ gold: 'A', prediction: 'A' }, { gold: 'A', prediction: null }]);
  close(c.accuracy, 0.5); close(c.coverage, 0.5); close(c.accuracyCovered, 1);
});
check('空数据集不产生 NaN', function () {
  const c = M.classificationMetrics([]);
  assert.strictEqual(c.accuracy, null); assert.strictEqual(c.ci95.low, null);
});

console.log('\n=== 2. 生成指标：SQuAD 归一化 / bag-of-tokens F1 / ROUGE-L ===');
check('normalizeAnswer 小写、去标点、去冠词、折叠空白', function () {
  assert.strictEqual(M.normalizeAnswer('  The   CAT, sat! '), 'cat sat');
  assert.strictEqual(M.normalizeAnswer('ＡＢＣ １２３'), 'abc 123');   // NFKC 全角
});
check('中文按字切 token（不因无空格退化成 1 个 token）', function () {
  const t = M.tokenize('信用卡账单怎么查');
  assert.ok(t.length >= 7, JSON.stringify(t));
});
check('冠词/标点/大小写差异不影响 exact_match', function () {
  const g = M.generationMetrics([{ gold: 'The cat sat.', prediction: 'a cat sat' }]);
  close(g.exactMatch, 1);   // SQuAD normalize 去掉冠词 a / the 与标点
});
check('token F1 与 ROUGE-L 对 "cat sat" vs "cat" = 2/3', function () {
  const g = M.generationMetrics([{ gold: 'the cat sat', prediction: 'cat' }]);
  close(g.tokenF1, 2 / 3); close(g.rougeL, 2 / 3);
});
check('多参考答案取逐样本最大值，再 macro 平均', function () {
  const g = M.generationMetrics([
    { gold: ['答案甲', '答案乙'], prediction: '答案乙' },
    { gold: '答案丙', prediction: '答案丁' }
  ]);
  close(g.exactMatch, 0.5); close(g.tokenF1, (1 + 2 / 3) / 2);
});
check('空预测按 0 分计入，不被静默丢弃', function () {
  const g = M.generationMetrics([{ gold: 'x', prediction: null }]);
  close(g.exactMatch, 0); close(g.coverage, 0); close(g.tokenF1, 0);
});

console.log('\n=== 3. 评估集指纹与泄漏检测 ===');
check('同一内容指纹稳定，字段顺序不影响', function () {
  assert.strictEqual(M.sampleFingerprint({ a: 1, b: 2 }), M.sampleFingerprint({ b: 2, a: 1 }));
});
check('decision 样本用 state+questions 做指纹', function () {
  const s = { state: 'x', questions: { q: { type: 'noul', label: true } } };
  const b = { state: 'x', questions: { q: { type: 'noul', label: false } } };
  assert.notStrictEqual(M.sampleFingerprint(s), M.sampleFingerprint(b));
});
check('评估集指纹与顺序无关（集合语义）', function () {
  assert.strictEqual(M.evalSetFingerprint([{ a: 1 }, { a: 2 }]).sha256,
    M.evalSetFingerprint([{ a: 2 }, { a: 1 }]).sha256);
});
check('训练/评估集重复率正确识别', function () {
  const l = M.leakage([{ q: 'a' }, { q: 'b' }], [{ q: 'a' }, { q: 'c' }]);
  assert.strictEqual(l.duplicated, 1); close(l.ratio, 0.5);
});

console.log('=== record-level bootstrap CI ===');
check('同记录多问题的 bootstrap 区间比 Wilson 宽', function () {
  // 60 条记录×3 问题，正确率 0.8167（真实数据同构）
  const rows = [];
  for (let r = 0; r < 60; r++) {
    for (let q = 0; q < 3; q++) {
      rows.push({ id: r + ':f' + q, gold: 'a', prediction: (r % 7 === 0 && q !== 0) ? 'b' : 'a', correct: !(r % 7 === 0 && q !== 0) });
    }
  }
  const boot = M.bootstrapRecordCI(rows, 500);
  const wil = M.classificationMetrics(rows);
  assert.ok(boot && boot.low != null && boot.nRecords === 60);
  assert.ok(boot.high - boot.low >= wil.ci95.high - wil.ci95.low - 0.001,
    'bootstrap 区间应不窄于 Wilson：' + (boot.high - boot.low).toFixed(3) + ' vs ' + (wil.ci95.high - wil.ci95.low).toFixed(3));
});
check('确定性（同 seed 同结果）', function () {
  const rows = Array.from({ length: 30 }, (_, i) => ({ id: Math.floor(i / 2) + ':x', gold: 'a', prediction: 'a', correct: true }));
  const a = M.bootstrapRecordCI(rows, 300), b = M.bootstrapRecordCI(rows, 300);
  assert.strictEqual(a.low, b.low); assert.strictEqual(a.high, b.high);
});
check('无分组信息返回 null', function () {
  assert.strictEqual(M.bootstrapRecordCI([{ gold: 'a', prediction: 'a', correct: true }]), null);
});

console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
process.exit(failed ? 1 : 0);