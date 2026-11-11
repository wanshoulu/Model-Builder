/**
 * 数据集导入→质检→转换 回归测试
 *
 * 背景（真实缺陷）：convertTo 曾在 qualityFunnel 之后对已归一化的样本
 * 再次 normalize，导致 sharegpt / openai_messages / dpo_standard / qa_pairs
 * 四种布局被二次映射成全空字段；质检报告却显示"全部通过"，空样本被静默
 * 注册进训练集。本测试覆盖 8 种布局，断言字段非空且原文不丢失，并验证
 * 误传 sourceLayout 时 fail-closed 抛错而非静默产出空数据。
 */
'use strict';
const assert = require('assert');
const DT = require('../routes/dataset-tools');

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}

const CASES = [
  {
    layout: 'alpaca', kind: 'sft', fields: ['instruction', 'output'], expect: '银行卡被吞',
    samples: [
      { instruction: '我的银行卡被吞了怎么办', input: '', output: '请立即拨打客服热线挂失' },
      { instruction: '如何修改预留手机号', input: '', output: '请携带身份证前往网点办理' },
      { instruction: '转账单笔限额是多少', input: '', output: '默认单笔上限为五万元' }
    ]
  },
  {
    layout: 'alpaca_preference', kind: 'preference', fields: ['instruction', 'chosen', 'rejected'], expect: '银行卡',
    samples: [
      { instruction: '银行卡被吞了怎么办', input: '', chosen: '拨打客服挂失并保留凭条', rejected: '用力拍打机器' },
      { instruction: '忘记密码如何处理', input: '', chosen: '通过身份证重置密码', rejected: '随便试几次密码' },
      { instruction: '如何查询账户余额', input: '', chosen: '登录手机银行查看', rejected: '把卡号告诉陌生人' }
    ]
  },
  {
    layout: 'dpo_standard', kind: 'preference', fields: ['instruction', 'chosen', 'rejected'], expect: '吞卡',
    samples: [
      { prompt: '客户遇到吞卡', chosen: '引导客户致电客服挂失', rejected: '让客户自行撬开机器' },
      { prompt: '客户忘记查询密码', chosen: '引导柜台凭身份证重置', rejected: '建议客户多试几个生日' },
      { prompt: '客户咨询汇率', chosen: '提供当日牌价并说明浮动', rejected: '随便报一个数字' }
    ]
  },
  {
    layout: 'sharegpt', kind: 'sft', fields: ['instruction', 'output'], expect: '银行卡被吞',
    samples: [
      { conversations: [{ from: 'human', value: '我的银行卡被吞了怎么办' }, { from: 'gpt', value: '请拨打客服热线挂失' }] },
      { conversations: [{ from: 'user', value: '如何修改预留手机号' }, { from: 'assistant', value: '携带身份证到网点办理' }] },
      { conversations: [{ from: 'human', value: '转账单笔限额多少' }, { from: 'gpt', value: '单笔上限为五万元' }] }
    ]
  },
  {
    layout: 'openai_messages', kind: 'sft', fields: ['instruction', 'output'], expect: '吞卡',
    samples: [
      { messages: [{ role: 'user', content: '遇到吞卡应该联系谁' }, { role: 'assistant', content: '联系发卡行客服并登记' }] },
      { messages: [{ role: 'user', content: '网点营业时间怎么查' }, { role: 'assistant', content: '在手机银行网点地图查询' }] },
      { messages: [{ role: 'user', content: '跨境汇款需要什么材料' }, { role: 'assistant', content: '身份证与收款账户信息' }] }
    ]
  },
  {
    layout: 'qa_pairs', kind: 'sft', fields: ['instruction', 'output'], expect: '吞卡',
    samples: [
      { question: '遇到吞卡怎么办', answer: '保留凭条并致电客服' },
      { q: '如何开通手机银行', a: '持身份证到柜台签约' },
      { question: '定期存款能提前支取吗', answer: '可以，按活期利率计息' }
    ]
  },
  {
    layout: 'pretrain_text', kind: 'pretrain', fields: ['text'], expect: '吞卡',
    samples: [
      { text: '遇到吞卡时，持卡人应保留 ATM 凭条并立即联系发卡行客服登记。' },
      { text: '手机银行支持转账、缴费、理财与账户明细查询等常用功能。' },
      { text: '跨境汇款业务需核对收款人姓名、账号与开户行 SWIFT 代码。' }
    ]
  },
  {
    layout: 'decision_kev', kind: 'decision', fields: [], expect: '',
    samples: [
      { state: '客户咨询吞卡处理', questions: { intent: { type: 'choice', label: 'card_retained' } } },
      { state: '客户询问账户余额', questions: { intent: { type: 'choice', label: 'balance_query' } } },
      { state: '客户要求转账限额', questions: { intent: { type: 'choice', label: 'transfer_limit' } } }
    ]
  }
];

function runPipeline(c) {
  const sniff = DT.sniffLayout(c.samples);
  const normalized = DT.normalize(c.samples, sniff.layout);
  const funnel = DT.qualityFunnel(normalized, { targetKind: c.kind });
  const converted = DT.convertTo(funnel.samples, c.kind);
  return { sniff, normalized, funnel, converted };
}

console.log('=== 1. 8 种布局：嗅探 / 质检 / 单次转换内容与字段完整性 ===');
CASES.forEach(function (c) {
  check(c.layout + ' → ' + c.kind, function () {
    const r = runPipeline(c);
    assert.strictEqual(r.sniff.layout, c.layout, '布局嗅探应为 ' + c.layout + '，实际 ' + r.sniff.layout);
    assert.ok(r.funnel.passed > 0, '质检不应把全部样本过滤掉');
    assert.strictEqual(r.converted.length, r.funnel.passed, '转换条数应与质检通过数一致');
    r.converted.forEach(function (s, i) {
      c.fields.forEach(function (k) {
        assert.ok(s[k] != null && String(s[k]).trim() !== '',
          '第 ' + i + ' 条 ' + k + ' 为空（历史缺陷：二次归一化）');
      });
    });
    if (c.expect) {
      const blob = r.converted.map(s => JSON.stringify(s)).join('\n');
      assert.ok(blob.indexOf(c.expect) >= 0,
        '转换结果未保留原文关键词 "' + c.expect + '"（字段可能被映射丢失）：' + blob.slice(0, 200));
    }
  });
});

console.log('\n=== 2. fail-closed：误传 sourceLayout（模拟历史双重归一化）必须抛错 ===');
['sharegpt', 'openai_messages', 'dpo_standard', 'qa_pairs'].forEach(function (layout) {
  check(layout + ' 二次归一化被拦截', function () {
    const c = CASES.find(function (x) { return x.layout === layout; });
    const r = runPipeline(c);
    assert.throws(function () { DT.convertTo(r.funnel.samples, c.kind, layout); },
      /中止注册|缺少/, '应抛出明确的空字段错误，而不是静默返回空样本');
  });
});

console.log('\n=== 3. 显式传 sourceLayout 时对原始样本仍能正确归一化（向后兼容） ===');
CASES.forEach(function (c) {
  check(c.layout + ' 显式源布局', function () {
    const out = DT.convertTo(c.samples, c.kind, c.layout);
    assert.strictEqual(out.length, c.samples.length);
    c.fields.forEach(function (k) {
      out.forEach(function (s, i) {
        assert.ok(s[k] != null && String(s[k]).trim() !== '', '第 ' + i + ' 条 ' + k + ' 为空');
      });
    });
  });
});

console.log('\n=== 4. 空字段输入直接中止，不会注册空数据集 ===');
check('全空 alpaca 样本被拒绝', function () {
  assert.throws(function () {
    DT.convertTo([{ instruction: '', input: '', output: '' }], 'sft');
  }, /中止注册/);
});
check('局部空样本也整体中止', function () {
  assert.throws(function () {
    DT.convertTo([
      { instruction: '正常问题', input: '', output: '正常回答内容' },
      { instruction: '', input: '', output: '' }
    ], 'sft');
  }, /1\/2 条/);
});

console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
process.exit(failed ? 1 : 0);
