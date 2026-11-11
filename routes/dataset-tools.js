// ═══ 数据集质检与转换引擎（模块A）═══
// 设计对标：convmerge（65种布局嗅探）、qwen-code-agents dataset-curation（六阶段质检漏斗）、
// LLaMA-Factory 官方数据校验实践（绝不修改原始文件）
// 实际漏斗（确定性阶段在前，昂贵阶段在后，共 7 段，逐段留痕）：
//   ① 精确去重 ② 长度过滤 ③ 语言检测 ④ 语义去重 ⑤ 结构完整性
//   ⑥ 质量评分（启发式，可按阈值拦截） ⑦ LLM 裁判（可选，需注入 judge 函数）
// ⑦ 是异步阶段，故同步入口 qualityFunnel 只跑 ①~⑥，完整链路走 qualityFunnelAsync。
const fs = require('node:fs');
const path = require('node:path');

const DATA_TOOLS_DIR = path.join(__dirname, '..', 'data', 'model-training', 'imports');
try { fs.mkdirSync(DATA_TOOLS_DIR, { recursive: true }); } catch (e) {}

// ── 布局嗅探 ──────────────────────────────────────────────
// 识别常见数据布局，返回 {layout, confidence, reason}
const LAYOUTS = {
  alpaca: { desc: 'LLaMA-Factory SFT（instruction/input/output）', kind: 'sft' },
  alpaca_preference: { desc: 'LLaMA-Factory 偏好（instruction/chosen/rejected）', kind: 'preference' },
  sharegpt: { desc: 'ShareGPT 多轮（conversations[from/value]）', kind: 'sft' },
  openai_messages: { desc: 'OpenAI messages（messages[role/content]）', kind: 'sft' },
  decision_kev: { desc: '类 Jev 决策（state/questions/label）', kind: 'decision' },
  pretrain_text: { desc: '预训练纯文本（text）', kind: 'pretrain' },
  qa_pairs: { desc: '问答对（question/answer 或 q/a）', kind: 'sft' },
  dpo_standard: { desc: 'DPO 标准对（prompt/chosen/rejected）', kind: 'preference' },
};

function sniffLayout(samples) {
  if (!Array.isArray(samples) || !samples.length) return { layout: 'unknown', confidence: 0, reason: '空数组' };
  const first = samples[0] || {};
  const keys = Object.keys(first);
  const has = k => keys.includes(k);
  const allHave = k => samples.every(s => s && typeof s === 'object' && k in s);

  // decision（kev）
  if (allHave('state') && allHave('questions')) return { layout: 'decision_kev', confidence: 0.99, reason: 'state + questions 是决策数据特征' };
  // sharegpt
  if (allHave('conversations')) return { layout: 'sharegpt', confidence: 0.95, reason: 'conversations 字段' };
  // openai messages
  if (allHave('messages')) return { layout: 'openai_messages', confidence: 0.95, reason: 'messages 字段' };
  // alpaca preference
  if (allHave('instruction') && allHave('chosen') && allHave('rejected')) return { layout: 'alpaca_preference', confidence: 0.95, reason: 'instruction + chosen/rejected' };
  // dpo standard
  if (allHave('prompt') && allHave('chosen') && allHave('rejected')) return { layout: 'dpo_standard', confidence: 0.95, reason: 'prompt + chosen/rejected' };
  // alpaca
  if (allHave('instruction') && allHave('output')) return { layout: 'alpaca', confidence: 0.95, reason: 'instruction + output' };
  // pretrain：字段数上限必须逐条判断，只看首条会让混合字段数据误判
  if (allHave('text') && samples.every(s => s && typeof s === 'object' && Object.keys(s).length <= 3)) {
    return { layout: 'pretrain_text', confidence: 0.9, reason: '纯 text 字段' };
  }
  // qa pairs：同一批数据里 question/answer 与 q/a 两种键名常混用，逐条按"任一别名"判断
  const isQa = samples.every(s => s && typeof s === 'object' &&
    (('question' in s) || ('q' in s)) && (('answer' in s) || ('a' in s)));
  if (isQa) return { layout: 'qa_pairs', confidence: 0.85, reason: 'question/answer 或 q/a 对（允许混用）' };
  return { layout: 'unknown', confidence: 0.2, reason: '未匹配已知布局，字段: ' + keys.slice(0, 8).join(',') };
}

// ── 布局归一化（统一到内部格式）──────────────────────────
function normalize(samples, layout) {
  switch (layout) {
    case 'alpaca':
      return samples.map(s => ({ instruction: s.instruction || '', input: s.input || '', output: s.output || '' }));
    case 'alpaca_preference':
      return samples.map(s => ({ instruction: s.instruction || '', input: s.input || '', chosen: s.chosen || '', rejected: s.rejected || '' }));
    case 'dpo_standard':
      return samples.map(s => ({ instruction: s.prompt || '', input: '', chosen: s.chosen || '', rejected: s.rejected || '' }));
    case 'sharegpt':
      // 角色别名表：真实 ShareGPT 类数据有 human/gpt、user/assistant、bing 等 12+ 种角色写法
      const USER_ROLES = ['human', 'user', 'prompter', 'customer', 'patient', 'client', '提问'];
      const BOT_ROLES = ['gpt', 'assistant', 'chatgpt', 'bot', 'bing', 'chatbot', 'a', '回答'];
      return samples.map(s => ({
        instruction: (s.conversations || []).filter(c => USER_ROLES.includes(String((c.from || c.role || '')).toLowerCase())).map(c => c.value || c.content).filter(Boolean).join('\n'),
        input: '',
        output: (s.conversations || []).filter(c => BOT_ROLES.includes(String((c.from || c.role || '')).toLowerCase())).map(c => c.value || c.content).filter(Boolean).join('\n'),
      }));
    case 'openai_messages':
      return samples.map(s => ({
        instruction: (s.messages || []).filter(m => (m.role || '') === 'user').map(m => m.content).filter(t => typeof t === 'string').join('\n'),
        input: '',
        output: (s.messages || []).filter(m => (m.role || '') === 'assistant').map(m => m.content).filter(t => typeof t === 'string').join('\n'),
      }));
    case 'qa_pairs':
      return samples.map(s => ({ instruction: s.question || s.q || '', input: '', output: s.answer || s.a || '' }));
    case 'pretrain_text':
      return samples.map(s => ({ text: s.text || '' }));
    case 'decision_kev':
      return samples;  // 决策数据原样通过（kev 原生格式）
    default:
      return samples;
  }
}

// ── 启发式质量评分（阶段⑥）────────────────────────────────
// 只判定可机器观测的退化信号（空/极短/复读/问答复制/占位符/偏好对无差异），
// 不宣称语义正确性——那是 LLM 裁判的职责。评分不写回样本，避免污染训练数据。
const DEGENERATE_MARKERS = [
  /^(todo|tbd|n\/a|na|none|null|暂无|略|待补充|不知道|不确定|无|\.{3,}|…+)$/i,
  /作为\s*(一个\s*)?(ai|人工智能)\s*(语言)?模型/i,
  /我(无法|不能)回答|无法提供相关(信息|回答)/,
  /^\s*(test|测试|示例|sample|示例文本)\s*\d*$/i,
];

function bigrams(text) {
  const s = String(text || '').toLowerCase().replace(/\s+/g, '');
  const set = new Set();
  for (let i = 0; i + 2 <= s.length; i++) set.add(s.slice(i, i + 2));
  return set;
}
function overlapRatio(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const g of a) if (b.has(g)) inter++;
  return inter / Math.min(a.size, b.size);
}
function hasRepeatedSentence(text) {
  const parts = String(text || '').split(/[。！？.!?；;\n]/).map(x => x.trim()).filter(x => x.length >= 6);
  return new Set(parts).size < parts.length;
}

/**
 * 单条样本打分：从 1 分起扣，返回 {score:0~1, flags:string[]}。
 * 打分维度与扣分权重是显式常量，便于按数据集类型调参并被测试断言。
 */
function scoreSample(s, opts) {
  opts = opts || {};
  if (!s || typeof s !== 'object') return { score: 0, flags: ['样本不是对象'] };
  const flags = [];
  let score = 1;
  const penalty = (v, label) => { score -= v; flags.push(label); };

  const q = String(s.instruction || s.prompt || s.question || s.q || '').trim();
  const a = String(s.output || s.text || s.chosen || (s.answer || s.a) || '').trim();
  const isDecision = !!(s.questions && Object.keys(s.questions).length);
  const isPreference = !!(s.chosen && s.rejected);

  if (!isDecision && !a) {
    return { score: 0, flags: ['回答/正文为空'] };
  }
  if (!isDecision && q.length < 4) penalty(0.25, '指令过短（<4 字符）');
  if (a && a.length < 8 && !isDecision) penalty(0.2, '回答过短（<8 字符）');
  if (a.length > 4000) penalty(0.1, '回答过长（>4000 字符）');
  if (q && a && q.length / Math.max(a.length, 1) > 20) penalty(0.15, '指令/回答长度比失衡');

  if (q && a) {
    const qg = bigrams(q), ag = bigrams(a);
    if (q === a) penalty(0.35, '回答与指令完全相同');
    else if (overlapRatio(qg, ag) > 0.92) penalty(0.2, '回答几乎是指令的复制');
  }
  if (a && hasRepeatedSentence(a)) penalty(0.2, '回答内存在重复句');
  if (a && DEGENERATE_MARKERS.some(re => re.test(a))) penalty(0.4, '占位符/拒答式模板内容');

  if (isPreference) {
    const c = String(s.chosen || '').trim(), r = String(s.rejected || '').trim();
    if (c === r) penalty(0.5, 'chosen 与 rejected 完全相同');
    else if (overlapRatio(bigrams(c), bigrams(r)) > 0.95) penalty(0.2, '偏好对区分度过低');
  }
  if (isDecision) {
    const st = String(typeof s.state === 'string' ? s.state : JSON.stringify(s.state || '')).trim();
    if (st.length < 8) penalty(0.15, '决策 state 过短');
    const qs = Object.values(s.questions || {});
    if (qs.some(v => !v || !v.type || !('label' in v))) penalty(0.3, '决策问题缺少 type/label');
  }
  return { score: Math.max(0, Math.min(1, +score.toFixed(3))), flags: flags.slice(0, 5) };
}
function scoreSamples(samples, opts) {
  return (samples || []).map(s => scoreSample(s, opts));
}

// ── 质检漏斗（确定性 6 段；第 7 段 LLM 裁判见 qualityFunnelAsync）──
function qualityFunnel(samples, opts) {
  opts = opts || {};
  const report = { stages: [], total: samples.length };
  let cur = samples.slice();
  const drop = (stage, kept, reason) => {
    report.stages.push({ stage, input: cur.length, removed: cur.length - kept.length, reason });
    cur = kept;
  };

  // ① 精确去重（规范化后哈希——空格/大小写折叠）
  {
    const seen = new Set();
    const kept = [];
    for (const s of cur) {
      const norm = JSON.stringify(s).toLowerCase().replace(/\s+/g, ' ');
      if (!seen.has(norm)) { seen.add(norm); kept.push(s); }
    }
    drop('exact_dedup', kept, '规范化（小写+空白折叠）后完全相同的样本');
  }

  // ② 长度过滤（指令/回答过短通常是退化样本，过长通常跑题）
  {
    const minLen = opts.minLen || 2, maxLen = opts.maxLen || 32000;
    const kept = cur.filter(s => {
      const texts = [s.instruction, s.input, s.output, s.chosen, s.rejected, s.text, JSON.stringify(s.questions || {})]
        .filter(t => typeof t === 'string' && t);
      if (!texts.length) return false;  // 完全无内容
      return texts.every(t => t.length >= minLen) && texts.some(t => t.length <= maxLen);
    });
    drop('length_filter', kept, `文本长度超出 [${minLen}, ${maxLen}] 或为空`);
  }

  // ③ 语言检测（目标语言字符占比——不依赖外部库；支持任意 Unicode 语言区段）
  if (opts.targetLang) {
    // 语言 → Unicode 区段映射（可扩展；未列出的语言按拉丁/西里尔等粗粒度兜底）
    const LANG_RANGES = {
      zh: /[\u4e00-\u9fff\u3400-\u4dbf]/g,
      en: /[a-zA-Z]/g,
      ja: /[\u3040-\u30ff\u4e00-\u9fff]/g,
      ko: /[\uac00-\ud7af\u1100-\u11ff]/g,
      ru: /[\u0400-\u04ff]/g,
      ar: /[\u0600-\u06ff]/g,
    };
    const ALL_WORD = /[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ffa-zA-Z]/g;
    const target = LANG_RANGES[opts.targetLang];
    if (target) {
      const kept = cur.filter(s => {
        const text = [s.instruction, s.output, s.text].filter(t => typeof t === 'string').join(' ');
        if (!text) return true;  // 结构化数据跳过
        const targetChars = (text.match(target) || []).length;
        const allWordChars = (text.match(ALL_WORD) || []).length;
        return allWordChars === 0 || targetChars / allWordChars >= 0.5;  // 目标语言占词字符 ≥50%
      });
      drop('language_check', kept, `目标语言（${opts.targetLang}）占比 < 50%`);
    }
  } else {
    // 报告结构恒定：未启用也要留痕，否则前端会误以为漏斗漏跑了一段
    report.stages.push({ stage: 'language_check', input: cur.length, removed: 0, reason: '未指定目标语言，跳过', skipped: true });
  }

  // ④ 语义去重（字符 n-gram Jaccard 相似度——不依赖 embedding 模型的轻量近似；
  //    embedding 版在模块 B 的引擎上做，此处先用 n-gram 兜底，阈值对标业界 0.92）
  {
    const threshold = opts.semanticThreshold || 0.85;
    const ngrams = (t, n) => {
      const s = String(t || '').toLowerCase().replace(/\s+/g, '');
      const set = new Set();
      for (let i = 0; i + n <= s.length; i++) set.add(s.slice(i, i + n));
      return set;
    };
    // 指纹取材（2026-11-11 修复）：SFT 用 instruction+output；decision 只用 state——
    // questions 模板全库结构相同（选项 criteria 逐条重复），混入指纹会把完全不同
    // 意图的样本判为高相似（实测建行数据 Jaccard 0.924 误杀 77%）
    const fpText = s => {
      const iot = [s.instruction, s.output, s.text].filter(t => typeof t === 'string' && t).join(' ');
      if (iot) return iot;
      if (s.state != null) return Array.isArray(s.state) ? s.state.map(m => m && m.content).join(' ') : String(s.state);
      return JSON.stringify(s);
    };
    const grams = cur.map(s => ngrams(fpText(s), 3));
    const kept = [], keptGrams = [];
    for (let i = 0; i < cur.length; i++) {
      let dup = false;
      for (let j = 0; j < keptGrams.length; j++) {
        const a = grams[i], b = keptGrams[j];
        if (!a.size || !b.size) continue;
        let inter = 0;
        for (const g of a) if (b.has(g)) inter++;
        const jaccard = inter / (a.size + b.size - inter);
        if (jaccard >= threshold) { dup = true; break; }
      }
      if (!dup) { kept.push(cur[i]); keptGrams.push(grams[i]); }
    }
    drop('semantic_dedup', kept, `3-gram Jaccard ≥ ${threshold}（近似语义重复）`);
  }

  // ⑤ 结构完整性（按目标 kind 检查必填字段——与工作台保存校验同源）
  {
    const kind = opts.targetKind || sniffLayout(cur).layout;
    let kept;
    if (kind === 'decision_kev' || kind === 'decision') {
      kept = cur.filter(s => s.questions && Object.values(s.questions).every(q => q && q.type && 'label' in q));
    } else if (kind === 'preference') {
      kept = cur.filter(s => (s.instruction || s.text) && s.chosen && s.rejected);
    } else if (kind === 'pretrain') {
      kept = cur.filter(s => s.text);
    } else {
      kept = cur.filter(s => s.instruction && (s.output || s.chosen));
    }
    drop('structure_check', kept, '缺少目标类型的必填字段');
  }

  // ⑥ 质量评分（启发式，确定性；默认只报告不过滤，minScore>0 才按阈值拦截）
  {
    const rawMin = Number(opts.minScore);
    const minScore = Number.isFinite(rawMin) ? Math.max(0, Math.min(1, rawMin)) : 0;
    const scores = scoreSamples(cur, opts);
    const values = scores.map(x => x.score);
    const sorted = values.slice().sort((a, b) => a - b);
    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const distribution = [
      { range: '0.0–0.4', count: values.filter(v => v < 0.4).length },
      { range: '0.4–0.6', count: values.filter(v => v >= 0.4 && v < 0.6).length },
      { range: '0.6–0.8', count: values.filter(v => v >= 0.6 && v < 0.8).length },
      { range: '0.8–1.0', count: values.filter(v => v >= 0.8).length },
    ];
    const lowExamples = [];
    const excerpt = i => JSON.stringify(cur[i]).slice(0, 160);
    let kept = cur;
    if (minScore > 0) {
      kept = cur.filter((s, i) => {
        if (scores[i].score >= minScore) return true;
        if (lowExamples.length < 5) lowExamples.push({ index: i, score: scores[i].score, flags: scores[i].flags, excerpt: excerpt(i) });
        return false;
      });
    } else {
      values.map((v, i) => ({ score: v, index: i, flags: scores[i].flags }))
        .sort((a, b) => a.score - b.score).slice(0, 3)
        .forEach(v => lowExamples.push({ index: v.index, score: v.score, flags: v.flags, excerpt: excerpt(v.index) }));
    }
    report.quality = {
      scored: cur.length,
      average: +avg.toFixed(3),
      median: sorted.length ? +sorted[Math.floor(sorted.length / 2)].toFixed(3) : 0,
      min: sorted.length ? sorted[0] : 0,
      max: sorted.length ? sorted[sorted.length - 1] : 0,
      threshold: minScore,
      distribution,
      lowExamples,
    };
    drop('quality_score', kept,
      '启发式质量评分（均值 ' + avg.toFixed(2) + '，阈值 ' + minScore +
      (minScore > 0 ? '，低于阈值拦截' : '，仅报告不过滤') + '）');
  }

  // ⑦ 评测集泄漏隔离（2026-11-11 新增）：holdoutQueries = ['q1','q2',...] 时，
  // 合成样本与评测集 query 精确/规范化匹配的直接剔除——训练/评测污染是对照实验最致命的坑
  if (Array.isArray(opts.holdoutQueries) && opts.holdoutQueries.length) {
    const norm = t => String(t || '').toLowerCase().replace(/\s+/g, '');
    const hold = new Set(opts.holdoutQueries.map(norm));
    const kept = cur.filter(s => {
      const texts = [s.instruction, s.query, s.state,
        Array.isArray(s.state) ? s.state.map(m => m && m.content).join('') : ''];
      return !texts.some(t => t && hold.has(norm(t)));
    });
    drop('holdout_leakage', kept, `与评测集（${opts.holdoutQueries.length} 条）规范化后重复`);
  }

  report.passed = cur.length;
  report.passRate = samples.length ? +(cur.length / samples.length).toFixed(3) : 0;
  report.samples = cur;
  return report;
}

/**
 * ⑦ LLM 裁判（异步，可选）。
 * judge: async (batch, startIndex) -> [{score:0~1, reason}]，长度须与 batch 一致。
 * 设计取舍：
 *  - 只裁判前 judgeLimit 条（默认 100），其余标为 unjudged 并保留，避免 20 万样本产生天价调用；
 *  - 裁判调用失败/解析失败的样本保留（fail-open），但失败条数写进报告，不静默吞掉；
 *  - 阈值默认 0.5（对应 10 分制的 5 分），只有显式启用且阈值 > 0 才拦截。
 */
async function qualityFunnelAsync(samples, opts) {
  opts = opts || {};
  const report = qualityFunnel(samples, opts);
  const entered = report.samples.length;
  if (typeof opts.judge !== 'function') {
    report.stages.push({
      stage: 'llm_judge', input: entered, removed: 0,
      reason: '未启用（需显式开启 LLM 裁判并指定生成引擎）', enabled: false, skipped: true,
    });
    return report;
  }
  const limit = Math.max(1, Math.min(Number(opts.judgeLimit) || 100, 2000));
  const batchSize = Math.max(1, Math.min(Number(opts.judgeBatchSize) || 5, 20));
  const rawThreshold = Number(opts.judgeThreshold);
  const threshold = Number.isFinite(rawThreshold) ? Math.max(0, Math.min(1, rawThreshold)) : 0.5;
  const verdicts = new Array(Math.min(entered, limit)).fill(null);
  let failed = 0, calls = 0;
  for (let i = 0; i < verdicts.length; i += batchSize) {
    const batch = report.samples.slice(i, Math.min(i + batchSize, verdicts.length));
    calls++;
    try {
      const out = await opts.judge(batch, i);
      if (!Array.isArray(out)) { failed += batch.length; continue; }
      for (let k = 0; k < batch.length; k++) {
        const v = out[k];
        if (!v || !Number.isFinite(Number(v.score))) { failed++; continue; }
        verdicts[i + k] = {
          score: Math.max(0, Math.min(1, Number(v.score))),
          reason: String(v.reason || '').slice(0, 120),
        };
      }
    } catch (e) { failed += batch.length; }
  }
  const lowExamples = [];
  const kept = report.samples.filter((s, idx) => {
    if (idx >= verdicts.length) return true;  // 未裁判样本保留
    const v = verdicts[idx];
    if (!v) return true;                       // 裁判失败保留（fail-open）
    if (threshold > 0 && v.score < threshold) {
      if (lowExamples.length < 5) lowExamples.push({ index: idx, score: v.score, reason: v.reason, excerpt: JSON.stringify(s).slice(0, 160) });
      return false;
    }
    return true;
  });
  const judgedOk = verdicts.filter(Boolean).length;
  report.stages.push({
    stage: 'llm_judge', input: entered, removed: entered - kept.length,
    reason: 'LLM 裁判（判定 ' + judgedOk + '/' + verdicts.length + ' 条，阈值 ' + threshold + '，未裁判 ' + Math.max(0, entered - verdicts.length) + ' 条保留）',
    enabled: true, threshold, judged: verdicts.length, judgedOk,
    unjudged: Math.max(0, entered - verdicts.length), failed, calls, lowExamples,
  });
  report.llmJudge = {
    enabled: true, threshold, limit, batchSize, calls,
    judged: verdicts.length, judgedOk, failed,
    unjudged: Math.max(0, entered - verdicts.length),
    average: judgedOk ? +(verdicts.filter(Boolean).reduce((a, v) => a + v.score, 0) / judgedOk).toFixed(3) : null,
    lowExamples,
  };
  report.passed = kept.length;
  report.passRate = report.total ? +(kept.length / report.total).toFixed(3) : 0;
  report.samples = kept;
  return report;
}

// ── 目标格式转换（输入契约：已归一化的内部格式样本）──────────────
// 归一化必须且只能在调用方做一次。历史缺陷：质检后再次 normalize 会把
// sharegpt / openai_messages / dpo_standard / qa_pairs 的字段二次映射成空串，
// 质检报告显示"全部通过"却把空样本写进训练集。故此处默认不再归一化，
// 且转换后强制校验必填字段，发现空值直接抛错中止注册（fail-closed）。
function convertTo(samples, targetKind, sourceLayout) {
  const norm = sourceLayout ? normalize(samples, sourceLayout) : samples;
  let out;
  switch (targetKind) {
    case 'sft':
      out = norm.map(s => ({ instruction: s.instruction, input: s.input || '', output: s.output }));
      break;
    case 'preference':
      out = norm.map(s => ({ instruction: s.instruction, input: s.input || '', chosen: s.chosen, rejected: s.rejected }));
      break;
    case 'decision':
      out = norm;  // 需要已是决策格式
      break;
    case 'pretrain':
      out = norm.map(s => ({ text: s.text || [s.instruction, s.output].filter(Boolean).join('\n\n') }));
      break;
    default:
      throw new Error('未知目标类型: ' + targetKind);
  }
  const REQUIRED = {
    sft: ['instruction', 'output'],
    preference: ['instruction', 'chosen', 'rejected'],
    pretrain: ['text']
  };
  const required = REQUIRED[targetKind] || [];
  if (required.length) {
    const bad = out.filter(s => required.some(k => s[k] == null || String(s[k]).trim() === ''));
    if (bad.length) {
      throw new Error('转换后有 ' + bad.length + '/' + out.length + ' 条样本缺少 ' +
        required.join(' / ') + '（疑似重复归一化或字段映射错误）；已中止注册，避免空数据进入训练集');
    }
  }
  return out;
}

// ── 文件解析（json/jsonl/csv）─────────────────────────────
function parseFile(content, fileName) {
  const ext = (fileName || '').split('.').pop().toLowerCase();
  if (ext === 'jsonl') {
    return content.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
  }
  if (ext === 'csv') {
    // 简易 CSV（含引号处理）
    const rows = [];
    let row = [], cell = '', inQ = false;
    for (let i = 0; i < content.length; i++) {
      const c = content[i];
      if (inQ) {
        if (c === '"' && content[i + 1] === '"') { cell += '"'; i++; }
        else if (c === '"') inQ = false;
        else cell += c;
      } else if (c === '"') inQ = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && content[i + 1] === '\n') i++;
        row.push(cell); cell = '';
        if (row.some(x => x !== '')) rows.push(row);
        row = [];
      } else cell += c;
    }
    if (cell !== '' || row.length) { row.push(cell); if (row.some(x => x !== '')) rows.push(row); }
    if (rows.length < 2) throw new Error('CSV 至少需要表头 + 1 行数据');
    const header = rows[0];
    return rows.slice(1).map(r => {
      const o = {};
      header.forEach((h, i) => o[h.trim()] = (r[i] || '').trim());
      return o;
    });
  }
  // json
  const parsed = JSON.parse(content);
  if (Array.isArray(parsed)) return parsed;
  if (parsed && typeof parsed === 'object') {
    // 单对象或 {data: []} 包装
    if (Array.isArray(parsed.data)) return parsed.data;
    if (Array.isArray(parsed.samples)) return parsed.samples;
    if (Array.isArray(parsed.rows)) return parsed.rows;
    return [parsed];
  }
  throw new Error('无法解析：JSON 顶层既不是数组也不是对象');
}

module.exports = {
  sniffLayout, normalize, qualityFunnel, qualityFunnelAsync, scoreSample, scoreSamples,
  convertTo, parseFile, LAYOUTS, DATA_TOOLS_DIR,
};
