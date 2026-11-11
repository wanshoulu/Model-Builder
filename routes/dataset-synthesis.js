// ═══ 数据集自动合成引擎（模块B）═══
// 三范式（按场景适配业界五范式）：
//   1. evol       — Evol-Instruct 种子进化（用户提供种子 → 深度/广度两方向扩展）
//   2. persona    — Persona 角色分化（画像 × 领域 → 多样性保障）
//   3. decision   — 决策合成（专属：从业务意图体系生成 state+questions+label）
// 生成引擎抽象：SynthesisEngine 接口（本机推理 / 远端 API / 免密钥云服务三种后端）
// 铁律：合成产物强制过质检漏斗；合成占比 > 75% 警告（Nature 2024 模型坍缩实证）
const { qualityFunnelAsync } = require('./dataset-tools');

/**
 * 上游错误脱敏：云 API 报错原文可能带密钥、鉴权头或内网地址，不能直接回显到前端。
 * 只保留错误类型与简短上下文，供排障用。
 */
function sanitizeErr(msg, limit) {
  let s = String(msg == null ? '未知错误' : msg);
  s = s.replace(/sk-[A-Za-z0-9_-]{6,}/g, 'sk-***')
    .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, 'Bearer ***')
    .replace(/(key|token|secret|password|api_key)=[^\s&"'<>]+/gi, '$1=***')
    .replace(/https?:\/\/[^\s"'<>]+/g, '〔URL 已隐藏〕')
    .replace(/[\r\n\t]+/g, ' ');
  return s.trim().slice(0, limit || 120) || '未知错误';
}

// ── 生成后端抽象 ─────────────────────────────────────────
// 每个后端实现 async generate(messages, opts) -> string
const backends = {
  // 云 API（复用平台 config/api-keys 的多模型密钥体系：QWEN_KEY/DEEPSEEK_KEY/GLM_KEY/KIMI_KEY/ARK_KEY）
  cloud: {
    name: '云 API（Qwen/DeepSeek/GLM 等，复用平台密钥）',
    available: () => {
      try {
        const { API_ROUTES } = require('../config/api-keys');
        return Object.values(API_ROUTES).some(r => r && r.getKey && r.getKey() && r.host !== 'http://localhost:3000');
      } catch (e) { return false; }
    },
    generate: async (messages, opts) => {
      const { API_ROUTES } = require('../config/api-keys');
      // 模型选择：opts.model 为空则按平台优先级自动挑（qwen3.7-flash → deepseek → glm → kimi）
      const PRIORITY = ['qwen3.7-flash', 'deepseek-chat', 'glm-5.2', 'kimi-latest'];
      let route = null;
      if (opts.model && API_ROUTES[opts.model] && API_ROUTES[opts.model].getKey()) route = { r: API_ROUTES[opts.model], model: opts.model };
      for (const name of opts.model ? [] : PRIORITY) {
        if (API_ROUTES[name] && API_ROUTES[name].getKey()) { route = { r: API_ROUTES[name], model: API_ROUTES[name].model || name }; break; }
      }
      if (!route) throw new Error('云 API 密钥未配置——请设置环境变量（QWEN_KEY / DEEPSEEK_KEY / GLM_KEY 任一）后重启服务，或在合成时改用 remote 后端（自建 API）');
      const https = require('node:https');
      const body = JSON.stringify({ model: route.model, messages, temperature: opts.temperature ?? 0.8, max_tokens: opts.maxTokens || 2000 });
      return new Promise((resolve, reject) => {
        const req = https.request({
          hostname: route.r.host, port: 443,
          path: (route.r.path || '/compatible-mode/v1/chat/completions'),
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + route.r.getKey() },
          timeout: opts.timeoutMs || 300000,
        }, resp => {
          let d = ''; resp.on('data', c => d += c);
          resp.on('end', () => {
            try {
              const j = JSON.parse(d);
              if (j.error) reject(new Error(j.error.message || j.error.code || 'API error'));
              else resolve((j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || '');
            } catch (e) { reject(new Error('响应解析失败: ' + d.slice(0, 150))); }
          });
        });
        req.on('error', reject);
        req.on('timeout', () => { req.destroy(new Error('云 API 超时')); });
        // 用户点击「停止」时立刻掐断在途 HTTPS 请求，而不是等下一个循环才检查
        const abortSignal = opts._abortSignal;
        let onAbort = null;
        if (abortSignal) {
          if (abortSignal.aborted) { req.destroy(new Error('合成已中断')); return; }
          onAbort = function () { req.destroy(new Error('合成已中断')); };
          abortSignal.addEventListener('abort', onAbort, { once: true });
          req.on('close', function () { abortSignal.removeEventListener('abort', onAbort); });
        }
        req.write(body); req.end();
      });
    },
  },
  // 远端服务器上的模型（通过 OpenAI 兼容 API——如 llamafactory-cli api 或 vllm serve）
  remote: {
    name: '远端服务器模型（OpenAI 兼容 API）',
    available: () => true,
    generate: async (messages, opts) => {
      const baseUrl = opts.baseUrl || 'http://localhost:8000/v1';
      const model = opts.model || 'default';
      // LLaMA-Factory API 只支持 user/assistant 交替（无 system）——把 system 合并进首条 user
      let merged = messages.slice();
      if (merged.length && merged[0].role === 'system') {
        const sys = merged.shift().content;
        if (merged.length && merged[0].role === 'user') merged[0] = { role: 'user', content: sys + '\n\n' + merged[0].content };
        else merged.unshift({ role: 'user', content: sys });
      }
      const r = await fetch(baseUrl + '/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(opts.apiKey ? { Authorization: 'Bearer ' + opts.apiKey } : {}) },
        body: JSON.stringify({ model, messages: merged, temperature: opts.temperature ?? 0.8, max_tokens: opts.maxTokens || 2000 }),
        signal: opts._abortSignal
          ? AbortSignal.any([AbortSignal.timeout(opts.timeoutMs || 300000), opts._abortSignal])
          : AbortSignal.timeout(opts.timeoutMs || 300000),
      });
      if (!r.ok) throw new Error('远端生成失败 HTTP ' + r.status + ': ' + (await r.text()).slice(0, 200));
      const d = await r.json();
      return (d.choices && d.choices[0] && d.choices[0].message && d.choices[0].message.content) || '';
    },
  },
};

function listEngines() {
  return Object.entries(backends).map(([id, b]) => ({ id, name: b.name, available: b.available() }));
}

// ── 提示词模板 ────────────────────────────────────────────
const PROMPTS = {
  evol: {
    depth: (seed, domain) => [
      { role: 'system', content: '你是训练数据合成专家。基于种子样本生成一个更难、更具体的变体：增加约束条件、加入边界情况或需要多步推理。严格输出 JSON：{"instruction": "...", "output": "..."}，不要输出其他内容。' },
      { role: 'user', content: '领域：' + domain + '\n种子样本：' + JSON.stringify(seed) + '\n\n生成一个深度进化变体（更难/更具体/多约束）。' },
    ],
    breadth: (seed, domain) => [
      { role: 'system', content: '你是训练数据合成专家。基于种子样本生成一个同领域但不同主题/角度的新样本（广度进化）。严格输出 JSON：{"instruction": "...", "output": "..."}，不要输出其他内容。' },
      { role: 'user', content: '领域：' + domain + '\n种子样本：' + JSON.stringify(seed) + '\n\n生成一个广度进化变体（新主题/新角度）。' },
    ],
  },
  persona: (persona, domain, formatHint, lang) => [
    { role: 'system', content: lang === 'en'
      ? 'Output ONE JSON object directly, first character must be {, no explanation, no reasoning. Format: {"instruction": "user question", "output": "professional answer"} — all content in English.'
      : '直接输出一个JSON对象，第一个字符必须是 {，不要解释、不要思考过程。格式：{"instruction": "用户问题", "output": "专业回答"}' },
    { role: 'user', content: (lang === 'en' ? 'Domain: ' : '领域：') + domain + '\n' + (lang === 'en' ? 'Persona: ' : '人物画像：') + persona + '\n' + (formatHint || '') + '\n' + (lang === 'en'
      ? 'Output the JSON of one question this persona would ask in this domain, with a professional answer.'
      : '直接输出该画像会问的一条问题及标准回答的JSON。') },
  ],
  preference: (persona, domain, lang) => [
    { role: 'system', content: lang === 'en'
      ? 'Output ONE JSON directly, first char must be {. Format: {"instruction": "question", "chosen": "a high-quality, accurate, well-structured answer", "rejected": "a plausible-looking but flawed answer (subtly wrong / incomplete / misleading)"} — the rejected must look tempting but be genuinely worse. English only.'
      : '直接输出一个JSON，第一个字符必须是 {。格式：{"instruction": "问题", "chosen": "高质量、准确、结构良好的回答", "rejected": "看似合理但有实质缺陷的回答（ subtly 错误/不完整/有误导）"}——rejected 必须有迷惑性但确实更差。' },
    { role: 'user', content: (lang === 'en' ? 'Domain: ' : '领域：') + domain + '\n' + (lang === 'en' ? 'Persona: ' : '人物画像：') + persona + '\n' + (lang === 'en'
      ? 'Output the JSON with one question in this domain and the chosen/rejected answer pair.'
      : '输出该领域一条问题及 chosen/rejected 回答对的JSON。') },
  ],
  doc: (chunk, domain, lang, idx) => [
    { role: 'system', content: lang === 'en'
      ? 'You are a training-data generation expert. Based ONLY on the provided document excerpt, generate ONE question-answer pair. The question should be answerable from the excerpt; the answer should be accurate, complete, and well-grounded in the excerpt (no invented facts). Output ONE JSON directly, first char must be {: {"instruction": "question", "output": "answer"} — English only.'
      : '你是训练数据生成专家。仅基于给定的文档片段生成一个问答对。问题必须能从片段中找到答案；回答要准确、完整、严格依据片段内容（不得编造）。直接输出一个JSON，第一个字符必须是 {：{"instruction": "问题", "output": "回答"}' },
    { role: 'user', content: (lang === 'en' ? 'Domain context: ' : '领域背景：') + domain + '\n' + (lang === 'en' ? 'Document excerpt (#' : '文档片段（#') + (idx + 1) + (lang === 'en' ? '):\n' : '）：\n') + String(chunk).slice(0, 3000) },
  ],
  decision: (spec, idx) => [
    { role: 'system', content: '你是决策模型训练数据合成专家。生成一条类型化决策样本（类 System One/Jev 格式）。严格输出 JSON（不要输出其他内容）：\n{"state": [{"role": "...", "content": "..."}], "questions": {"<问题名>": {"type": "choice|noul|score", "instructions": "...", "criteria": {...}, "label": <正确答案>}}}\nchoice 的 criteria 是选项名→描述（可为 null）的对象，label 是正确选项名；noul 的 label 是 true/false；score 的 criteria 是有序档位数组，label 是档位号。\n' + (spec.questionSchema
      ? '必须恰好生成以下问题（名字与类型严格一致）：' + JSON.stringify(spec.questionSchema)
      : '问题可自拟 1-3 个，须覆盖给定的选项体系。') + '\nstate 是该业务场景下的真实上下文（对话/工单/记录/文书等，形式贴合场景）。' +
      (spec.seeds && spec.seeds.length
        ? '\n【重要】你会拿到一条真实种子样本（query 与其意图标签）。必须以该种子为基准生成：state 用种子的 query（可轻微润色但语义不得改变），label 按种子标签与问题体系推导——禁止编造与种子无关的内容。'
        : '') },
    { role: 'user', content: '业务场景：' + spec.domain + '\n选项/意图体系：' + JSON.stringify(spec.options || []) + '\n' +
      (spec.seeds && spec.seeds.length
        ? '种子样本（第 ' + (idx % spec.seeds.length + 1) + ' 条）：' + JSON.stringify(spec.seeds[idx % spec.seeds.length]) + '\n请基于这条种子生成决策样本。'
        : '第 ' + (idx + 1) + ' 条：生成一条多样化的决策样本（改变 state 内容、措辞与难度；不要与常规示例雷同）。') },
  ],
  // 按标签扩写（label-variant 模式）：few-shot 种子 → 同意图变体
  labelVariant: (label, examples, nVariants, styles) => [
    { role: 'system', content: '你是训练数据合成专家。我会给你一个意图标签和几条真实用户问句样例。' +
      '请生成 ' + nVariants + ' 条该意图的【新用户问句变体】，要求：\n' +
      '- 覆盖不同表达风格：' + (styles || '口语化/正式/简短命令/带错别字或拼音缩写/带上下文的长句') + '\n' +
      '- 严格保持与样例相同的意图（不得漂移到别的业务）\n' +
      '- 长度 3~40 字，不得复读样例\n' +
      '严格输出 JSON 数组（不要其他内容）：\n[{"query": "...", "style": "..."}]' },
    { role: 'user', content: '意图标签：' + label + '\n样例问句：\n' + examples.slice(0, 5).map(function (q) { return '- ' + q; }).join('\n') },
  ],
  // 质检漏斗第 ⑦ 段：LLM 裁判。一次评一批，输出数组，i 与批内序号对应。
  judge: (items, targetKind) => [
    { role: 'system', content: '你是训练数据质检裁判。逐条判断样本是否适合作为「' + String(targetKind || 'sft') + '」训练数据，重点看：事实是否自洽、回答是否真正回应了问题、是否含占位符/复读/无意义内容、偏好对的 chosen 是否明显优于 rejected、决策样本的 label 是否与 state 一致。\n给每条打 0-10 分（10=可直接用于训练，0=必须丢弃），并给一句中文短理由。严格输出 JSON 数组，不要解释： [{"i":0,"score":8,"reason":"..."},{"i":1,"score":3,"reason":"..."}]。数组长度必须与样本数一致，i 用样本前的 # 序号。' },
    { role: 'user', content: '待评样本：\n' + items.map(x => '#' + x.i + ' ' + x.sample).join('\n') },
  ],
};

/**
 * 生成 LLM 裁判函数，注入 qualityFunnelAsync 的 opts.judge。
 * 与合成共用同一套生成后端，但模型/地址/密钥可单独指定（裁判通常用更小更便宜的模型）。
 * 温度固定 0，输出长度受限：裁判只需要分数与短理由。
 */
function createLlmJudge(p) {
  p = p || {};
  if (!p.llmJudge) return null;
  const backendId = p.judgeBackend || p.backend || 'cloud';
  const backend = backends[backendId];
  if (!backend) throw new Error('未知裁判后端: ' + backendId);
  const judgeOpts = {
    baseUrl: p.judgeBaseUrl || p.baseUrl,
    model: p.judgeModel || (p.judgeBackend === p.backend || !p.judgeBackend ? p.model : undefined),
    apiKey: p.judgeApiKey || p.apiKey,
    temperature: 0,
    maxTokens: Math.min(Number(p.judgeMaxTokens) || 1200, 4000),
    timeoutMs: Number(p.judgeTimeoutMs) || 120000,
    _abortSignal: p._abortSignal,   // 任务被停止时同步掐断裁判在途请求
  };
  return async function judge(batch) {
    const items = batch.map((s, i) => ({ i, sample: JSON.stringify(s).slice(0, 1500) }));
    const out = await backend.generate(PROMPTS.judge(items, p.targetKind), judgeOpts);
    const arr = extractJSONArray(out);
    if (!arr) throw new Error('裁判响应无法解析为 JSON 数组');
    const byIndex = new Map();
    arr.forEach(v => { if (v && Number.isFinite(Number(v.i))) byIndex.set(Number(v.i), v); });
    return batch.map((s, i) => {
      const v = byIndex.get(i);
      if (!v || !Number.isFinite(Number(v.score))) return null;
      const raw = Number(v.score);
      const score = raw > 1 ? raw / 10 : raw;  // 兼容 0-10 与 0-1 两种打分
      return { score, reason: String(v.reason || '').slice(0, 120) };
    });
  };
}

// ── 小模型 JSON 容错提取（0.6B 级模型常输出带控制字符/尾逗号的伪 JSON）──
function parseLooseJson(s) {
  s = s.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
  s = s.replace(/,\s*([}\]])/g, '$1');
  try { return JSON.parse(s); } catch (e) { /* 继续 */ }
  // 二次修复：把裸换行替换成 \n 转义（字符串值内）
  try {
    const fixed = s.replace(/"([^"\\]*?)\n([^"\\]*?)"/g, '"$1\\n$2"');
    return JSON.parse(fixed);
  } catch (e) { return null; }
}
function extractJSON(text) {
  if (!text) return null;
  // 去掉 think 块
  const noThink = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  const raw = noThink.includes('{') ? noThink : text;
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) return null;
  const v = parseLooseJson(m[0]);
  return v && typeof v === 'object' && !Array.isArray(v) ? v : null;
}
// 裁判响应是 JSON 数组（不是对象），单独提取
function extractJSONArray(text) {
  if (!text) return null;
  const raw = String(text).replace(/<think>[\s\S]*?<\/think>/g, '');
  const m = raw.match(/\[[\s\S]*\]/);
  if (!m) return null;
  const v = parseLooseJson(m[0]);
  return Array.isArray(v) ? v : null;
}

// ── 文档分块（Easy Dataset 对标的轻量版：段落聚合，无需 embedding）──
function chunkDocument(text, targetChars) {
  targetChars = targetChars || 1500;
  const paras = String(text || '').split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  if (!paras.length) return [];
  const chunks = [];
  let cur = '';
  for (const p of paras) {
    // 单段超长：按句切
    if (p.length > targetChars * 2) {
      if (cur) { chunks.push(cur); cur = ''; }
      const sentences = p.split(/(?<=[。！？.!?])\s*/);
      let sc = '';
      for (const s of sentences) {
        if ((sc + s).length > targetChars && sc) { chunks.push(sc); sc = s; }
        else sc += s;
      }
      if (sc) chunks.push(sc);
      continue;
    }
    if ((cur + '\n\n' + p).length > targetChars && cur) { chunks.push(cur); cur = p; }
    else cur = cur ? cur + '\n\n' + p : p;
  }
  if (cur) chunks.push(cur);
  return chunks;
}

// ── 合成主流程 ─────────────────────────────────────────────
async function synthesize(p) {
  const mode = p.mode || 'evol';
  const count = Math.min(Math.max(p.count || 20, 1), 500);  // 单次 1-500 条
  const domain = p.domain || '';
  const targetKind = p.targetKind || 'sft';
  const backendId = p.backend || 'remote';
  const backend = backends[backendId];
  if (!backend) throw new Error('未知生成后端: ' + backendId + '（可选: ' + Object.keys(backends).join(', ') + '）');
  if (backendId === 'cloud') {
    // 云 API 后端：检查密钥（模型名可选映射平台 API_ROUTES，如 qwen3.7-flash）
    const { API_ROUTES } = require('../config/api-keys');
    const hasKey = Object.values(API_ROUTES).some(r => r && r.getKey && r.getKey() && r.host !== 'http://localhost:3000');
    if (!hasKey) throw new Error('云 API 密钥未配置：请设置环境变量 QWEN_KEY（或 DEEPSEEK_KEY / GLM_KEY / KIMI_KEY）后重启服务，或改用 remote 后端');
  }
  if (!domain) throw new Error('domain（领域描述）必填');

  const opts = { baseUrl: p.baseUrl, model: p.model, apiKey: p.apiKey, temperature: p.temperature, maxTokens: p.maxTokens, _abortSignal: p._abortSignal };
  const raw = [];
  const errors = [];

  if (mode === 'doc') {
    // 文档驱动合成（Easy Dataset 对标）：文档文本 → 分块 → 逐块 QA 生成（严格依据原文，不编造）
    const docText = p.docText || '';
    if (!docText.trim()) throw new Error('doc 模式需要 docText（文档全文，可通过导入路径/上传读取）');
    const targetChars = p.chunkChars || 1500;
    const chunks = chunkDocument(docText, targetChars);
    if (!chunks.length) throw new Error('文档分块后为空——文本过短或为空');
    // 每块生成 nPerChunk 条，凑够 count（块不足则循环块）
    const nPerChunk = p.nPerChunk || 1;
    const lang = p.lang || '';
    let generated = 0;
    outer:
    for (let round = 0; round < nPerChunk; round++) {
      for (let ci = 0; ci < chunks.length; ci++) {
        if (generated >= count) break outer;
        if (p._progress && p._progress(generated, count, raw)) throw new Error('已中断');
        try {
          const out = await backend.generate(PROMPTS.doc(chunks[ci], domain, lang, ci), opts);
          const j = extractJSON(out);
          if (j && j.instruction && j.output) { j._chunkIndex = ci; raw.push(j); generated++; }
        } catch (e) { errors.push('块' + (ci + 1) + '第' + (round + 1) + '条: ' + sanitizeErr(e.message)); }
      }
    }
  } else if (mode === 'evol') {
    const seeds = p.seeds || [];
    if (!seeds.length) throw new Error('evol 模式需要 seeds（5-20 条种子样本）');
    for (let i = 0; i < count; i++) {
      if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
      const seed = seeds[i % seeds.length];
      const direction = i % 2 === 0 ? 'depth' : 'breadth';  // 深度/广度交替
      try {
        const out = await backend.generate(PROMPTS.evol[direction](seed, domain), opts);
        const j = extractJSON(out);
        if (j) raw.push(j);
      } catch (e) { errors.push('第' + (i + 1) + '条: ' + sanitizeErr(e.message)); }
    }
  } else if (mode === 'persona') {
    const personas = p.personas && p.personas.length ? p.personas : defaultPersonas(domain);
    for (let i = 0; i < count; i++) {
      if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
      const persona = personas[i % personas.length];
      try {
        const out = await backend.generate(PROMPTS.persona(persona, domain, p.formatHint, p.lang), opts);
        const j = extractJSON(out);
        if (j) raw.push(j);
      } catch (e) { errors.push('第' + (i + 1) + '条: ' + sanitizeErr(e.message)); }
    }
  } else if (mode === 'decision') {
    if (targetKind !== 'decision') throw new Error('decision 合成模式需要 targetKind=decision');
    if (!p.options && !(Array.isArray(p.seeds) && p.seeds.length)) throw new Error('decision 模式需要 options（意图/选项体系）或 seeds（种子驱动）');
    // 种子注入（2026-11-11 补能力）：seeds = [{query, label}] 时逐条基于真实种子生成决策样本，
    // 而非凭空编造——建行意图合成实测：few-shot 种子让分布贴合真实业务、长尾意图可覆盖
    const spec = { domain, options: p.options, questionSchema: p.questionSchema || null,
      seeds: Array.isArray(p.seeds) && p.seeds.length ? p.seeds : null };
    for (let i = 0; i < count; i++) {
      if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
      try {
        const out = await backend.generate(PROMPTS.decision(spec, i));
        const j = extractJSON(out);
        if (j) raw.push(j);
      } catch (e) { errors.push('第' + (i + 1) + '条: ' + sanitizeErr(e.message)); }
    }
  } else if (mode === 'label-variant') {
    // 按标签扩写（2026-11-11 新增模式）：给一批 (query,label) 种子，按标签分组、
    // few-shot 扩写出同意图的新变体。意图识别/分类任务的 SFT 数据主力合成方式。
    // 参数：labelField（默认 'output'）、variantsPerBatch（默认 5）、styles（风格约束）
    if (targetKind === 'decision') throw new Error('label-variant 模式不支持 decision 目标（用 rule-map）');
    const seeds = Array.isArray(p.seeds) && p.seeds.length ? p.seeds : null;
    if (!seeds) throw new Error('label-variant 模式需要 seeds（[{instruction/query, output/label}]）');
    const labelField = p.labelField || 'output';
    const textField = seeds[0] && seeds[0].instruction != null ? 'instruction' : 'query';
    const byLabel = new Map();
    seeds.forEach(function (s) {
      const label = String(s[labelField] != null ? s[labelField] : s.label || '');
      const text = String(s[textField] != null ? s[textField] : s.query || '').trim();
      if (label && text) {
        if (!byLabel.has(label)) byLabel.set(label, []);
        byLabel.get(label).push(text);
      }
    });
    const vpBatch = Math.max(1, Math.min(parseInt(p.variantsPerBatch, 10) || 5, 10));
    const labels = Array.from(byLabel.keys());
    for (let li = 0; li < labels.length; li++) {
      const label = labels[li];
      const examples = byLabel.get(label);
      const nBatches = p.batchesPerLabel != null
        ? Math.max(0, Math.min(parseInt(p.batchesPerLabel, 10) || 1, 5))
        : (examples.length >= 20 ? 2 : 1);
      for (let b = 0; b < nBatches; b++) {
        if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
        try {
          const out = await backend.generate(PROMPTS.labelVariant(label, examples, vpBatch, p.styles));
          const arr = extractJSONArray(out);
          (arr || []).forEach(function (x) {
            if (x && x.query && String(x.query).length >= 2) {
              const rec = { instruction: String(x.query).trim(), input: '', output: label };
              if (x.style) rec.synthStyle = x.style;
              raw.push(rec);
            }
          });
        } catch (e) { errors.push('标签[' + label.slice(0, 12) + ']批' + b + ': ' + sanitizeErr(e.message)); }
      }
      if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
    }
  } else if (mode === 'rule-map') {
    // 确定性规则映射（2026-11-11 新增模式）：不调 LLM，把种子按规则模板转成 decision 样本。
    // 参数：rules = { questionSchema, map(sample) → labels }（平台内由调用方提供映射函数的
    // JSON 化版本：domainKeywords 域路由 + queryVerbs 操作词表，均可配置）
    if (targetKind !== 'decision') throw new Error('rule-map 模式需要 targetKind=decision');
    const seeds = Array.isArray(p.seeds) && p.seeds.length ? p.seeds : null;
    if (!seeds) throw new Error('rule-map 模式需要 seeds');
    const rs = p.rules || {};
    if (!rs.questionSchema) throw new Error('rule-map 需要 rules.questionSchema（questions 模板）');
    const kwMap = rs.domainKeywords || [];
    seeds.forEach(function (s) {
      const text = String(s.query != null ? s.query : s.instruction || '').trim();
      const label = String(s.label != null ? s.label : s.output || '').trim();
      if (!text || !label) return;
      // 域路由：关键词表按序匹配意图标签
      let domainVal = rs.defaultDomain || '综合';
      for (const kw of kwMap) {
        if (kw && kw.keywords && kw.keywords.some(function (k) { return label.indexOf(k) >= 0; })) {
          domainVal = kw.domain; break;
        }
      }
      const q = {};
      rs.questionSchema.forEach(function (qs) {
        q[qs.name] = { type: qs.type, instructions: qs.instructions || '',
          criteria: qs.criteria || null, label: qs.derive === 'domain' ? domainVal : (qs.label != null ? qs.label : null) };
      });
      raw.push({ state: text, questions: q });
    });
  } else if (mode === 'preference') {
    // 偏好对合成（DPO/KTO 数据）：对同一问题生成 chosen/rejected 两个回答
    if (targetKind === 'decision') throw new Error('preference 模式不支持 decision 目标');
    for (let i = 0; i < count; i++) {
      if (p._progress && p._progress(raw.length, count, raw)) throw new Error('已中断');
      const persona = (p.personas && p.personas.length ? p.personas : defaultPersonas(domain))[i % (p.personas && p.personas.length ? p.personas.length : 10)];
      try {
        const out = await backend.generate(PROMPTS.preference(persona, domain, p.lang), opts);
        const j = extractJSON(out);
        if (j && j.instruction && j.chosen && j.rejected) raw.push(j);
      } catch (e) { errors.push('第' + (i + 1) + '条: ' + sanitizeErr(e.message)); }
    }
  } else {
    throw new Error('未知合成模式: ' + mode + '（可选: doc / evol / persona / decision / label-variant / rule-map / preference）');
  }

  // 强制质检漏斗（业界共识：合成数据不过滤等于投毒）
  // 完整 7 段：确定性 6 段 + 可选 LLM 裁判（异步）。裁判与合成共用生成后端。
  const funnel = await qualityFunnelAsync(raw, {
    targetKind,
    targetLang: p.targetLang,
    minLen: p.minLen, maxLen: p.maxLen, minScore: p.minScore,
    semanticThreshold: p.semanticThreshold,
    holdoutQueries: p.holdoutQueries || null,   // 评测集隔离：合成样本与留出集重复直接剔除
    judge: createLlmJudge(Object.assign({}, p, { targetKind, _abortSignal: p._abortSignal })),
    judgeLimit: p.judgeLimit, judgeBatchSize: p.judgeBatchSize, judgeThreshold: p.judgeThreshold,
  });

  // 25% 真实数据铁律检查（返回给前端警告，不阻断）
  let mixWarning = null;
  const realCount = p.realCount || 0;
  if (funnel.passed > 0 && funnel.passed / (funnel.passed + realCount) > 0.75 && realCount === 0) {
    mixWarning = '本次全部为合成数据。Nature 2024 实证：纯合成数据训练会导致模型坍缩，建议混入 ≥25% 真实数据（当前合成 ' + funnel.passed + ' 条，建议再准备至少 ' + Math.ceil(funnel.passed / 3) + ' 条真实数据）。';
  }

  return {
    mode, rawCount: raw.length,
    samples: funnel.samples,
    funnel: {
      total: funnel.total, passed: funnel.passed, passRate: funnel.passRate,
      stages: funnel.stages, quality: funnel.quality || null, llmJudge: funnel.llmJudge || null,
    },
    mixWarning,
    errors: errors.slice(0, 10),
    engine: backend.name,
    backend: backendId,
    model: p.model || (backendId === 'cloud' ? '自动' : 'default'),
    domain,
    targetKind,
  };
}

function defaultPersonas(domain) {
  // 领域无关的用户画像维度（跨领域通用：任何"领域 × 画像"组合都能产生合理提问者）
  // 对齐 Persona-based generation 的多样性原则——画像控制的是提问风格而非领域内容
  const dimensions = [
    ['急躁紧迫型', '赶时间，说话直接，没耐心听长解释'],
    ['谨慎仔细型', '对操作不熟悉，需要一步步的耐心解释，会反复确认'],
    ['熟练自助型', '习惯自己摸索，问题偏功能细节和进阶用法'],
    ['专业挑剔型', '懂行（同行/内部人员），问题刁钻专业'],
    ['小白萌新型', '连基础概念都不懂，会用大白话描述'],
    ['情绪不满型', '带着负面情绪描述问题，措辞激烈'],
    ['务实计较型', '关注费用、时限、性价比'],
    ['比较权衡型', '总在问"别家怎么样""哪个更好"'],
    ['特殊处境型', '处境特殊（深夜/异地/紧急），需求迫切'],
    ['好奇探索型', '爱追问原理和"为什么"，想要扩展知识'],
  ];
  return dimensions.map(([name, trait]) => name + '：' + trait + '（所在领域：' + domain + '）');
}

// ── 合成任务管理（持久化：后台任务 + 进度 + 中断 + 重启可恢复）──
// 历史缺陷：任务只存在内存 Map，服务重启后进度与产物全丢；且 2 小时 TTL 会
// 把仍可追溯的历史任务抹掉。现在每个任务落盘到 data/model-training/synth-tasks/，
// 内存只作最近 20 个的缓存，查不到时回读磁盘。
const fs = require('node:fs');
const path = require('node:path');
const STORE = require('./json-store');
const TASKS_DIR = path.join(__dirname, '..', 'data', 'model-training', 'synth-tasks');
try { fs.mkdirSync(TASKS_DIR, { recursive: true }); } catch (e) {}

const TASK_ID_RE = /^syn-[a-z0-9]+$/;
const FILE_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const MAX_IN_MEMORY = 20;
const synthTasks = new Map();
let taskSeq = 0;

function taskFile(id) { return path.join(TASKS_DIR, id + '.json'); }
function isTaskId(id) { return TASK_ID_RE.test(String(id == null ? '' : id)); }

// 落盘只保留可序列化字段：AbortController、中间快照、密钥一律不写文件
function serializeTask(t) {
  const o = {};
  for (const [k, v] of Object.entries(t)) {
    if (k === 'controller' || k === 'partialSamples' || k.startsWith('_')) continue;
    if (/apiKey|secret|token|password/i.test(k)) continue;
    o[k] = v;
  }
  return o;
}
function persistTask(t) {
  if (!isTaskId(t.id)) return;
  try { STORE.writeJsonSync(taskFile(t.id), serializeTask(t)); } catch (e) { /* 落盘失败不阻断合成 */ }
}
function readTaskFile(id) {
  if (!isTaskId(id)) return null;
  try {
    const o = STORE.readJson(taskFile(id), null);
    return o && o.id === id ? o : null;
  } catch (e) { return null; }
}

// 启动时恢复：服务重启时仍在 running 的后台协程已不存在，标记 interrupted 而非假装还在跑
function loadPersistedTasks() {
  let files = [];
  try { files = fs.readdirSync(TASKS_DIR).filter(f => /^syn-[a-z0-9]+\.json$/.test(f)); } catch (e) { return; }
  const cutoff = Date.now() - FILE_TTL_MS;
  const loaded = [];
  for (const f of files) {
    const id = f.slice(0, -5);
    const o = readTaskFile(id);
    if (!o) continue;
    if (o.finishedAt && o.finishedAt < cutoff) {
      try { fs.unlinkSync(taskFile(id)); } catch (e) {}
      continue;
    }
    if (o.status === 'running' || o.status === 'pending') {
      o.status = 'interrupted';
      o.finishedAt = Date.now();
      o.interruptedAt = o.finishedAt;
      o.errors = ['服务重启导致任务中断：后台生成协程已不存在，仅保留已完成样本'].concat(o.errors || []).slice(0, 12);
      o.samples = o.samples || [];
      STORE.writeJsonSync(taskFile(id), o);
    }
    o.controller = null; o.abort = false; o.partialSamples = [];
    loaded.push(o);
  }
  loaded.sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0));
  for (const o of loaded.slice(0, MAX_IN_MEMORY)) synthTasks.set(o.id, o);
}
loadPersistedTasks();

// 内存回收只淘汰终态任务：绝不移除运行中任务，否则会变成不可查、不可停的幽灵任务
function pruneMemory() {
  if (synthTasks.size <= MAX_IN_MEMORY) return;
  const finished = [...synthTasks.entries()]
    .filter(([, t]) => t.status !== 'running' && t.status !== 'pending')
    .sort((a, b) => (a[1].startedAt || 0) - (b[1].startedAt || 0));
  const excess = synthTasks.size - MAX_IN_MEMORY;
  for (let i = 0; i < Math.min(excess, finished.length); i++) synthTasks.delete(finished[i][0]);
}

function publicTask(t) {
  const o = Object.assign({}, t);
  delete o.controller;
  delete o.partialSamples;
  for (const k of Object.keys(o)) if (k.startsWith('_')) delete o[k];
  if (t.status === 'running' || t.status === 'pending') delete o.samples;
  if (t.status === 'stopped' || t.status === 'failed' || t.status === 'interrupted') {
    o.partialCount = (t.samples || []).length;
    o.partialNote = '仅为本轮已生成样本，未经过完整质检漏斗，请勿直接注册为训练数据集';
  }
  return o;
}

function startSynthTask(p) {
  const id = 'syn-' + Date.now().toString(36) + (++taskSeq).toString(36);
  const controller = new AbortController();
  const task = {
    id, status: 'running', mode: p.mode || 'evol', domain: p.domain || '',
    targetKind: p.targetKind || 'sft', backend: p.backend || 'remote', model: p.model || '',
    engine: '', total: Math.min(Math.max(p.count || 20, 1), 500), done: 0,
    samples: [], partialSamples: [], errors: [], mixWarning: null, funnel: null,
    quality: null, llmJudge: null, rawCount: 0,
    // 只记非敏感参数：apiKey / judgeApiKey 绝不落盘
    rules: {
      minScore: Number(p.minScore) || 0,
      semanticThreshold: p.semanticThreshold || null,
      llmJudge: !!p.llmJudge,
      judgeThreshold: p.judgeThreshold == null ? 0.5 : Number(p.judgeThreshold),
      judgeBackend: p.judgeBackend || null,
      lang: p.lang || '',
    },
    startedAt: Date.now(), finishedAt: null, abort: false, controller,
    _lastPersistDone: 0, _lastPersistAt: Date.now(),
  };
  synthTasks.set(id, task);
  persistTask(task);
  // 后台执行（不阻塞响应）
  (async () => {
    try {
      const result = await synthesize(Object.assign({}, p, {
        _abortSignal: controller.signal,
        _progress: (done, total, partial) => {
          task.done = done;
          // 每轮同步已生成样本：失败/停止时不再丢弃全部产物
          if (Array.isArray(partial)) task.partialSamples = partial.slice();
          const now = Date.now();
          if (done - task._lastPersistDone >= 5 || now - task._lastPersistAt >= 3000) {
            task._lastPersistDone = done; task._lastPersistAt = now;
            persistTask(task);   // 进度节流落盘，崩溃/重启后仍能恢复
          }
          return task.abort;  // 返回 true 则中断
        },
      }));
      task.samples = result.samples;
      task.funnel = result.funnel;
      task.quality = result.funnel && result.funnel.quality || null;
      task.llmJudge = result.funnel && result.funnel.llmJudge || null;
      task.engine = result.engine || '';
      task.backend = result.backend || task.backend;
      task.model = result.model || task.model;
      task.rawCount = result.rawCount || 0;
      task.mixWarning = result.mixWarning;
      task.errors = (result.errors || []).map(function (x) { return sanitizeErr(x); });
      task.status = 'completed';
    } catch (e) {
      if (task.abort || /中断/.test(String(e.message || ''))) { task.status = 'stopped'; }
      else { task.status = 'failed'; }
      task.errors = [sanitizeErr(e.message, 200)].concat(task.errors || []).slice(0, 12);
      task.samples = task.partialSamples || [];
      task.partialCount = task.samples.length;
    } finally {
      task.finishedAt = Date.now();
      task.controller = null;
      persistTask(task);
    }
  })();
  pruneMemory();
  return task;
}

function getSynthTask(id) {
  let t = synthTasks.get(id);
  if (!t) {
    t = readTaskFile(id);
    if (t) {
      t.controller = null; t.abort = false; t.partialSamples = [];
      synthTasks.set(id, t);
      pruneMemory();
    }
  }
  return t ? publicTask(t) : null;
}

function stopSynthTask(id) {
  const t = synthTasks.get(id);
  if (!t) return false;
  t.abort = true;
  // 立即掐断在途云 API / 远端请求，而不是等当前请求返回后才停
  if (t.controller) { try { t.controller.abort(); } catch (e) {} }
  persistTask(t);
  return true;
}

// 历史任务列表：磁盘全量 + 内存最新，只返回摘要，不回传样本
function listSynthTasks(limit) {
  const out = new Map();
  for (const [id, t] of synthTasks) out.set(id, t);
  try {
    for (const f of fs.readdirSync(TASKS_DIR)) {
      const id = f.replace(/\.json$/, '');
      if (!isTaskId(id) || out.has(id)) continue;
      const o = readTaskFile(id);
      if (o) out.set(id, o);
    }
  } catch (e) {}
  return [...out.values()]
    .sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0))
    .slice(0, Math.max(1, Math.min(Number(limit) || 20, 100)))
    .map(t => ({
      id: t.id, status: t.status, mode: t.mode, domain: t.domain,
      targetKind: t.targetKind, backend: t.backend, engine: t.engine, model: t.model,
      total: t.total, done: t.done, count: (t.samples || []).length,
      passRate: t.funnel ? t.funnel.passRate : null,
      startedAt: t.startedAt, finishedAt: t.finishedAt, interruptedAt: t.interruptedAt || null,
      hasSamples: !!(t.samples || []).length, mixWarning: !!t.mixWarning,
    }));
}

module.exports = {
  synthesize, listEngines, backends, PROMPTS, createLlmJudge, extractJSONArray,
  startSynthTask, getSynthTask, stopSynthTask, listSynthTasks,
};
