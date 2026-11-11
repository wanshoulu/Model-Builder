// ═══ 数据集工具路由（模块A：导入质检转换 / 模块B：自动合成）═══
const STORE = require('./json-store');
// 挂载点：server.js 在 model-training 路由之前调用 handleDatasetToolsRequest
function sendJson(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json;charset=utf-8' });
  res.end(JSON.stringify(obj));
}
function parseRequestBody(req, maxSize) {
  maxSize = maxSize || 50 * 1024 * 1024;
  return new Promise(function (resolve) {
    var chunks = [];
    var totalLen = 0;
    req.on('data', function (chunk) {
      totalLen += chunk.length;
      if (totalLen > maxSize) { req.destroy(); resolve({ error: 'Request body too large' }); return; }
      chunks.push(chunk);
    });
    req.on('end', function () {
      try { resolve({ data: JSON.parse(Buffer.concat(chunks).toString('utf8')) }); }
      catch (e) { resolve({ error: (e && e.message) || 'invalid json' }); }
    });
    req.on('error', function () { resolve({ error: 'request aborted' }); });
  });
}
const { sniffLayout, normalize, qualityFunnelAsync, convertTo, parseFile, LAYOUTS } = require('./dataset-tools');
const { synthesize, listEngines, createLlmJudge, startSynthTask, getSynthTask, stopSynthTask, listSynthTasks } = require('./dataset-synthesis');
const { guardImportPath } = require('./path-guard');
const { normalizeProvenance } = require('./model-training');  // 导入与合成注册共用同一套血缘规范

// 质检参数统一构造：导入与合成共用同一套 7 段漏斗，避免两处阈值漂移
function buildFunnelOptions(p, sniff, targetKind) {
  return {
    targetKind,
    targetLang: p.targetLang || null,
    minLen: p.minLen, maxLen: p.maxLen, minScore: p.minScore,
    semanticThreshold: p.semanticThreshold,
    judge: createLlmJudge(Object.assign({}, p, { targetKind })),
    judgeLimit: p.judgeLimit, judgeBatchSize: p.judgeBatchSize, judgeThreshold: p.judgeThreshold,
  };
}
function summarizeFunnel(funnel) {
  return {
    total: funnel.total, passed: funnel.passed, passRate: funnel.passRate,
    stages: funnel.stages, quality: funnel.quality || null, llmJudge: funnel.llmJudge || null,
  };
}

const JSON_LIMIT = 200 * 1024 * 1024;  // 200MB

async function handleDatasetToolsRequest(req, res) {
  const u = new URL(req.url, 'http://localhost');
  const pathname = u.pathname;
  if (!pathname.startsWith('/api/dataset-tools/')) return false;

  // ── 读取文档全文（doc 合成模式的数据源：txt/md 文件或文件夹）──
  if (pathname === '/api/dataset-tools/read-doc' && req.method === 'POST') {
    const body = await parseRequestBody(req, 10 * 1024 * 1024);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    const rawPath = (p.path || '').trim();
    if (!rawPath) return sendJson(res, 400, { ok: false, error: 'path 必填' });
    try {
      const fsp = require('node:fs');
      const pathMod = require('node:path');
      const guard = guardImportPath(rawPath);
      if (!guard.ok) return sendJson(res, 403, { ok: false, error: guard.error });
      const abs = guard.abs;
      const st = fsp.statSync(abs);
      const DOCS = ['.txt', '.md', '.markdown'];
      let files = [];
      const collect = (dir, depth) => {
        if (depth > 5) return;
        for (const e of fsp.readdirSync(dir, { withFileTypes: true })) {
          if (e.name.startsWith('.') || e.name === 'node_modules') continue;
          const fp = pathMod.join(dir, e.name);
          if (e.isDirectory()) collect(fp, depth + 1);
          else if (DOCS.includes(pathMod.extname(e.name).toLowerCase())) files.push(fp);
        }
      };
      if (st.isDirectory()) collect(abs, 0);
      else if (DOCS.includes(pathMod.extname(abs).toLowerCase())) files.push(abs);
      else return sendJson(res, 400, { ok: false, error: '不支持的文档类型（支持 ' + DOCS.join('/') + '）' });
      if (!files.length) return sendJson(res, 400, { ok: false, error: '文件夹内没有 txt/md 文档' });
      if (files.length > 100) files = files.slice(0, 100);
      const parts = [];
      const names = [];
      for (const f of files) {
        const c = fsp.readFileSync(f, 'utf8');
        if (c.trim()) { parts.push(c); names.push(pathMod.basename(f)); }
        if (parts.join('').length > 20 * 1024 * 1024) break;  // 20MB 上限
      }
      const text = parts.join('\n\n');
      if (!text.trim()) return sendJson(res, 400, { ok: false, error: '文档内容为空' });
      return sendJson(res, 200, { ok: true, text, files: files.length, names });
    } catch (e) {
      return sendJson(res, 400, { ok: false, error: '路径不存在或不可读：' + e.message.slice(0, 100) });
    }
  }

  // ── 拖拽多文件上传（Easy Dataset 对标：批量文件 → 解析合并）──
  if (pathname === '/api/dataset-tools/upload' && req.method === 'POST') {
    try {
      const formidable = require('formidable');
      const fsp = require('node:fs');
      const pathMod = require('node:path');
      const form = new (formidable.Formidable || formidable.IncomingForm || formidable)({ maxFileSize: 200 * 1024 * 1024 });
      const [fields, filesObj] = await new Promise((resolve, reject) => {
        form.parse(req, (err, fields, files) => err ? reject(err) : resolve([fields, files]));
      });
      // 兼容单文件字段 file 与多文件字段 files[]
      let uploads = [];
      for (const key of Object.keys(filesObj)) {
        const v = filesObj[key];
        if (Array.isArray(v)) uploads = uploads.concat(v); else uploads.push(v);
      }
      if (!uploads.length) return sendJson(res, 400, { ok: false, error: '未收到文件' });
      const EXTS = ['.jsonl', '.json', '.csv'];
      const samples = [];
      const fileReports = [];
      for (const f of uploads) {
        const name = f.originalFilename || f.name || 'unnamed';
        if (!EXTS.includes(pathMod.extname(name).toLowerCase())) {
          fileReports.push({ file: name, count: 0, error: '不支持的类型（仅 json/jsonl/csv）' });
          continue;
        }
        try {
          const content = fsp.readFileSync(f.filepath, 'utf8');
          const parsed = parseFile(content, name);
          samples.push(...parsed);
          fileReports.push({ file: name, count: parsed.length, error: null });
        } catch (e) {
          fileReports.push({ file: name, count: 0, error: e.message.slice(0, 100) });
        }
      }
      if (!samples.length) {
        return sendJson(res, 400, { ok: false, error: '0 条样本被解析', files: fileReports });
      }
      return sendJson(res, 200, { ok: true, samples, files: fileReports, totalFiles: uploads.length });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: '上传解析失败：' + e.message.slice(0, 150) });
    }
  }

  // ── 本地路径导入：扫描文件/文件夹（json/jsonl/csv，支持目录递归与多文件合并）──
  if (pathname === '/api/dataset-tools/import-path' && req.method === 'POST') {
    const body = await parseRequestBody(req, 10 * 1024 * 1024);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    const rawPath = (p.path || '').trim();
    if (!rawPath) return sendJson(res, 400, { ok: false, error: 'path 必填（本地文件或文件夹绝对路径）' });
    try {
      const fsp = require('node:fs');
      const pathMod = require('node:path');
      const guard = guardImportPath(rawPath);
      if (!guard.ok) return sendJson(res, 403, { ok: false, error: guard.error });
      const abs = guard.abs;
      const st = fsp.statSync(abs);
      // 收集目标文件
      let files = [];
      const EXTS = ['.jsonl', '.json', '.csv'];
      // 元数据/配置文件忽略清单（业界惯例：这些是数据集的"说明书"不是数据本身）
      const IGNORE_NAMES = ['manifest', 'dataset_info', 'meta', 'metadata', 'schema', 'config', 'index', 'readme', 'license', 'terms'];
      const isIgnored = (name) => {
        const base = name.toLowerCase().replace(/\.(json|jsonl|csv|md|txt)$/i, '');
        return IGNORE_NAMES.includes(base) || name.startsWith('.');
      };
      const collect = (dir, depth) => {
        if (depth > 5) return;  // 防深层递归
        for (const e of fsp.readdirSync(dir, { withFileTypes: true })) {
          if (isIgnored(e.name)) continue;
          const fp = pathMod.join(dir, e.name);
          if (e.isDirectory()) {
            if (e.name === 'node_modules') continue;
            collect(fp, depth + 1);
          } else if (EXTS.includes(pathMod.extname(e.name).toLowerCase())) {
            files.push(fp);
          }
        }
      };
      if (st.isDirectory()) collect(abs, 0);
      else if (EXTS.includes(pathMod.extname(abs).toLowerCase())) files.push(abs);
      else return sendJson(res, 400, { ok: false, error: '不支持的文件类型（支持 ' + EXTS.join('/') + '），路径：' + abs });
      if (!files.length) return sendJson(res, 400, { ok: false, error: '文件夹内没有 json/jsonl/csv 文件：' + abs });
      if (files.length > 100) files = files.slice(0, 100);  // 上限保护

      // 逐文件解析合并
      const samples = [];
      const fileReports = [];
      for (const f of files) {
        try {
          const content = fsp.readFileSync(f, 'utf8');
          const parsed = parseFile(content, pathMod.basename(f));
          samples.push(...parsed);
          fileReports.push({ file: pathMod.relative(abs, f) || pathMod.basename(f), count: parsed.length, error: null });
        } catch (e) {
          fileReports.push({ file: pathMod.basename(f), count: 0, error: e.message.slice(0, 100) });
        }
      }
      const totalOk = fileReports.filter(r => !r.error).length;
      if (!samples.length) {
        return sendJson(res, 400, {
          ok: false,
          error: '解析出 0 条样本（' + totalOk + '/' + files.length + ' 个文件解析失败）',
          files: fileReports,
        });
      }
      return sendJson(res, 200, {
        ok: true,
        samples,
        files: fileReports,
        path: abs,
        isDir: st.isDirectory(),
        totalFiles: files.length,
      });
    } catch (e) {
      return sendJson(res, 400, { ok: false, error: '路径不存在或不可读：' + e.message.slice(0, 120) });
    }
  }

  // ── 布局清单（前端下拉）──
  if (pathname === '/api/dataset-tools/layouts' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, layouts: LAYOUTS });
  }

  // ── 导入分析：嗅探 + 质检漏斗（纯只读分析，不保存）──
  if (pathname === '/api/dataset-tools/analyze' && req.method === 'POST') {
    const body = await parseRequestBody(req, JSON_LIMIT);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    try {
      // 输入二选一：rawContent+fileName（直接粘贴/上传）或 samples 数组
      let samples = p.samples;
      if (!samples && p.rawContent) samples = parseFile(p.rawContent, p.fileName || 'data.json');
      if (!Array.isArray(samples) || !samples.length) return sendJson(res, 400, { ok: false, error: 'samples 数组或 rawContent 必填' });
      if (samples.length > 200000) return sendJson(res, 400, { ok: false, error: '样本数超过 20 万上限' });

      const sniff = sniffLayout(samples);
      const targetKind = p.targetKind || (LAYOUTS[sniff.layout] ? LAYOUTS[sniff.layout].kind : undefined);
      const funnel = await qualityFunnelAsync(normalize(samples, sniff.layout), buildFunnelOptions(p, sniff, targetKind));
      return sendJson(res, 200, {
        ok: true,
        layout: sniff,
        suggestedKind: LAYOUTS[sniff.layout] ? LAYOUTS[sniff.layout].kind : 'sft',
        funnel: summarizeFunnel(funnel),
        preview: funnel.samples.slice(0, 5),  // 通过样本预览
        rejectedPreview: samples.slice(0, 3), // 原始样本预览
      });
    } catch (e) {
      return sendJson(res, 400, { ok: false, error: '解析失败：' + e.message });
    }
  }

  // ── 转换并注册：质检通过后转成目标格式，直接注册进工作台数据集 ──
  if (pathname === '/api/dataset-tools/convert' && req.method === 'POST') {
    const body = await parseRequestBody(req, JSON_LIMIT);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    try {
      let samples = p.samples;
      if (!samples && p.rawContent) samples = parseFile(p.rawContent, p.fileName || 'data.json');
      if (!Array.isArray(samples) || !samples.length) return sendJson(res, 400, { ok: false, error: 'samples 或 rawContent 必填' });

      const sniff = sniffLayout(samples);
      const targetKind = p.targetKind || (LAYOUTS[sniff.layout] ? LAYOUTS[sniff.layout].kind : 'sft');
      if (targetKind === 'decision' && sniff.layout !== 'decision_kev') {
        return sendJson(res, 400, { ok: false, error: '决策格式只能从 decision_kev 布局转换（源布局: ' + sniff.layout + '）' });
      }
      // 质检漏斗（转换前强制执行——脏数据进训练集是事故之源）
      const funnel = await qualityFunnelAsync(normalize(samples, sniff.layout), buildFunnelOptions(p, sniff, targetKind));
      if (!funnel.samples.length) return sendJson(res, 400, { ok: false, error: '质检后样本数为 0——全部被漏斗拦截，请检查数据或放宽阈值' });
      // 转换：funnel.samples 已是归一化后的内部格式，只做一次映射，禁止二次 normalize
      const converted = convertTo(funnel.samples, targetKind);
      // 注册（复用工作台数据集存储：走内部注册接口逻辑）
      const reg = await registerDataset({
        name: p.name || ('imported-' + Date.now()),
        kind: targetKind,
        description: p.description || ('导入转换自 ' + sniff.layout + '（漏斗通过率 ' + (funnel.passRate * 100).toFixed(1) + '%）'),
        samples: converted,
        provenance: {
          schemaVersion: 1,
          source: 'import',
          sourceLayout: sniff.layout,
          sourceLayoutConfidence: sniff.confidence,
          targetKind,
          importedAt: Date.now(),
          rules: {
            targetLang: p.targetLang || null,
            minLen: p.minLen || null, maxLen: p.maxLen || null,
            minScore: Number(p.minScore) || 0,
            semanticThreshold: p.semanticThreshold || null,
            llmJudge: !!p.llmJudge,
          },
          funnel: summarizeFunnel(funnel),
        },
      });
      return sendJson(res, 200, { ok: true, dataset: reg, funnel: summarizeFunnel(funnel) });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: e.message });
    }
  }

  // ── 合成引擎清单（模块B）──
  if (pathname === '/api/dataset-tools/engines' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, engines: listEngines() });
  }

  // ── 数据合成（模块B：同步版，小批量）──
  if (pathname === '/api/dataset-tools/synthesize' && req.method === 'POST') {
    const body = await parseRequestBody(req, JSON_LIMIT);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    try {
      const result = await synthesize(p);
      return sendJson(res, 200, { ok: true, ...result });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: e.message });
    }
  }

  // ── 合成任务历史（持久化：服务重启后可查、可回读结果重新注册）──
  if (pathname === '/api/dataset-tools/synth-tasks' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, tasks: listSynthTasks(u.searchParams.get('limit')) });
  }

  // ── 合成任务（异步版：立返任务ID，后台跑，可查进度/中断——大批量用这个）──
  if (pathname === '/api/dataset-tools/synth-tasks' && req.method === 'POST') {
    const body = await parseRequestBody(req, JSON_LIMIT);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    try {
      if ((p.count || 20) <= 30) return sendJson(res, 400, { ok: false, error: '小批量（≤30条）请用同步 synthesize 接口，秒级返回无需任务' });
      const task = startSynthTask(p);
      return sendJson(res, 200, { ok: true, task: { id: task.id, status: task.status, total: task.total } });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: e.message });
    }
  }
  const taskMatch = pathname.match(/^\/api\/dataset-tools\/synth-tasks\/([^\/]+)(?:\/(stop))?$/);
  if (taskMatch) {
    const task = getSynthTask(taskMatch[1]);
    if (!task) return sendJson(res, 404, { ok: false, error: 'task not found' });
    if (taskMatch[2] === 'stop' && req.method === 'POST') {
      if (task.status === 'running') {
        stopSynthTask(task.id);
        return sendJson(res, 200, { ok: true, task: Object.assign({}, task, { status: 'stopping' }) });
      }
      return sendJson(res, 200, { ok: true, task });
    }
    return sendJson(res, 200, { ok: true, task });
  }

  return false;
}

// 内部：注册数据集（复用 model-training 的存储路径与 meta 规范）
const fs = require('node:fs');
const path = require('node:path');
const DATASETS_DIR = path.join(__dirname, '..', 'data', 'model-training', 'datasets');
try { fs.mkdirSync(DATASETS_DIR, { recursive: true }); } catch (e) {}
function newId(prefix) {
  return prefix + '-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}
async function registerDataset(ds) {
  ds.id = newId('ds');
  ds.fileName = ds.id + '.json';
  ds.createdAt = Date.now();
  ds.provenance = normalizeProvenance(ds.provenance, ds);
  // 原子写：避免注册过程中进程被杀留下半截数据集（旧 readJson 会把损坏文件当空库）
  STORE.writeJsonSync(path.join(DATASETS_DIR, ds.fileName), ds.samples);
  STORE.writeJsonSync(path.join(DATASETS_DIR, ds.id + '.meta.json'),
    Object.assign({}, ds, { samples: undefined, count: ds.samples.length }));
  return { id: ds.id, name: ds.name, kind: ds.kind, count: ds.samples.length, createdAt: ds.createdAt };
}

module.exports = { handleDatasetToolsRequest };
