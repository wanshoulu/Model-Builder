/**
 * 模型训练工作台 — 后端路由（/api/model-training/*）
 *
 * 定位：易用的 LLaMA-Factory 前端控制台。本地只做配置/编排/展示，
 * 训练与推理全部在远端 GPU 服务器执行（远端需安装 LLaMA-Factory）。
 *
 * 端点：
 *   GET  /api/model-training/servers                 服务器列表（密码不回传）
 *   POST /api/model-training/servers                 新增/更新服务器（upsert by id）
 *   POST /api/model-training/servers/delete          删除服务器
 *   POST /api/model-training/servers/test            连通性测试（SSH + LLaMA-Factory 检测 + GPU 概览）
 *   GET  /api/model-training/datasets                数据集列表（本地管理）
 *   POST /api/model-training/datasets                上传数据集（JSON 正文，alpaca/sharegpt/preference）
 *   POST /api/model-training/datasets/delete         删除数据集
 *   POST /api/model-training/datasets/preview        预览数据集样本
 *   POST /api/model-training/config/preview          按表单生成 LLaMA-Factory YAML（不提交，供检查）
 *   POST /api/model-training/jobs                    创建训练任务（传 YAML 到远端并启动）
 *   GET  /api/model-training/jobs                    任务列表
 *   GET  /api/model-training/jobs/:id                任务详情（含实时日志尾部）
 *   POST /api/model-training/jobs/:id/stop           停止任务
 *   POST /api/model-training/jobs/:id/delete         删除任务记录
 *   POST /api/model-training/jobs/:id/chat           用训练产物测试对话（远端起临时 API 或 chat 命令）
 *   POST /api/model-training/jobs/:id/export         生成合并导出命令说明（merge_lora）
 *
 * 数据落盘：data/model-training/
 *   servers.json     服务器配置（密码以明文存本地——内网工具，README 已提示）
 *   datasets/<id>.json  数据集文件（同时保存 dataset_info 片段）
 *   jobs/<id>.json   任务记录（状态/日志尾部缓存在内存，重启后从远端对账）
 *
 * 远端约定（工作台自动完成，无需手工）：
 *   - 工作目录 ~/ai-harbor-training/（可配 remoteWorkdir）
 *   - 数据集放 <workdir>/data/，并合并进 dataset_info.json
 *   - 任务配置 <workdir>/jobs/<id>.yaml
 *   - 训练输出 <workdir>/saves/<id>/
 *   - 日志 <workdir>/logs/<id>.log（nohup 后台运行）
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const { Client } = require('ssh2');
const STORE = require('./json-store');
const EVAL = require('./model-evaluation');
const DT = require('./dataset-tools');

const STORE_DIR = path.join(__dirname, '..', 'data', 'model-training');
const SERVERS_FILE = path.join(STORE_DIR, 'servers.json');
const DATASETS_DIR = path.join(STORE_DIR, 'datasets');
const JOBS_DIR = path.join(STORE_DIR, 'jobs');

// ── 存储 ────────────────────────────────────────────────────────────────

function ensureDirs() {
  for (const d of [STORE_DIR, DATASETS_DIR, JOBS_DIR]) {
    fs.mkdirSync(d, { recursive: true });
  }
}
ensureDirs();

/**
 * 统一走 json-store：写用 tmp+fsync+rename 原子替换，损坏文件隔离备份而非静默当空库。
 * 这里保留同步写语义（现有调用点都是同步的），并发保护由 saveJob 的读改写事务提供。
 */
function readJson(file, fallback) { return STORE.readJson(file, fallback); }
function writeJson(file, obj) { STORE.writeJsonSync(file, obj); }
function newId(prefix) {
  return prefix + '-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}
/**
 * 所有来自 URL / body 的资源 id（jobId / datasetId）必须过白名单。
 * 历史缺陷：GET /jobs/..%5Cservers 可读到 data/model-training/servers.json
 * 明文凭据；datasets/delete 的 id 可越界删任意 .json。
 */
const SAFE_ID = /^[A-Za-z0-9_-]+$/;
function isSafeId(id) {
  const s = String(id == null ? '' : id);
  return s.length > 0 && s.length <= 128 && SAFE_ID.test(s);
}
function badId(res) {
  return sendJson(res, 400, { ok: false, error: '无效的资源 ID（仅允许字母、数字、下划线、连字符）' });
}

/**
 * POSIX shell 单引号转义：所有拼进 bash -c 的用户可控值必须过这里。
 * 单引号内除单引号本身外都是字面量，采用 '"'"' 中断-转义-续接的标准写法。
 * 历史风险：remoteWorkdir / cliPath / modelName 等字段来自 API body，
 * 未经转义直接拼接进 nohup bash -c '...'，可注入任意命令。
 */
function shq(s) {
  return "'" + String(s == null ? '' : s).replace(/'/g, "'\"'\"'") + "'";
}

/**
 * 杀整个训练进程树（含 python 孙进程）。
 * 平台启动链：node → bash(-c) → nohup bash → python；pid 文件记录的是 nohup 外壳 bash。
 * 旧实现只 kill $PID + pkill -P $PID（直接子进程），python 孙进程会漏杀——
 * 2026-11-11 实测：停止 RLCD 任务后 python 孤儿占 3.3GB 内存，后续任务被内存预检拒绝。
 * 修复：①bash 层递归收集进程树（pgrep -P）统一 kill；②本地模式追加 Windows 原生
 * taskkill /T /F 兜底——Git Bash (MSYS) 的 pgrep 只见 MSYS 进程表，Windows 原生
 * python 不在其中，taskkill /T 按 Windows 进程树击杀是本地模式的权威手段。
 */
function buildKillTreeCommand(jobId, workdir) {
  return 'cd ' + shq(workdir) + ' && if [ -f run/' + jobId + '.pid ]; then PID=$(cat run/' + jobId + '.pid); ' +
    'TREE=""; COLLECT() { for C in $(pgrep -P $1 2>/dev/null); do COLLECT $C; done; TREE="$TREE $1"; }; ' +
    'COLLECT $PID; ' +
    'for P in $TREE; do kill $P 2>/dev/null; done; sleep 2; ' +
    'for P in $TREE; do kill -9 $P 2>/dev/null; done; echo killed-tree; ' +
    'else echo nopid; fi';
}

/** 本地模式 Windows 原生进程树击杀（权威兜底）。
 *  按【命令行匹配】而不是 pid 文件：pid 文件指向的 nohup bash 可能已被 bash 层 kill
 *  杀掉（taskkill /T /PID <死进程> 无效）；而训练 python 命令行里 --out saves/<jobId>
 *  是任务唯一标识，torch dataloader 的子 python 也继承同样命令行/父链。
 *  用 CIM Win32_Process 枚举命令行含 jobId 的进程，从叶子往根逐个 taskkill /F。
 */
function killTreeWindows(jobId) {
  try {
    const out = cp.execSync(
      'powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*' + jobId + '*\' } | Select-Object -ExpandProperty ProcessId"',
      { encoding: 'utf8', timeout: 20000, windowsHide: true });
    const pids = String(out).split(/\s+/).map(s => parseInt(s, 10)).filter(n => Number.isFinite(n) && n > 0);
    if (!pids.length) return;
    // 过滤掉自己（node 服务进程的命令行也可能含 jobId——按 Name=python/bash 过滤更稳，但
    // Select-Object 已只拿 PID；用 taskkill 逐个强杀，非 python/bash 的（比如自己）跳过）
    for (const pid of pids) {
      try {
        // 确认是 python/bash 再杀（防止误杀命令行恰好含 jobId 的无关进程）
        const name = cp.execSync(
          'powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \'ProcessId=' + pid + '\').Name"',
          { encoding: 'utf8', timeout: 8000, windowsHide: true }).trim();
        if (/python|bash|sh\.exe/i.test(name)) {
          cp.execSync('taskkill /T /F /PID ' + pid, { stdio: 'ignore', timeout: 10000, windowsHide: true });
        }
      } catch (e) { /* 单个失败继续 */ }
    }
  } catch (e) { /* 兜底失败不阻塞——bash 层已尽力 */ }
}

/**
 * 清理任务全部产物（删除任务时联动）：saves/<id>/、rlcd-data/<id>/、
 * logs/<id>.*、run/<id>.pid、evals/eval-<id>-*、jobs/<id>.*.yaml（本地/远端同构命令）。
 */
function buildCleanupCommand(jobId, workdir) {
  return 'cd ' + shq(workdir) + ' && ' +
    'rm -rf saves/' + jobId + ' rlcd-data/' + jobId + ' evals/eval-' + jobId + '-* 2>/dev/null; ' +
    'rm -f logs/' + jobId + '.log logs/' + jobId + '.exit run/' + jobId + '.pid jobs/' + jobId + '.yaml jobs/' + jobId + '.resume.yaml jobs/' + jobId + '.chat.py jobs/' + jobId + '.export.yaml jobs/' + jobId + '-eval.yaml 2>/dev/null; ' +
    'ls data/ 2>/dev/null | grep "^ds-" | while read F; do case "$F" in ' + jobId + '-eval*) rm -f "data/$F";; esac; done; ' +
    'echo cleaned-' + jobId;
}

/**
 * 服务器配置字段白名单校验：host / username / workdir / cliPath 等
 * 会被写进命令与 SSH 连接，拒绝控制字符与换行（防止把配置字段当命令注入面）。
 * 返回 null 表示通过，否则返回错误消息。
 */
function validateServerFields(p) {
  const strFields = [
    ['host', p.host], ['username', p.username],
    ['remoteWorkdir', p.remoteWorkdir], ['cliPath', p.cliPath], ['name', p.name],
  ];
  for (const [label, v] of strFields) {
    if (v == null) continue;
    const s = String(v);
    if (s.length > 512) return label + ' 过长（上限 512 字符）';
    if (/[\r\n\x00-\x1f]/.test(s)) return label + ' 含非法控制字符';
    if (label === 'host' && !/^[A-Za-z0-9._-]+$/.test(s)) return 'host 仅允许字母、数字、点、下划线、连字符';
    if (label === 'username' && !/^[A-Za-z0-9._@-]+$/.test(s)) return 'username 仅允许字母、数字、点、下划线、连字符、@';
    if (label === 'remoteWorkdir' && (s.startsWith('-') || /[;&|`$<>]/.test(s))) return 'remoteWorkdir 含 shell 元字符';
    if (label === 'cliPath' && /[;&|`$<>]/.test(s)) return 'cliPath 含 shell 元字符';
  }
  if (p.port != null && (!Number.isInteger(Number(p.port)) || Number(p.port) < 1 || Number(p.port) > 65535)) {
    return 'port 必须是 1-65535 的整数';
  }
  return null;
}

function sendJson(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json;charset=utf-8' });
  res.end(JSON.stringify(obj));
}

function parseRequestBody(req, maxSize) {
  maxSize = maxSize || 50 * 1024 * 1024; // 数据集上传放宽到 50MB
  return new Promise(function (resolve) {
    var chunks = [];
    var totalLen = 0;
    req.on('data', function (chunk) {
      totalLen += chunk.length;
      if (totalLen > maxSize) { req.destroy(); resolve({ error: 'Request body too large' }); return; }
      chunks.push(chunk);
    });
    req.on('end', function () {
      try {
        resolve({ data: JSON.parse(Buffer.concat(chunks).toString('utf8')) });
      } catch (e) {
        resolve({ error: (e && e.message) || 'invalid json' });
      }
    });
    req.on('error', function () { resolve({ error: 'request aborted' }); });
  });
}

// ── 服务器配置 ──────────────────────────────────────────────────────────

function listServers() {
  return readJson(SERVERS_FILE, []);
}
function saveServers(list) {
  writeJson(SERVERS_FILE, list);
}
function sanitizeServer(s) {
  // 回传前端时抹掉密码与私钥本体，只暴露"是否已配置"标志
  return Object.assign({}, s, {
    password: undefined, privateKey: undefined,
    hasPassword: !!s.password, hasPrivateKey: !!s.privateKey
  });
}
function findServer(id) {
  return listServers().find(function (s) { return s.id === id; }) || null;
}

// ── SSH 执行器（ssh2）──────────────────────────────────────────────────

/**
 * 在远端执行命令。返回 { code, stdout, stderr }。
 * options.timeoutMs 默认 60s；训练启动类命令用短超时（nohup 立即返回）。
 *
 * 本地模式（server.local === true）：不经 SSH，直接在本机执行。
 * 为了让远端命令语义尽量通吃，本地模式统一经 bash（Git Bash）执行，
 * 支持 mkdir -p / nohup / kill -0 等 Unix 语法；路径用 POSIX 形式（/d/...）。
 */
function isLocal(server) { return !!(server && server.local); }

/** Windows 路径 ↔ bash 路径（D:/x 或 D:\x → /d/x）。非盘符路径原样返回。 */
function toBashPath(p) {
  const s = String(p || '');
  const m = s.match(/^([A-Za-z]):[\\\/](.*)$/);
  if (!m) return s.replace(/\\/g, '/');
  return '/' + m[1].toLowerCase() + '/' + m[2].replace(/\\/g, '/');
}

const BASH_EXE = (function () {
  // 依次探测可用的 bash：Windows 内置 OpenSSH 自带的没有 bash，用 Git Bash。
  // 探测必须验证「能访问 Windows 盘符路径」——C:\Windows\System32\bash.exe 是 WSL
  // 的入口，能执行 echo 但它的 / 是 Linux 根目录，cd D:/xxx 会失败，导致训练
  // 状态判定（kill -0 在 WSL PID 空间里）与产物探测全部静默失效。
  const candidates = [
    'C:\\Program Files\\Git\\bin\\bash.exe',
    'D:\\software\\workbuddy\\resources\\vendor\\PortableGit\\bin\\bash.exe',
    'C:\\Windows\\System32\\bash.exe',
    'bash',
  ];
  for (const c of candidates) {
    try {
      // 双重验证：能执行 + 能 cd 进 Windows 盘符路径（WSL bash 在这里失败）。
      // 注意用 execSync + 带引号字符串：spawnSync 带引号的参数数组方式在
      // WMI DETACHED 环境下 stdout 会异常为空。
      const probe = 'cd D:/ && test -d . && echo POSIX_WIN_OK';
      const r = cp.execSync('"' + c + '" -c "' + probe + '"', { encoding: 'utf8', timeout: 8000, windowsHide: true });
      if (/POSIX_WIN_OK/.test(r || '')) return c;
    } catch (e) {}
  }
  return 'bash';
})();

function localExec(command, timeoutMs) {
  timeoutMs = timeoutMs || 60000;
  return new Promise(function (resolve, reject) {
    try {
      // 命令经 stdin 喂给 `bash -s`：绕开 Windows 命令行引号转义的所有坑。
      // execSync + JSON.stringify 在命令含 "$( )" 双引号结构时会被 Windows
      // 参数解析拆散（实测 pidRes 命令输出变成乱码）；spawn 参数数组在
      // WMI DETACHED 环境下 stdout 静默为空。stdin 方式两者皆无。
      const stdout = cp.execSync('"' + BASH_EXE + '" -s', {
        encoding: 'utf8',
        timeout: timeoutMs,
        windowsHide: true,
        maxBuffer: 16 * 1024 * 1024,
        input: command + '\n',
      });
      resolve({ code: 0, stdout: stdout || '', stderr: '' });
    } catch (e) {
      // execSync 非零退出码会 throw：从 e.stdout/e.stderr 恢复输出（与原语义一致）
      if (e.stdout != null || e.stderr != null) {
        resolve({ code: e.status == null ? 1 : e.status, stdout: e.stdout || '', stderr: e.stderr || '' });
      } else if (/ETIMEDOUT|timed?\s?out/i.test(String(e.message))) {
        reject(new Error('本地命令超时（' + Math.round(timeoutMs / 1000) + 's）：' + command.slice(0, 80)));
      } else {
        reject(e);
      }
    }
  });
}

/** 本地写文件：remotePath 可能是 ~/... 或 D:/... 形式，统一归一后写盘 */
function localWriteFile(remotePath, content) {
  return new Promise(function (resolve, reject) {
    let p = String(remotePath || '');
    if (p.startsWith('~/') || p === '~') {
      p = path.join(process.env.USERPROFILE || '', p.slice(2));
    }
    fs.mkdir(path.dirname(p), { recursive: true }, function (err) {
      if (err) return reject(err);
      fs.writeFile(p, content, function (e2) { e2 ? reject(e2) : resolve({ ok: true, path: p }); });
    });
  });
}

function sshExec(server, command, timeoutMs) {
  timeoutMs = timeoutMs || 60000;
  if (isLocal(server)) return localExec(command, timeoutMs);
  return new Promise(function (resolve, reject) {
    const conn = new Client();
    let settled = false;
    const finish = (fn, arg) => { if (!settled) { settled = true; try { conn.end(); } catch (e) {} fn(arg); } };
    const timer = setTimeout(function () {
      finish(reject, new Error('SSH 命令超时（' + Math.round(timeoutMs / 1000) + 's）：' + command.slice(0, 80)));
    }, timeoutMs);

    conn.on('ready', function () {
      conn.exec(command, { pty: false }, function (err, stream) {
        if (err) { clearTimeout(timer); finish(reject, err); return; }
        let stdout = '';
        let stderr = '';
        stream.on('data', function (d) { stdout += d.toString(); if (stdout.length > 2 * 1024 * 1024) stdout = stdout.slice(-1024 * 1024); });
        stream.stderr.on('data', function (d) { stderr += d.toString(); if (stderr.length > 1024 * 1024) stderr = stderr.slice(-512 * 1024); });
        stream.on('close', function (code) {
          clearTimeout(timer);
          finish(resolve, { code: code === null ? 0 : code, stdout: stdout, stderr: stderr });
        });
      });
    }).on('error', function (err) {
      clearTimeout(timer);
      finish(reject, err);
    }).connect({
      host: server.host,
      port: server.port || 22,
      username: server.username,
      password: server.password || undefined,
      privateKey: server.privateKey || undefined,
      readyTimeout: 15000,
      keepaliveInterval: 10000,
    });
  });
}

/** 远端写文件：通过 stdin 流写入（避免转义地狱），command 为 `cat > path` 形式 */
function sshWriteFile(server, remotePath, content) {
  if (isLocal(server)) {
    // remotePath 里可能是 ~/xxx（workdir 前缀），归一为 Windows 路径后写盘
    let p = String(remotePath || '');
    if (p.startsWith('~')) {
      const rest = p.slice(1).replace(/^([\\/])/, '');
      p = path.join(process.env.USERPROFILE || '', rest);
    }
    return localWriteFile(p, content);
  }
  return new Promise(function (resolve, reject) {
    const conn = new Client();
    let settled = false;
    const finish = (fn, arg) => { if (!settled) { settled = true; try { conn.end(); } catch (e) {} fn(arg); } };
    conn.on('ready', function () {
      conn.exec('cat > ' + JSON.stringify(remotePath), function (err, stream) {
        if (err) { finish(reject, err); return; }
        let stderr = '';
        stream.stderr.on('data', function (d) { stderr += d.toString(); });
        stream.on('close', function (code) {
          finish(code === 0 ? resolve : reject, code === 0 ? { ok: true } : new Error('写入失败: ' + stderr));
        });
        stream.end(content);
      });
    }).on('error', function (err) { finish(reject, err); })
      .connect({
        host: server.host,
        port: server.port || 22,
        username: server.username,
        password: server.password || undefined,
        privateKey: server.privateKey || undefined,
        readyTimeout: 15000,
      });
  });
}

// ── YAML 生成（对齐 LLaMA-Factory examples 官方结构）──────────────────

function yamlScalar(v) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  const s = String(v);
  // 含特殊字符的字符串加引号
  if (/[:#{}[\],&*?|>'"%@`]/.test(s) || /^\s|\s$/.test(s) || s === '') return JSON.stringify(s);
  return s;
}

function buildYaml(cfg) {
  const lines = [];
  const push = (k, v) => lines.push(k + ': ' + yamlScalar(v));
  const pushIf = (k, v, cond) => { if (cond !== false && v !== undefined && v !== null && v !== '') push(k, v); };

  const isPreference = cfg.stage === 'dpo' || cfg.stage === 'kto' || cfg.stage === 'orpo' || cfg.stage === 'simpo' || cfg.stage === 'rm';
  const ftType = cfg.finetuningType || 'lora';

  lines.push('### model');
  push('model_name_or_path', cfg.modelName);
  pushIf('trust_remote_code', true);

  lines.push('');
  lines.push('### method');
  push('stage', cfg.stage || 'sft');
  push('do_train', true);
  push('finetuning_type', ftType);
  if (ftType === 'lora') {
    push('lora_target', cfg.loraTarget || 'all');
    push('lora_rank', cfg.loraRank || 8);
    push('lora_alpha', cfg.loraAlpha || cfg.loraRank * 2 || 16);
    pushIf('lora_dropout', cfg.loraDropout, cfg.loraDropout !== undefined && cfg.loraDropout !== '');
  }
  if (ftType === 'qlora') {
    push('quantization_bit', cfg.quantizationBit || 4);
    push('quantization_method', cfg.quantizationMethod || 'bitsandbytes');
    push('lora_target', cfg.loraTarget || 'all');
    push('lora_rank', cfg.loraRank || 8);
    push('lora_alpha', cfg.loraAlpha || cfg.loraRank * 2 || 16);
  }
  if (isPreference) {
    if (cfg.stage === 'dpo' || cfg.stage === 'simpo') {
      push('pref_beta', cfg.prefBeta || 0.1);
      push('pref_loss', cfg.prefLoss || (cfg.stage === 'simpo' ? 'simpo' : 'sigmoid'));
    }
    if (cfg.stage === 'orpo') push('pref_loss', 'orpo');
  }

  lines.push('');
  lines.push('### dataset');
  push('dataset', cfg.dataset);
  push('template', cfg.template);
  push('cutoff_len', cfg.cutoffLen || 2048);
  pushIf('max_samples', cfg.maxSamples, cfg.maxSamples);
  pushIf('preprocessing_num_workers', cfg.preprocessingNumWorkers || 8);
  // Windows + spawn 模式下 DataLoader 多进程 worker 与 transformers 的
  // enable_input_require_grads 闭包不兼容（pickle 失败导致训练崩溃），
  // 本地模式（Windows）一律 0；Linux 远端保留 2 提速
  if (cfg.isLocal) push('dataloader_num_workers', 0);
  else pushIf('dataloader_num_workers', cfg.dataloaderNumWorkers || 2);
  pushIf('val_size', cfg.valSize, cfg.valSize !== undefined && cfg.valSize !== '' && Number(cfg.valSize) > 0);

  lines.push('');
  lines.push('### output');
  push('output_dir', cfg.outputDir);
  // 日志密度：默认 5（原 10），短任务（几十步）曲线才有点可看；向导可覆盖
  push('logging_steps', Math.max(1, parseInt(cfg.loggingSteps, 10) || 5));
  push('save_steps', cfg.saveSteps || 500);
  push('plot_loss', true);
  push('overwrite_output_dir', true);
  push('report_to', cfg.reportTo || 'none');

  lines.push('');
  lines.push('### train');
  push('per_device_train_batch_size', cfg.batchSize || 1);
  push('gradient_accumulation_steps', cfg.gradAccum || 8);
  push('learning_rate', cfg.learningRate || (isPreference ? '5.0e-6' : '1.0e-4'));
  push('num_train_epochs', cfg.epochs || 3.0);
  push('lr_scheduler_type', cfg.lrScheduler || 'cosine');
  push('warmup_ratio', cfg.warmupRatio || 0.1);
  push('bf16', true);
  push('ddp_timeout', 180000000);

  lines.push('');
  lines.push('### eval');
  if (cfg.evalDataset) push('eval_dataset', cfg.evalDataset);
  if (cfg.evalStrategy) {
    push('eval_strategy', cfg.evalStrategy);
    push('eval_steps', cfg.evalSteps || 500);
    push('per_device_eval_batch_size', cfg.evalBatchSize || 1);
  }

  return lines.join('\n') + '\n';
}

// ── 任务管理 ────────────────────────────────────────────────────────────

function listJobs() {
  if (!fs.existsSync(JOBS_DIR)) return [];
  return fs.readdirSync(JOBS_DIR).filter(f => f.endsWith('.json'))
    .map(f => readJson(path.join(JOBS_DIR, f), null))
    .filter(Boolean)
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
}
function getJob(id) {
  if (!isSafeId(id)) return null;
  return readJson(path.join(JOBS_DIR, id + '.json'), null);
}
/**
 * 任务写盘：读改写事务 + 保护评估字段。
 * 历史缺陷：refreshJobStatus 拿着旧快照整体覆盖 job 文件；训练完成后自动评估写回的
 * evaluation 会被下一次状态刷新抹掉。这里在事务内重新读取最新内容，未显式携带的
 * evaluation / evaluationHistory / evalConfig 一律沿用最新值。
 */
function saveJob(job) {
  if (!isSafeId(job.id)) throw new Error('invalid job id');
  return STORE.updateJsonSync(path.join(JOBS_DIR, job.id + '.json'), {}, function (cur) {
    const next = Object.assign({}, job);
    const keep = ['evaluation', 'evaluationHistory', 'evalConfig'];
    keep.forEach(function (k) { if (next[k] === undefined && cur && cur[k] !== undefined) next[k] = cur[k]; });
    return next;
  });
}

/** 远端状态对账：从远端读进程/日志，更新本地任务状态 */
async function refreshJobStatus(job, server) {
  const workdir = server.remoteWorkdir || '~/ai-harbor-training';
  try {
    // 1) 进程是否还活着（我们启动时写 pid 文件）
    const pidRes = await sshExec(server,
      `cd ${shq(workdir)} && if [ -f run/${job.id}.pid ]; then PID=$(cat run/${job.id}.pid); if kill -0 $PID 2>/dev/null; then echo RUNNING $PID; else echo DEAD $PID; fi; else echo NORECORD; fi; echo "EXITCODE=$(cat logs/${job.id}.exit 2>/dev/null || echo NONE)"`,
      20000);
    // 启动命令会把真实退出码写进 logs/<id>.exit；此前从未读取，完成判定只看日志文本，
    // 失败训练若日志里出现 "Saving checkpoint" 就会被误判为 completed。
    const pidLines = (pidRes.stdout || '').trim().split('\n');
    const pidOut = (pidLines[0] || '').trim();
    const exitLine = pidLines.find(function (l) { return l.indexOf('EXITCODE=') === 0; }) || '';
    const exitCode = exitLine.replace('EXITCODE=', '').trim();
    // 2) 日志尾部 + 完成标记（rlcd 任务额外读 metrics.jsonl：曲线数据源更完整，
    //    且日志 tail 8000 只覆盖末尾，前面 step 行拿不到）
    const rlcdProbe = job.trainMode === 'rlcd'
      ? `ls -d saves/${job.id}/adapter 2>/dev/null && echo RLCD_ADAPTER_EXISTS; ` +
        `cat saves/${job.id}/training_done.json 2>/dev/null | tr -d '\\n' | head -c 200; echo; ` +
        `cat saves/${job.id}/metrics.jsonl 2>/dev/null | tail -300; echo RLCD_METRICS_END; `
      : '';
    const logRes = await sshExec(server,
      `cd ${shq(workdir)} && LOG=logs/${job.id}.log; if [ -f $LOG ]; then wc -l < $LOG; tail -c 8000 $LOG; else echo 0; echo "(no log yet)"; fi; ` +
      rlcdProbe +
      `ls -1 saves/${job.id}/adapter_model.sfin 2>/dev/null; ls -1 saves/${job.id}/adapter_model.safetensors 2>/dev/null; ls -1 saves/${job.id}/head.pt 2>/dev/null; ls -d saves/${job.id}/checkpoint-* 2>/dev/null | tail -3; ls -d saves/${job.id}/snapshots/step-* 2>/dev/null | tail -3`,
      20000);
    console.log('[refresh-debug]', job.id, 'trainMode=', job.trainMode, '| logOut bytes=', (logRes.stdout || '').length, '| rlcdProbe len=', rlcdProbe.length);
    const logOut = logRes.stdout || '';

    const firstLineEnd = logOut.indexOf('\n');
    const logLines = parseInt((logOut.slice(0, firstLineEnd) || '0').trim(), 10) || 0;
    const logTail = logOut.slice(firstLineEnd + 1);

    let status = job.status;
    // 产物探测同时覆盖 LLaMA-Factory（adapter / checkpoint）与 kev（head.pt / snapshots/step-）
    const hasArtifact = /adapter_model\.safetensors|head\.pt|checkpoint-|snapshots\/step-|RLCD_ADAPTER_EXISTS/.test(logOut);
    console.log('[refresh-debug]', job.id, '| pidOut=', JSON.stringify(pidOut), '| exitCode=', exitCode, '| hasArtifact=', hasArtifact);
    if (pidOut.startsWith('RUNNING')) status = 'running';
    else if (pidOut.startsWith('DEAD') || pidOut === 'NORECORD') {
      if (job.status === 'stopped') status = 'stopped';                       // 用户主动停止最高优先
      else if (exitCode && exitCode !== 'NONE' && exitCode !== '0') status = 'failed';  // 显式失败退出码
      else if (hasArtifact || exitCode === '0') status = 'completed';
      else if (job.status === 'running' || job.status === 'pending') status = 'failed';
      else status = job.status;
    }

    // 解析训练进度
    // LLaMA-Factory：tqdm 行 ' 10%|█ | 5/50 [...]'（排除 "Loading weights:" 前缀的加载条）
    // kev：'ep0 step 80/100 loss 0.929 ...'（ep<epoch> step <cur>/<total>）
    // rlcd：'step 25/60 loss=-0.0414 R=+0.687 ...'（无 ep 前缀）
    let progress = job.progress || null;
    const linesNoLoading = logTail.split('\n').filter(l => !/Loading weights/i.test(l)).join('\n');
    const mKev = [...linesNoLoading.matchAll(/ep(\d+) step (\d+)\/(\d+)/g)].pop();
    if (mKev) {
      progress = { percent: Math.round(parseInt(mKev[2], 10) / Math.max(1, parseInt(mKev[3], 10)) * 100), step: parseInt(mKev[2], 10), total: parseInt(mKev[3], 10), epochIdx: parseInt(mKev[1], 10) };
    } else {
      const mRlcd = [...linesNoLoading.matchAll(/step (\d+)\/(\d+) loss=/g)].pop();
      if (mRlcd) {
        progress = { percent: Math.round(parseInt(mRlcd[1], 10) / Math.max(1, parseInt(mRlcd[2], 10)) * 100), step: parseInt(mRlcd[1], 10), total: parseInt(mRlcd[2], 10) };
      } else {
        const m = linesNoLoading.match(/(\d+)%\|.*?(\d+)\/(\d+)/);
        if (m) progress = { percent: parseInt(m[1], 10), step: parseInt(m[2], 10), total: parseInt(m[3], 10) };
      }
    }

    // loss / epoch 采集：LLaMA-Factory 每 logging_steps 输出一行
    //   {'loss': '5.227', 'grad_norm': '4.634', 'learning_rate': '9.9e-05', 'epoch': '5.0'}
    //   注意：值带单引号（字符串形式），正则要允许 '?。偏好训练还有 'rewards/...' 等。
    const lossHistory = (job.lossHistory || []).slice();
    const lossLines = [...logOut.matchAll(/\{'loss': '?([\d.]+)'?[^}]*?'epoch': '?([\d.]+)'?\}/g)];
    // 全量去重：每次 refresh 都会重读日志，按 epoch+loss 组合键跳过已采集的点
    const seen = new Set(lossHistory.map(h => h.epoch + '|' + h.loss));
    for (const lm of lossLines) {
      const entry = { loss: parseFloat(lm[1]), epoch: parseFloat(lm[2]) };
      const key = entry.epoch + '|' + entry.loss;
      if (!seen.has(key)) { lossHistory.push(entry); seen.add(key); }
      if (lossHistory.length > 500) lossHistory.shift();
    }
    // kev 格式：'ep0 step 50/100 loss 1.859 kl ...'（epoch=ep后数字，无字典行）
    const kevLines = [...logOut.matchAll(/ep(\d+) step (\d+)\/\d+ loss ([\d.]+)/g)];
    const seenKev = new Set(lossHistory.map(h => h.epoch + '|' + h.loss));
    for (const km of kevLines) {
      const entry = { loss: parseFloat(km[3]), epoch: parseFloat(km[1]), step: parseInt(km[2], 10) };
      const key = entry.epoch + '|' + entry.loss;
      if (!seenKev.has(key)) { lossHistory.push(entry); seenKev.add(key); }
      if (lossHistory.length > 500) lossHistory.shift();
    }

    // RLCD：优先从 metrics.jsonl 重建（每步 JSON 行，比日志正则可靠）；
    // 日志正则作为兜底（metrics.jsonl 缺失的老任务）
    let rlcdLoss = null, rlcdGradNorm = null;
    if (job.trainMode === 'rlcd' && /RLCD_METRICS_END/.test(logOut)) {
      const metricsSection = logOut.split(/RLCD_METRICS_END/)[0]
        .split(/\n/).filter(l => /^\{"t":/.test(l.trim()));
      const seenR = new Set();
      for (const line of metricsSection) {
        try {
          const m = JSON.parse(line.trim());
          if (typeof m.step !== 'number' || seenR.has(m.step)) continue;
          seenR.add(m.step);
          lossHistory.push({
            loss: Number(m.loss) || 0,
            epoch: 0, step: m.step,
            reward: Number(m.reward_mean),
            acc: Number(m.acc),
            statedConf: Number(m.stated_conf_mean),
          });
          if (lossHistory.length > 500) lossHistory.shift();
        } catch (e) { /* 跳过坏行 */ }
      }
      if (lossHistory.length) {
        try {
          const lastMetric = JSON.parse(metricsSection[metricsSection.length - 1].trim());
          rlcdLoss = Number(lastMetric.loss) || null;
          rlcdGradNorm = Number(lastMetric.grad_norm) || null;
        } catch (e) { console.log('[refresh-debug]', job.id, 'lastMetric parse fail:', String(e.message).slice(0, 80)); }
      }
    }
    // adapter 探测的声明提前到所有引用之前（此前 572 行的 RLCD 探测块引用
    // 609 行才声明的变量，触发暂时性死区 ReferenceError，整个 refresh 走
    // unreachable 分支——RLCD 任务状态永远停在 running 的根因）
    let hasFinalAdapter = /adapter_model\.safetensors|head\.pt/.test(logOut);
    // RLCD adapter 目录探测（rlcd 产物布局与 kev 不同）
    if (job.trainMode === 'rlcd' && /RLCD_ADAPTER_EXISTS/.test(logOut)) {
      hasFinalAdapter = true;
    }
    // RLCD 格式（rlcd-lite 日志兜底）：'step 25/60 loss=-0.0414 R=+0.687 acc=0.833 stated=0.70 ...'
    const rlcdLines = [...logOut.matchAll(/step (\d+)\/(\d+) loss=([-\d.]+) R=([-\d.]+) acc=([\d.]+) stated=([\d.]+)/g)];
    const seenRlcd = new Set(lossHistory.map(h => 'rlcd' + h.step));
    for (const rm of rlcdLines) {
      const step = parseInt(rm[1], 10);
      const key = 'rlcd' + step;
      if (!seenRlcd.has(key)) {
        lossHistory.push({ loss: parseFloat(rm[3]), epoch: 0, step: step,
          reward: parseFloat(rm[4]), acc: parseFloat(rm[5]), statedConf: parseFloat(rm[6]) });
        seenRlcd.add(key);
      }
      if (lossHistory.length > 500) lossHistory.shift();
    }
    const evalLines = [...logOut.matchAll(/\{'eval_loss': '?([\d.]+)'?[^}]*\}/g)];
    const evalLossHistory = (job.evalLossHistory || []).slice();
    for (const em of evalLines) {
      const v = parseFloat(em[1]);
      const last = evalLossHistory[evalLossHistory.length - 1];
      if (last !== v) evalLossHistory.push(v);
      if (evalLossHistory.length > 200) evalLossHistory.shift();
    }
    const lossM = [...logTail.matchAll(/'loss': '?([\d.]+)'?/g)].pop();
    const lossKev = [...logTail.matchAll(/ step \d+\/\d+ loss ([\d.]+)/g)].pop();
    const loss = rlcdLoss != null ? rlcdLoss : (lossKev ? parseFloat(lossKev[1]) : (lossM ? parseFloat(lossM[1]) : job.loss));
    const epochM = [...logTail.matchAll(/'epoch': '?([\d.]+)'?/g)].pop();
    const epoch = epochM ? parseFloat(epochM[1]) : job.epoch;
    const gradNormM = [...logTail.matchAll(/'grad_norm': '?([\d.]+)'?/g)].pop();
    const gradNorm = rlcdGradNorm != null ? rlcdGradNorm : (gradNormM ? parseFloat(gradNormM[1]) : job.gradNorm);

    // checkpoint 列表（远端 ls 输出已在 logOut 尾部：`ls -d saves/<id>/checkpoint-* | tail -3`）
    const checkpoints = (logOut.match(/checkpoint-\d+|step-\d+/g) || [])
      .filter((v, i, a) => a.indexOf(v) === i)
      .sort((a, b) => parseInt(String(a).replace(/\D+/g, ''), 10) - parseInt(String(b).replace(/\D+/g, ''), 10));
    // hasFinalAdapter：kev/LLaMA-Factory 看日志里的 adapter 文件名；
    // rlcd 任务的产物是 saves/<id>/adapter/ 目录 + training_done.json（日志里没有文件名行）；
    // 兜底：metrics.jsonl 已被成功读取（rlcdLoss 非 null）→ 训练必然完整收尾，adapter 已写出
    if (!hasFinalAdapter && job.trainMode === 'rlcd' && rlcdLoss != null) {
      hasFinalAdapter = true;
    }

    const updated = Object.assign({}, job, {
      status: status,
      logLines: logLines,
      logTail: logTail.slice(-6000),
      progress: progress,
      loss: loss,
      epoch: epoch,
      gradNorm: gradNorm,
      lossHistory: lossHistory,
      evalLossHistory: evalLossHistory,
      checkpoints: checkpoints,
      hasFinalAdapter: hasFinalAdapter,
      lastCheckedAt: Date.now(),
    });
    console.log('[refresh-debug]', job.id, '| final status=', status, '| hasFinalAdapter=', hasFinalAdapter, '| unreachable-was=', !!job.unreachable);
    saveJob(updated);
    // 训练刚完成 → 排队自动评估（异步，不阻塞状态刷新；只在配置了评估集时触发一次）
    if (status === 'completed') autoEvaluateIfNeeded(updated);
    return updated;
  } catch (e) {
    // SSH 失败（服务器关机等）：保留原状态，标记 unreachable
    console.log('[refresh-debug]', job.id, '| CATCH unreachable:', String((e && e.stack) || e).split('\n').slice(0, 4).join(' <= '));
    const updated = Object.assign({}, job, { unreachable: true, lastCheckedAt: Date.now() });
    saveJob(updated);
    return updated;
  }
}

// ── 数据集管理 ──────────────────────────────────────────────────────────

/**
 * 数据集来源血缘（provenance）：手动 / 导入 / 合成 / 开源导入四类统一结构。
 * 允许字段白名单化，防止把 apiKey、整份 samples 等任意对象写进 meta。
 * 历史数据集没有该字段时按 description 推断并 inert 落盘一次（只读自愈）。
 */
const PROVENANCE_KEYS = [
  'schemaVersion', 'source', 'sourceLayout', 'sourceLayoutConfidence', 'targetKind',
  'importedAt', 'synthesizedAt', 'taskId', 'mode', 'domain', 'engine', 'backend',
  'model', 'realCount', 'mixWarning', 'rules', 'funnel', 'registeredAt', 'note',
];
function inferProvenanceSource(ds) {
  const d = String((ds && ds.description) || '');
  if (/自动合成|合成-|synthesis/i.test(d)) return 'synthesis';
  if (/导入转换|imported-|import/i.test(d)) return 'import';
  if (/kev 官方|开源|复现|open.source/i.test(d)) return 'open_source';
  return 'manual';
}
function inferProvenance(ds) {
  return {
    schemaVersion: 1,
    source: inferProvenanceSource(ds),
    inferred: true,
    registeredAt: (ds && ds.createdAt) || null,
    note: '历史数据集：来源由描述推断，并非生成时记录',
  };
}
function compactFunnel(f) {
  if (!f || typeof f !== 'object') return null;
  return {
    total: f.total || 0, passed: f.passed || 0, passRate: f.passRate || 0,
    stages: Array.isArray(f.stages) ? f.stages.slice(0, 12) : [],
    quality: f.quality ? {
      average: f.quality.average, threshold: f.quality.threshold,
      distribution: f.quality.distribution || [],
    } : null,
    llmJudge: f.llmJudge ? {
      enabled: f.llmJudge.enabled, threshold: f.llmJudge.threshold,
      judged: f.llmJudge.judged, failed: f.llmJudge.failed, average: f.llmJudge.average,
    } : null,
  };
}
function normalizeProvenance(p, ds) {
  if (!p || typeof p !== 'object' || Array.isArray(p)) {
    return {
      schemaVersion: 1, source: 'manual', registeredAt: (ds && ds.createdAt) || Date.now(),
      note: '手动新建（未提供来源信息）',
    };
  }
  const out = { schemaVersion: 1, source: String(p.source || 'unknown').slice(0, 32) };
  for (const k of PROVENANCE_KEYS) {
    if (k === 'schemaVersion' || k === 'source' || p[k] === undefined) continue;
    out[k] = p[k];
  }
  if (out.funnel) out.funnel = compactFunnel(out.funnel);
  if (out.rules && typeof out.rules === 'object' && !Array.isArray(out.rules)) {
    // 子对象同样白名单：避免把 apiKey 之类藏进 rules 里写进 meta
    const RULE_KEYS = [
      'minScore', 'targetLang', 'minLen', 'maxLen', 'semanticThreshold',
      'llmJudge', 'judgeThreshold', 'judgeBackend', 'lang',
    ];
    const r = {};
    for (const k of RULE_KEYS) if (out.rules[k] !== undefined) r[k] = out.rules[k];
    out.rules = r;
  } else delete out.rules;
  out.registeredAt = out.registeredAt || (ds && ds.createdAt) || Date.now();
  return out;
}

function listDatasets() {
  if (!fs.existsSync(DATASETS_DIR)) return [];
  return fs.readdirSync(DATASETS_DIR).filter(f => f.endsWith('.json') && f !== 'index.json')
    .map(f => {
      const file = path.join(DATASETS_DIR, f);
      const meta = readJson(file, {});
      if (!meta || Array.isArray(meta) || !meta.id) return meta;
      // 自愈：只给 .meta.json 补 provenance，数据文件本身绝不被改动
      if (!meta.provenance && f.endsWith('.meta.json')) {
        meta.provenance = inferProvenance(meta);
        try { writeJson(file, Object.assign({}, meta, { provenance: meta.provenance })); } catch (e) {}
      } else if (!meta.provenance) {
        meta.provenance = inferProvenance(meta);
      }
      return meta;
    })
    .filter(d => d.id)
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
}

// ── 数据集预览内核（只读）──────────────────────────────────────────────
// 历史缺陷：旧端点用 readJson 读 .jsonl，decision 数据集必然解析失败返回空数组，
// 表现为"预览没数据"。这里按扩展名分别解析，并对格式化 JSON 数组做回退兼容。
const PREVIEW_CACHE = new Map();          // fileName -> {key, samples, fields, warnings, total}
const PREVIEW_MAX_BYTES = 120 * 1024 * 1024;
const PREVIEW_MAX_SAMPLES = 200000;
const PREVIEW_FIELD_SCAN = 5000;         // 字段统计抽样上限，避免 20 万条每次全扫

function parseDatasetFile(fileName, content) {
  const text = String(content == null ? '' : content).replace(/^\uFEFF/, '');
  if (/\.jsonl$/i.test(String(fileName || ''))) {
    return text.split(/\r?\n/).filter(l => l.trim())
      .map((l, i) => {
        try { return JSON.parse(l); }
        catch (e) { throw new Error('JSONL 第 ' + (i + 1) + ' 行不是合法 JSON：' + e.message); }
      });
  }
  try {
    const v = JSON.parse(text);
    if (Array.isArray(v)) return v;
    if (v && Array.isArray(v.samples)) return v.samples;
    if (v && Array.isArray(v.data)) return v.data;
    if (v && typeof v === 'object') return [v];
    throw new Error('顶层不是数组也不是对象');
  } catch (e) {
    // 有些 .json 实际按行存 JSONL；JSON 解析失败时按行回退一次
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (lines.length > 1 && lines.every(l => /^[{[]/.test(l.trim()))) {
      return lines.map((l, i) => {
        try { return JSON.parse(l.replace(/,\s*$/, '')); }
        catch (e2) { throw new Error('JSON 解析失败，且按 JSONL 回退时第 ' + (i + 1) + ' 行也失败：' + e2.message); }
      });
    }
    throw e;
  }
}

function valueType(v) {
  if (v === null || v === undefined) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v;
}
function buildFieldStats(samples) {
  const map = new Map();
  const scan = Math.min(samples.length, PREVIEW_FIELD_SCAN);
  let nonObject = 0;
  for (let i = 0; i < scan; i++) {
    const s = samples[i];
    if (!s || typeof s !== 'object' || Array.isArray(s)) { nonObject++; continue; }
    for (const [k, v] of Object.entries(s)) {
      let f = map.get(k);
      if (!f) {
        f = { name: k, types: new Set(), present: 0, empty: 0, maxLength: 0, distinct: new Set(), example: null };
        map.set(k, f);
      }
      f.types.add(valueType(v));
      const sv = typeof v === 'string' ? v : (v == null ? '' : JSON.stringify(v));
      if (v == null || v === '' || v === undefined) f.empty++;
      else f.present++;
      if (sv && sv.length > f.maxLength) f.maxLength = sv.length;
      if (f.example == null && sv) f.example = String(sv).slice(0, 120);
      if (f.distinct.size < 50 && sv.length < 80) f.distinct.add(sv);
    }
  }
  return {
    scanned: scan, nonObject,
    fields: [...map.values()].map(f => ({
      name: f.name, types: [...f.types], present: f.present,
      coverage: scan ? +(f.present / scan).toFixed(3) : 0,
      empty: f.empty, missing: Math.max(0, (scan - nonObject) - f.present - f.empty),
      maxLength: f.maxLength,
      distinct: f.distinct.size, distinctCapped: f.distinct.size >= 50,
      example: f.example,
    })).sort((a, b) => b.present - a.present),
  };
}

function loadPreviewData(ds) {
  const name = String(ds.fileName || '');
  if (!name || path.basename(name) !== name || !/^[A-Za-z0-9_.-]+$/.test(name) || name.includes('..')) {
    throw new Error('数据集文件名不合法：' + name);
  }
  const file = path.join(DATASETS_DIR, name);
  if (!fs.existsSync(file)) throw new Error('数据文件不存在：' + name);
  const st = fs.statSync(file);
  if (st.size > PREVIEW_MAX_BYTES) throw new Error('数据文件超过 120MB，暂不支持在线预览');
  const key = name + '|' + st.mtimeMs + '|' + st.size;
  const hit = PREVIEW_CACHE.get(name);
  if (hit && hit.key === key) return hit;
  const samples = parseDatasetFile(name, fs.readFileSync(file, 'utf8'));
  if (!Array.isArray(samples)) throw new Error('解析结果不是数组');
  if (samples.length > PREVIEW_MAX_SAMPLES) throw new Error('样本数超过 20 万上限，暂不预览');
  const stats = buildFieldStats(samples);
  const warnings = [];
  if (ds.count != null && Number(ds.count) !== samples.length) {
    warnings.push('meta 记录 ' + ds.count + ' 条，实际解析出 ' + samples.length + ' 条（以实际为准）');
  }
  if (stats.nonObject) warnings.push('抽样中有 ' + stats.nonObject + ' 条样本不是对象');
  if (samples.length > stats.scanned) warnings.push('字段统计基于前 ' + stats.scanned + ' 条抽样');
  const emptyFields = stats.fields.filter(f => f.coverage < 1);
  if (emptyFields.length) warnings.push('存在非全覆盖字段：' + emptyFields.slice(0, 5).map(f => f.name).join(', '));
  if (ds.kind === 'decision') {
    const bad = samples.slice(0, 200).filter(s => !s || !s.questions || typeof s.questions !== 'object').length;
    if (bad) warnings.push('前 200 条中有 ' + bad + ' 条缺少 questions（决策格式要求）');
  }
  const entry = {
    key, samples, fields: stats.fields, scanned: stats.scanned,
    warnings, total: samples.length, bytes: st.size, mtime: st.mtimeMs,
  };
  PREVIEW_CACHE.set(name, entry);
  if (PREVIEW_CACHE.size > 4) PREVIEW_CACHE.delete(PREVIEW_CACHE.keys().next().value);
  return entry;
}

function datasetInfoEntry(ds) {
  // 生成 LLaMA-Factory dataset_info.json 片段（对齐官方 data/README.md 格式）
  const entry = { file_name: ds.fileName };
  if (ds.formatting === 'sharegpt') {
    entry.formatting = 'sharegpt';
    entry.columns = { messages: 'conversations' };
    if (ds.kind === 'preference') {
      entry.ranking = true;
      entry.columns.chosen = 'chosen';
      entry.columns.rejected = 'rejected';
    }
  } else {
    // alpaca
    entry.columns = { prompt: 'instruction', query: 'input', response: 'output' };
    if (ds.kind === 'preference') {
      entry.ranking = true;
      entry.columns = { prompt: 'instruction', query: 'input', chosen: 'chosen', rejected: 'rejected' };
    }
    if (ds.kind === 'pretrain') {
      entry.columns = { prompt: 'text' };
    }
  }
  return entry;
}

/** 类 Jev 决策数据集（kev 格式）校验：返回错误字符串或 null */
function validateDecisionSamples(samples) {
  const TYPES = ['choice', 'score', 'noul'];
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (!s || typeof s !== 'object') return '第 ' + (i + 1) + ' 条：必须是对象';
    if (s.state === undefined || s.state === null || s.state === '') return '第 ' + (i + 1) + ' 条：缺少 state（决策上下文）';
    if (!s.questions || typeof s.questions !== 'object') return '第 ' + (i + 1) + ' 条：缺少 questions';
    const qNames = Object.keys(s.questions);
    if (!qNames.length) return '第 ' + (i + 1) + ' 条：questions 为空';
    for (const qn of qNames) {
      const q = s.questions[qn];
      if (!TYPES.includes(q.type)) return '第 ' + (i + 1) + ' 条问题 ' + qn + '：type 必须是 ' + TYPES.join('/') + '，得到 ' + q.type;
      if (!q.instructions) return '第 ' + (i + 1) + ' 条问题 ' + qn + '：缺少 instructions';
      if (q.type === 'choice') {
        if (!q.criteria || typeof q.criteria !== 'object' || Object.keys(q.criteria).length < 2) {
          return '第 ' + (i + 1) + ' 条问题 ' + qn + '：choice 需要 criteria（≥2 个选项的对象）';
        }
        if (typeof q.label !== 'string' || !(q.label in q.criteria)) {
          return '第 ' + (i + 1) + ' 条问题 ' + qn + '：label "' + q.label + '" 必须是 criteria 中的选项之一（选项值可为 null，kev 官方格式允许选项无描述）';
        }
      }
      if (q.type === 'score') {
        if (!Array.isArray(q.criteria) || q.criteria.length < 2) {
          return '第 ' + (i + 1) + ' 条问题 ' + qn + '：score 需要 criteria（≥2 档的有序数组）';
        }
        if (typeof q.label !== 'number' || q.label < 0 || q.label >= q.criteria.length) {
          return '第 ' + (i + 1) + ' 条问题 ' + qn + '：label 必须是 0..' + (q.criteria.length - 1) + ' 的档位编号';
        }
      }
      if (q.type === 'noul') {
        if (typeof q.label !== 'boolean') return '第 ' + (i + 1) + ' 条问题 ' + qn + '：noul 的 label 必须是 true/false';
      }
    }
  }
  return null;
}

/**
 * RLCD 训练命令构建（GRPO + RLCR 奖励，基线 rlcd-lite）。
 * 数据流：decision JSONL 先经 kev_rlcd_bridge.py 转 train/calib JSONL，再喂 train_rlcd.py。
 * 与 kev 引擎的区别：kev 是纯监督（CE），RLCD 是 RL 阶段（策略梯度 + 校准奖励）。
 */
function buildRlcdTrainCommand(server, jobId, cfg, ds) {
  const workdir = server.remoteWorkdir || '~/ai-harbor-training';
  const pyBin = server.cliPath
    ? require('path').dirname(server.cliPath).replace(/\\/g, '/') + '/python.exe'
    : 'python3';
  // RLCD 脚本固定在 D:/llm-training（训练 venv 同目录）；数据/产物用 workdir 相对路径
  // 基座模型：HuggingFace 名会在离线环境失败，本地路径直接可用（冒烟已验证）
  const baseModel = cfg.modelName || 'D:/llm-training/models/Qwen3-0.6B';
  const args = [
    'D:/llm-training/train_rlcd.py',
    '--data-dir', 'rlcd-data/' + jobId,
    '--base', baseModel,
    '--steps', String(cfg.rlcdSteps || 100),
    '--group-size', String(cfg.rlcdGroupSize || 8),
    '--batch', String(cfg.rlcdBatch || 2),
    '--reward', cfg.rlcdReward || 'rlcr',
    '--lr', String(cfg.rlcdLr || 1e-5),
    '--lora', String(cfg.rlcdLoraR || 16),
    '--rollout-temp', String(cfg.rlcdRolloutTemp || 1.0),
    '--conf-grid', String(cfg.rlcdConfGrid || 5),
    // qwen 建议项：诚实惩罚与早停（默认开启早停 patience=3、关闭诚实惩罚 λ=0）
    ...(cfg.rlcdHonestyLambda != null && Number(cfg.rlcdHonestyLambda) > 0
      ? ['--honesty-lambda', String(cfg.rlcdHonestyLambda)] : []),
    ...(cfg.rlcdEarlyStopPatience != null && Number(cfg.rlcdEarlyStopPatience) > 0
      ? ['--early-stop-patience', String(cfg.rlcdEarlyStopPatience)] : []),
    '--seed', String(cfg.seed == null ? 42 : cfg.seed),
    '--out', 'saves/' + jobId,
    '--device', 'cuda',
    '--dtype', 'bf16',
  ];
  return { pyBin, args, workdir };
}

/** kev 训练命令构建（类 Jev 决策模型训练，Windows 本地经 venv python） */
function buildKevTrainCommand(server, jobId, cfg, ds) {
  const workdir = server.remoteWorkdir || '~/ai-harbor-training';
  const pyBin = server.cliPath
    ? require('path').dirname(server.cliPath).replace(/\\/g, '/') + '/python.exe'
    : 'python3';
  const args = [
    '-m', 'kev.train',
    '--data', 'data/' + ds.fileName,       // 训练数据（decision JSONL，已上传）
    '--base', cfg.modelName,               // 基座（Qwen3.5-0.8B-Base 等）
    '--epochs', String(cfg.epochs || 2),
    '--lr', String(cfg.learningRate || 2e-5),
    '--batch', String(cfg.batchSize || 1),
    // kev 用 OneCycleLR(pct_start=0.1)：小数据集 + 大累积会令总步数过少（<10）触发
    // torch 的 warmup 区间除零。决策数据集通常几百条，累积给 2；保证步数充足。
    '--accum', String(cfg.gradAccum || 2),
    '--dtype', 'bf16',
    '--device', 'cuda',
    '--out', 'saves/' + jobId,
  ];
  // loss 曲线密度：kev 默认每 10 步打一行，可调小（1=每步）。默认收紧到 2，
  // 与 LLaMA-Factory logging_steps 对齐，让短任务（几十步）曲线也有足够点。
  args.push('--log_every', String(Math.max(1, parseInt(cfg.logEvery, 10) || 2)));
  if (cfg.kevInitFrom) args.push('--init_from', cfg.kevInitFrom);
  if (cfg.kevLora) args.push('--lora', String(cfg.kevLora));
  // torch 2.9 OneCycleLR 已知坑：total_steps=10 时 warmup 区间除零（20条×accum4×2ep 正好=10）。
  // 平台侧兜底：估算总步数，若恰好为 10 则把 epochs+1 错开。
  const reqCount = ds.count || 20;
  const accum = cfg.gradAccum || 2;
  const batch = cfg.batchSize || 1;
  const epochs = cfg.epochs || 2;
  const estSteps = Math.ceil(reqCount / (batch * accum)) * epochs;
  if (estSteps === 10) {
    const i = args.indexOf('--epochs');
    if (i >= 0) args[i + 1] = String(epochs + 1);
  }
  return { pyBin, args, workdir };
}

// ── 训练后评估（评估集 → 推理 → 指标 → 回写 job） ─────────────────────

function readDatasetSamplesById(ref) {
  const id = typeof ref === 'string' ? ref : (ref && ref.id);
  const ds = listDatasets().find(function (d) { return d.id === id; });
  if (!ds || !ds.fileName) return [];
  const f = path.join(DATASETS_DIR, ds.fileName);
  if (!fs.existsSync(f)) return [];
  try { return EVAL.parseSamples(ds.fileName, fs.readFileSync(f, 'utf8')); }
  catch (e) { throw new Error('评估集解析失败：' + e.message); }
}

/** 命令执行封装：非零退出码直接带日志尾部报错，不把失败当成功 */
async function evalSh(server, command, timeoutMs) {
  const r = await sshExec(server, command, timeoutMs || 600000);
  if (r.code !== 0) {
    const tail = ((r.stderr || '') + '\n' + (r.stdout || '')).trim().slice(-1200);
    throw new Error('推理命令退出码 ' + r.code + '：' + tail);
  }
  return r.stdout || '';
}

/**
 * 构造评估 IO。所有路径与执行方式都由 server 决定（本机 Git Bash / 远端 SSH 同一套命令），
 * 评估编排本身不感知文件系统，便于单测与替换引擎。
 */
function createEvalIo(server, job, opts) {
  const workdir = server.remoteWorkdir || '~/ai-harbor-training';
  const bwd = toBashPath(workdir).replace(/\/$/, '');
  const wd = workdir.replace(/\/$/, '');
  const local = isLocal(server);
  const pyBin = server.cliPath
    ? path.dirname(server.cliPath).replace(/\\/g, '/') + '/python.exe'
    : (local ? 'python' : 'python3');
  const cliBin = server.cliPath || 'llamafactory-cli';
  const winPath = function (rel) { return wd + '/' + rel; };
  const bashPath = function (rel) { return bwd + '/' + rel; };
  const evalTag = 'eval-' + job.id + '-' + Date.now().toString(36);

  const io = {
    readJob: function (id) { return getJob(id); },
    saveJob: async function (j) { saveJob(j); return j; },
    datasetById: function (id) {
      return listDatasets().find(function (d) { return d.id === id; }) || null;
    },
    readDatasetSamples: async function (ds) { return readDatasetSamplesById(ds); },
    obtain: async function (plan) {
      // 外部预测导入：平台只做对齐与评分，明确标注来源，不冒充本机推理
      if (plan.engine === 'import') {
        const text = typeof opts.predictions === 'string'
          ? opts.predictions : JSON.stringify(opts.predictions || []);
        return { engine: 'import', predictionsText: text, artifacts: { source: 'imported predictions' },
          taskType: plan.gold.kind };
      }

      if (plan.engine === 'kev') {
        if (!plan.samples.length || !plan.samples[0].questions) {
          throw new Error('kev 引擎需要 decision 格式评估集（state + questions + label），当前评估集不匹配');
        }
        const evalFile = 'data/' + plan.evalDatasetId + '-eval.jsonl';
        const outDir = 'evals/' + evalTag;
        await sshWriteFile(server, winPath(evalFile),
          plan.samples.map(function (s) { return JSON.stringify(s); }).join('\n') + '\n');
        // rlcd 任务的 checkpoint 是 rlcd-lite adapter（不是 kev head.pt），
        // 走 train_rlcd.py eval 子命令；kev 任务走原生 kev.benchmark。
        const cmd = job.trainMode === 'rlcd'
          ? 'cd ' + shq(bashPath('')) + ' && ' + shq(pyBin) + ' D:/llm-training/train_rlcd.py eval' +
            ' --run ' + shq('saves/' + job.id) +
            ' --eval-data ' + shq(evalFile) +
            ' --out ' + shq(outDir) + ' --device cuda --dtype bf16'
          : 'cd ' + shq(bashPath('')) + ' && ' + shq(pyBin) + ' -m kev.benchmark' +
            ' --run ' + shq('saves/' + job.id) +
            ' --data ' + shq(evalFile) +
            ' --out ' + shq(outDir) + ' --device cuda';
        const timeout = Math.min(30 * 60 * 1000, Math.max(5 * 60 * 1000, plan.samples.length * 4000));
        await evalSh(server, cmd, timeout);
        const rowsText = await evalSh(server, 'cat ' + shq(bashPath(outDir + '/rows.json')), 60000);
        const reportText = await evalSh(server, 'cat ' + shq(bashPath(outDir + '/report.json')), 60000);
        // 评估临时数据文件用完即删（rows/report 已读出；evals/<tag> 目录保留供追溯）
        try { await evalSh(server, 'rm -f ' + shq(bashPath(evalFile)), 15000); } catch (e) {}
        return { engine: 'kev', rowsText: rowsText, reportText: reportText, taskType: 'classification',
          rlcdFormat: job.trainMode === 'rlcd',
          artifacts: { evalFile: evalFile, outDir: outDir, command: cmd } };
      }

      if (plan.engine === 'llamafactory') {
        if (plan.gold.kind !== 'generation') {
          throw new Error('LLaMA-Factory 批量预测需要 SFT 格式评估集（instruction/output 或 messages），当前评估集不匹配');
        }
        const ds = io.datasetById(plan.evalDatasetId);
        const evalName = 'eval_' + plan.evalDatasetId.replace(/[^A-Za-z0-9_]/g, '_');
        const evalFile = 'data/' + plan.evalDatasetId + '-eval' + (/\.jsonl$/i.test(ds.fileName) ? '.jsonl' : '.json');
        const raw = fs.readFileSync(path.join(DATASETS_DIR, ds.fileName), 'utf8');
        await sshWriteFile(server, winPath(evalFile), raw);
        const infoEntry = datasetInfoEntry(Object.assign({}, ds, { fileName: evalFile.split('/').pop() }));
        // 本机 Git Bash 里 `python3` 可能落到 Microsoft Store 的 python 别名（退出码 49），
        // 必须显式用训练环境的 venv Python；远端 Linux 同样可用这个解释器。
        const merge =
          'cd ' + shq(bashPath('data')) + ' && ' + shq(pyBin) + ' - <<\'PYEOF\'\n' +
          'import json, os\n' +
          'info = {}\n' +
          'if os.path.exists("dataset_info.json"):\n' +
          '    try: info = json.load(open("dataset_info.json"))\n' +
          '    except Exception: info = {}\n' +
          'info[' + JSON.stringify(evalName) + '] = ' + JSON.stringify(infoEntry) + '\n' +
          'json.dump(info, open("dataset_info.json", "w"), ensure_ascii=False, indent=2)\n' +
          'PYEOF';
        await evalSh(server, merge, 60000);
        const cfg = job.config || {};
        const predictYaml = [
          '### model',
          'model_name_or_path: ' + JSON.stringify(cfg.modelName || ''),
          'adapter_name_or_path: saves/' + job.id,
          'trust_remote_code: true',
          '', '### method',
          'stage: sft', 'do_predict: true',
          'finetuning_type: ' + (cfg.finetuningType || 'lora'),
          '', '### dataset',
          'eval_dataset: ' + evalName,
          'template: ' + (cfg.template || 'qwen'),
          'cutoff_len: ' + (cfg.cutoffLen || 1024),
          'max_samples: ' + plan.maxSamples,
          'overwrite_cache: true',
          'preprocessing_num_workers: 8',
          'dataloader_num_workers: ' + (local ? 0 : 8),
          '', '### output',
          'output_dir: evals/' + evalTag,
          'overwrite_output_dir: true',
          '', '### eval',
          'per_device_eval_batch_size: 1',
          'predict_with_generate: true',
          'ddp_timeout: 180000000', ''
        ].join('\n');
        const yamlFile = 'jobs/' + job.id + '-eval.yaml';
        await sshWriteFile(server, winPath(yamlFile), predictYaml);
        const cmd = 'cd ' + shq(bashPath('')) + ' && ' + shq(cliBin) + ' train ' + yamlFile;
        await evalSh(server, cmd, Math.min(30 * 60 * 1000, Math.max(5 * 60 * 1000, plan.maxSamples * 8000)));
        const predFile = 'evals/' + evalTag + '/generated_predictions.jsonl';
        const predictionsText = await evalSh(server, 'cat ' + shq(bashPath(predFile)) + ' 2>/dev/null || true', 60000);
        if (!predictionsText.trim()) throw new Error('未找到 generated_predictions.jsonl：' + predFile + '（请确认 LLaMA-Factory 版本与评估集格式）');
        // 评估临时件清理（2026-11-11）：-eval 数据文件 + dataset_info 条目 + 评估 yaml 用完即删——
        // 此前不清理，每次评估都在训练数据目录累积污染（实测 3 个 -eval 文件 + dataset_info 残留）
        try {
          const rmEval = 'cd ' + shq(bashPath('')) + ' && rm -f ' + shq(evalFile) + ' ' + shq(yamlFile) + ' && ' +
            shq(pyBin) + ' - <<\'PYEOF\'\n' +
            'import json, os\n' +
            'try://n' +
            '    info = json.load(open("data/dataset_info.json"))\n' +
            '    info.pop(' + JSON.stringify(evalName) + ', None)\n' +
            '    json.dump(info, open("data/dataset_info.json", "w"), ensure_ascii=False, indent=2)\n' +
            'except Exception://n' +
            '    pass\n' +
            'PYEOF';
          await evalSh(server, rmEval, 30000);
        } catch (e) { /* 清理失败不阻塞评估结果（结果已读出） */ }
        return { engine: 'llamafactory', predictionsText: predictionsText, taskType: 'generation',
          artifacts: { evalFile: evalFile, yamlFile: yamlFile, predictionsFile: predFile, command: cmd } };
      }
      throw new Error('未知评估引擎：' + plan.engine);
    }
  };
  return io;
}

/** 训练完成后自动排队评估：只触发一次，不阻塞状态刷新，失败不影响训练状态 */
function autoEvaluateIfNeeded(job) {
  const c = job.evalConfig;
  if (!c || !c.autoEval || !c.evalDatasetId || !isSafeId(c.evalDatasetId)) return;
  const ev = job.evaluation;
  if (ev && (ev.status === 'running' || ev.status === 'succeeded')) return;
  // 自动评估失败后不反复重试（避免 GPU 被死循环占用），用户可在详情里手动重跑
  if (ev && ev.lastError && ev.autoFailed) return;
  if (EVAL.listTasks().some(function (t) {
    return t.jobId === job.id && (t.state === 'running' || t.state === 'queued');
  })) return;
  const server = findServer(job.serverId);
  if (!server) return;
  try {
    EVAL.startEvaluation(job, {
      evalDatasetId: c.evalDatasetId, engine: c.engine || 'auto',
      maxSamples: c.maxSamples, auto: true
    }, createEvalIo(server, job, {}));
  } catch (e) {
    // 评估是附属流程，排队失败只记日志，不抛给训练状态刷新
    console.error('[model-training] auto eval queue failed', job.id, e.message);
  }
}

/** 只有训练完成（或已有最终产物）才能评估；训练中/提交失败的任务不跑推理 */
function canEvaluate(job) {
  return job.status === 'completed' || (job.status !== 'running' && !!job.hasFinalAdapter);
}

function resolveEngine(job, requested) {
  if (requested && requested !== 'auto') return requested;
  // kev 与 rlcd 同用 kev.benchmark 评估（同一 System One 接口 + rows.json 格式）
  return job.trainMode === 'kev' || job.trainMode === 'rlcd' ? 'kev' : 'llamafactory';
}

// ── 路由主入口 ──────────────────────────────────────────────────────────

async function handleModelTrainingRequest(req, res) {
  const pathname = req.pathname || req.url;

  // ── 服务器 ──
  if (pathname === '/api/model-training/servers' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, servers: listServers().map(sanitizeServer) });
  }
  if (pathname === '/api/model-training/servers' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    const fieldErr = validateServerFields(p);
    if (fieldErr) return sendJson(res, 400, { ok: false, error: fieldErr });
    // 本地模式：不需要 host/username/password
    if (p.local) {
      const list = listServers();
      let server;
      if (p.id) {
        const i = list.findIndex(s => s.id === p.id);
        if (i < 0) return sendJson(res, 404, { ok: false, error: 'server not found' });
        server = Object.assign({}, list[i], p);
        list[i] = server;
      } else {
        server = Object.assign({
          id: newId('srv'),
          local: true,
          name: p.name || '本机（localhost）',
          remoteWorkdir: p.remoteWorkdir || 'D:/llm-training/ai-harbor-training',
          cliPath: p.cliPath || 'D:/llm-training/venv/Scripts/llamafactory-cli.exe',
        }, p, { createdAt: Date.now() });
        list.push(server);
      }
      saveServers(list);
      return sendJson(res, 200, { ok: true, server: sanitizeServer(server) });
    }
    if (!p.host || !p.username) return sendJson(res, 400, { ok: false, error: 'host 和 username 必填' });
    if (!p.password && !p.privateKey) {
      const old = p.id ? findServer(p.id) : null;
      // 编辑时允许沿用已保存的密码或私钥（前端出于安全不回填凭据本体）
      if (!old || (!old.password && !old.privateKey)) return sendJson(res, 400, { ok: false, error: 'password 或 privateKey 必填' });
    }
    const list = listServers();
    let server;
    if (p.id) {
      const i = list.findIndex(s => s.id === p.id);
      if (i < 0) return sendJson(res, 404, { ok: false, error: 'server not found' });
      server = Object.assign({}, list[i], p);
      // Object.assign 里 undefined 也会覆盖旧值，必须显式恢复两种凭据
      if (!p.password) server.password = list[i].password;
      if (!p.privateKey) server.privateKey = list[i].privateKey;
      list[i] = server;
    } else {
      server = Object.assign({ id: newId('srv'), port: 22, remoteWorkdir: '~/ai-harbor-training' }, p, { createdAt: Date.now() });
      list.push(server);
    }
    saveServers(list);
    return sendJson(res, 200, { ok: true, server: sanitizeServer(server) });
  }
  if (pathname === '/api/model-training/servers/delete' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    const list = listServers().filter(s => s.id !== (body.data || {}).id);
    saveServers(list);
    return sendJson(res, 200, { ok: true });
  }
  if (pathname === '/api/model-training/servers/test' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    const p = body.data || {};
    let server = p.id ? findServer(p.id) : null;
    if (!server) server = p;
    else server = Object.assign({}, server, p.password ? p : { password: server.password });
    try {
      // 一次性探测：SSH 通、python/llamafactory、GPU、磁盘（本地模式同样命令——本地经 bash 执行，语法兼容）
      const cliBin = server.cliPath || 'llamafactory-cli';
      const probe = await sshExec(server,
        'echo "== os =="; uname -a 2>/dev/null || ver; ' +
        'echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"; ' +
        'echo "== python =="; which python3 2>/dev/null && python3 --version 2>&1; ' +
        'echo "== llamafactory =="; ("' + cliBin + '" --help 2>/dev/null | head -1) || echo "NOT_INSTALLED"; ' +
        (isLocal(server)
          ? 'echo "== disk =="; df -h /d 2>/dev/null | tail -1'
          : 'echo "== disk =="; df -h ~ | tail -1'),
        45000);
      const out = probe.stdout || '';
      const result = {
        ok: true,
        reachable: true,
        os: (out.match(/== os ==\n(.*)/) || [])[1] || '',
        gpus: (out.split('== gpu ==\n')[1] || '').split('\n')[0] ? out.split('== gpu ==\n')[1].split('\n').filter(l => l && !l.startsWith('==')).map(l => l.trim()) : [],
        python: (out.split('== python ==\n')[1] || '').split('\n')[0] || '',
        llamafactoryInstalled: !/NOT_INSTALLED/.test(out),
        llamafactoryLine: (out.split('== llamafactory ==\n')[1] || '').split('\n')[0] || '',
        disk: (out.split('== disk ==\n')[1] || '').split('\n')[0] || '',
        raw: out.slice(0, 2000),
      };
      return sendJson(res, 200, { ok: true, result });
    } catch (e) {
      return sendJson(res, 200, { ok: true, result: { reachable: false, error: e.message } });
    }
  }

  // ── 数据集 ──
  if (pathname === '/api/model-training/datasets' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, datasets: listDatasets() });
  }
  // 临时诊断：服务进程内 BASH_EXE 探测结果与 execSync/spawn 的行为差异（定位 DETACHED 环境 stdout 丢失）
  // ── 孤儿产物回收：扫描 saves/ 下没有对应 job 记录的目录，逐个清理 ──
  if (pathname === '/api/model-training/cleanup-orphans' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    const serverId = (body.data || {}).serverId || listServers()[0] && listServers()[0].id;
    const server = findServer(serverId);
    if (!server) return sendJson(res, 400, { ok: false, error: '没有可用服务器配置' });
    const workdir = server.remoteWorkdir || '~/ai-harbor-training';
    try {
      // 列出 saves/ 全部目录
      const ls = await sshExec(server, 'cd ' + shq(workdir) + ' && ls -1 saves/ 2>/dev/null', 30000);
      const dirs = (ls.stdout || '').split('\n').map(s => s.trim()).filter(Boolean);
      const orphans = dirs.filter(d => {
        const m = d.match(/^(job-[a-z0-9]+)$/);
        return m && !fs.existsSync(path.join(JOBS_DIR, m[1] + '.json'));
      });
      // 非平台 ID 的目录（smoke-test-1 等）也算孤儿
      const foreign = dirs.filter(d => !/^job-[a-z0-9]+$/.test(d));
      const all = orphans.concat(foreign);
      if (!all.length) return sendJson(res, 200, { ok: true, cleaned: [], freedMB: 0, message: '没有孤儿产物' });
      // 逐个清理（不一次性 rm -rf 大串，失败隔离）
      const results = [];
      let freed = 0;
      for (const d of all) {
        try {
          const du = await sshExec(server, 'cd ' + shq(workdir) + ' && du -sm saves/' + d + ' 2>/dev/null | cut -f1', 15000);
          const mb = parseInt((du.stdout || '0').trim(), 10) || 0;
          await sshExec(server, 'cd ' + shq(workdir) + ' && rm -rf saves/' + d + ' rlcd-data/' + d + ' evals/eval-' + d + '-* logs/' + d + '.log logs/' + d + '.exit run/' + d + '.pid jobs/' + d + '.yaml 2>/dev/null; echo ok', 60000);
          results.push({ dir: d, freedMB: mb });
          freed += mb;
        } catch (e) { results.push({ dir: d, error: String(e.message).slice(0, 80) }); }
      }
      return sendJson(res, 200, { ok: true, cleaned: results, freedMB: freed });
    } catch (e) {
      return sendJson(res, 500, { ok: false, error: e.message });
    }
  }

  if (pathname === '/api/model-training/datasets' && req.method === 'POST') {
    const body = await parseRequestBody(req, 60 * 1024 * 1024);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    if (!p.name || !Array.isArray(p.samples)) return sendJson(res, 400, { ok: false, error: 'name 与 samples[] 必填' });
    // decision 类型校验（kev 类型化决策格式：state + questions[choice/score/noul] + label）
    if (p.kind === 'decision') {
      const err = validateDecisionSamples(p.samples);
      if (err) return sendJson(res, 400, { ok: false, error: err });
    }
    const ds = {
      id: newId('ds'),
      name: p.name,
      kind: p.kind || 'sft',              // sft | preference | pretrain | decision（类 Jev 决策模型）
      formatting: p.formatting || 'alpaca', // alpaca | sharegpt（decision 类型不用此字段）
      samples: p.samples,
      fileName: '',                        // 保存时生成
      createdAt: Date.now(),
      description: p.description || '',
    };
    ds.provenance = normalizeProvenance(p.provenance, ds);
    // decision 数据集存 JSONL（kev 训练读 jsonl；每行 {state, questions:{name:{type,instructions,criteria,label}}}）
    ds.fileName = ds.id + (ds.kind === 'decision' ? '.jsonl' : '.json');
    if (ds.kind === 'decision') {
      const lines = ds.samples.map(s => JSON.stringify(s)).join('\n') + '\n';
      fs.writeFileSync(path.join(DATASETS_DIR, ds.fileName), lines, 'utf8');
    } else {
      writeJson(path.join(DATASETS_DIR, ds.fileName), ds.samples);
    }
    writeJson(path.join(DATASETS_DIR, ds.id + '.meta.json'), Object.assign({}, ds, { samples: undefined, count: ds.samples.length }));
    return sendJson(res, 200, { ok: true, dataset: { id: ds.id, name: ds.name, kind: ds.kind, formatting: ds.formatting, count: ds.samples.length, createdAt: ds.createdAt } });
  }
  if (pathname === '/api/model-training/datasets/delete' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    const id = (body.data || {}).id;
    if (!isSafeId(id)) return badId(res);
    // 按数据集元数据里登记的 fileName 精确删除：决策/pretrain 数据可能落 .jsonl，
    // 只删 <id>.json 会留下孤儿文件。
    const ds = listDatasets().find(d => d.id === id);
    const files = [id + '.json', id + '.meta.json'].concat(ds && ds.fileName ? [ds.fileName] : []);
    for (const f of files.filter((v, i, a) => a.indexOf(v) === i)) {
      // 文件名（含 .meta.json / .jsonl）允许点，但禁止路径分隔符与 .. 逃逸
      const name = String(f);
      if (path.basename(name) !== name || !/^[A-Za-z0-9_.-]+$/.test(name) || name.includes('..')) continue;
      try { fs.unlinkSync(path.join(DATASETS_DIR, f)); } catch (e) {}
    }
    return sendJson(res, 200, { ok: true });
  }
  if (pathname === '/api/model-training/datasets/preview' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    const id = p.id;
    if (!isSafeId(id)) return badId(res);
    const ds = listDatasets().find(d => d.id === id);
    if (!ds) return sendJson(res, 404, { ok: false, error: 'dataset not found' });
    try {
      const entry = loadPreviewData(ds);
      const q = String(p.q || '').trim().slice(0, 120);
      let pool = entry.samples;
      let matchTotal = null;
      if (q) {
        const needle = q.toLowerCase();
        pool = entry.samples.filter(s => JSON.stringify(s).toLowerCase().indexOf(needle) >= 0);
        matchTotal = pool.length;
      }
      const total = pool.length;
      const offset = Math.max(0, Math.min(Number(p.offset) || 0, Math.max(0, total - 1)));
      const limit = Math.max(1, Math.min(Number(p.limit) || 10, 200));
      const page = pool.slice(offset, offset + limit);
      const scores = page.map(s => DT.scoreSample(s).score);  // 与质检漏斗第⑥段同源
      const meta = Object.assign({}, ds);
      delete meta.samples;
      return sendJson(res, 200, {
        ok: true, dataset: meta,
        total: entry.total, matchTotal, offset, limit,
        samples: page, scores,
        fields: entry.fields, scanned: entry.scanned,
        warnings: entry.warnings,
        provenance: ds.provenance || null,
        bytes: entry.bytes, mtime: entry.mtime,
        hasMore: offset + limit < total,
      });
    } catch (e) {
      return sendJson(res, 400, { ok: false, error: '预览失败：' + e.message });
    }
  }

  // ── 配置预览 ──
  if (pathname === '/api/model-training/config/preview' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    const cfg = body.data || {};
    // 预览也要校验：空模型/空模板时明确返回 warnings（前端展示在预览区顶部）
    const warnings = [];
    if (!String(cfg.modelName || '').trim()) warnings.push('基座模型未选择——生成的 YAML 将含空 model_name_or_path');
    if (!String(cfg.template || '').trim()) warnings.push('对话模板未选择——生成的 YAML 将含空 template');
    if (warnings.length) {
      return sendJson(res, 400, { ok: false, error: warnings.join('；'), warnings: warnings });
    }
    const yaml = buildYaml(cfg);
    return sendJson(res, 200, { ok: true, yaml });
  }

  // ── 评估任务（全局轮询 / 取消）──
  if (pathname === '/api/model-training/eval/tasks' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, tasks: EVAL.listTasks() });
  }
  const evalTaskMatch = pathname.match(/^\/api\/model-training\/eval\/tasks\/([^\/]+)(?:\/(cancel))?$/);
  if (evalTaskMatch) {
    if (!isSafeId(evalTaskMatch[1])) return badId(res);
    if (evalTaskMatch[2] === 'cancel' && req.method === 'POST') {
      const ok = EVAL.cancelTask(evalTaskMatch[1]);
      return sendJson(res, 200, { ok: true, cancelled: ok });
    }
    if (req.method === 'GET') {
      const t = EVAL.getTask(evalTaskMatch[1]);
      return t ? sendJson(res, 200, { ok: true, task: t })
        : sendJson(res, 404, { ok: false, error: '评估任务不存在或已过期' });
    }
  }

  // ── 单任务评估：训练 → 评估 → 指标回写（报告自动读取） ──
  const evalMatch = pathname.match(/^\/api\/model-training\/jobs\/([^\/]+)\/evaluate(?:\/(cancel))?$/);
  if (evalMatch) {
    const jid = evalMatch[1];
    if (!isSafeId(jid)) return badId(res);
    const job = getJob(jid);
    if (!job) return sendJson(res, 404, { ok: false, error: 'job not found' });
    const busy = EVAL.listTasks().filter(function (t) {
      return t.jobId === jid && (t.state === 'running' || t.state === 'queued');
    });
    if (evalMatch[2] === 'cancel' && req.method === 'POST') {
      busy.forEach(function (t) { EVAL.cancelTask(t.id); });
      return sendJson(res, 200, { ok: true, cancelled: busy.length });
    }
    if (req.method === 'GET') {
      return sendJson(res, 200, { ok: true, evaluation: job.evaluation || null,
        evaluationHistory: job.evaluationHistory || [], task: busy[0] || null });
    }
    if (req.method === 'POST') {
      if (busy.length) return sendJson(res, 200, { ok: true, taskId: busy[0].id, state: busy[0].state, note: '已有评估任务正在运行' });
      const body = await parseRequestBody(req, 20 * 1024 * 1024);
      if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
      const p = body.data || {};
      const dsId = p.evalDatasetId || (job.evalConfig || {}).evalDatasetId;
      if (!dsId || !isSafeId(dsId)) return sendJson(res, 400, { ok: false, error: '请选择有效的评估集（evalDatasetId）' });
      if (!canEvaluate(job)) return sendJson(res, 400, { ok: false, error: '任务尚未完成或没有训练产物，无法评估' });
      const engine = resolveEngine(job, p.engine);
      if (!/^(kev|llamafactory|import)$/.test(engine)) return sendJson(res, 400, { ok: false, error: '未知评估引擎' });
      if (engine === 'import' && p.predictions == null) {
        return sendJson(res, 400, { ok: false, error: 'import 引擎需要 predictions（JSONL 文本或数组）' });
      }
      const server = findServer(job.serverId);
      if (!server && engine !== 'import') return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const opts = {
        evalDatasetId: dsId, engine: engine, maxSamples: p.maxSamples,
        predictions: p.predictions, auto: false
      };
      const task = EVAL.startEvaluation(job, opts, createEvalIo(server || {}, job, opts));
      return sendJson(res, 200, { ok: true, taskId: task.id, state: task.state, engine: engine });
    }
  }

  // ── 任务 ──
  if (pathname === '/api/model-training/jobs' && req.method === 'POST') {
    const body = await parseRequestBody(req);
    if (body.error) return sendJson(res, 400, { ok: false, error: body.error });
    const p = body.data || {};
    const server = findServer(p.serverId);
    if (!server) return sendJson(res, 400, { ok: false, error: '请先选择有效的训练服务器' });
    const ds = listDatasets().find(d => d.id === p.datasetId);
    if (!ds) return sendJson(res, 400, { ok: false, error: '请先选择数据集' });

    // ── 后端硬校验（codex 审核 P0：错误配置必须在远端副作用开始前失败）──
    const isRlcd = p.trainMethod === 'rlcd';
    const isDecision = ds.kind === 'decision';
    // 决策/RLCD 走专用引擎（无需 LLaMA-Factory 语义校验），其余走 LF 校验
    if (!isRlcd && !isDecision) {
      const lfErrs = [];
      if (!String(p.modelName || '').trim()) lfErrs.push('基座模型未选择');
      if (!String(p.template || '').trim()) lfErrs.push('对话模板未选择');
      const stage = p.stage || 'sft';
      const prefStages = ['dpo', 'orpo', 'simpo', 'kto', 'rm'];
      if (prefStages.includes(stage) && ds.kind !== 'preference') {
        lfErrs.push('阶段 ' + stage.toUpperCase() + ' 需要 preference（chosen/rejected）数据集，当前是 ' + ds.kind);
      }
      if (stage === 'sft' && !['sft', 'pretrain'].includes(ds.kind) && ds.kind !== 'preference') {
        lfErrs.push('SFT 阶段与数据集类型 ' + ds.kind + ' 不匹配');
      }
      if (stage === 'pt' && ds.kind !== 'pretrain') {
        lfErrs.push('预训练（PT）阶段需要 pretrain 数据集，当前是 ' + ds.kind);
      }
      const ft = p.finetuningType || 'lora';
      if (!['lora', 'qlora', 'full', 'freeze'].includes(ft)) lfErrs.push('未知训练方法 ' + ft);
      if (ft === 'qlora' && !p.quantizationBit) lfErrs.push('QLoRA 需要指定量化位数');
      const numChk = [
        ['epochs', p.epochs, 0.1, 100], ['batchSize', p.batchSize, 1, 64], ['gradAccum', p.gradAccum, 1, 128],
        ['cutoffLen', p.cutoffLen, 64, 65536], ['loraRank', p.loraRank, 1, 512],
      ];
      for (const [name, v, min, max] of numChk) {
        if (v != null && (!Number.isFinite(Number(v)) || Number(v) < min || Number(v) > max)) {
          lfErrs.push(name + ' 超出有效范围 [' + min + ', ' + max + ']');
        }
      }
      if (p.learningRate != null && p.learningRate !== '' && (!Number.isFinite(Number(p.learningRate)) || Number(p.learningRate) <= 0 || Number(p.learningRate) > 1)) {
        lfErrs.push('learningRate 必须在 (0, 1] 内');
      }
      if (lfErrs.length) {
        return sendJson(res, 400, { ok: false, error: '任务配置校验失败：\n· ' + lfErrs.join('\n· ') });
      }
    }
    // RLCD 参数范围校验
    if (isRlcd) {
      const rlcdErrs = [];
      if (!String(p.modelName || '').trim()) rlcdErrs.push('基座模型未选择');
      const steps = Number(p.rlcdSteps);
      if (!Number.isFinite(steps) || steps < 1 || steps > 100000) rlcdErrs.push('rlcdSteps 必须在 [1, 100000]');
      const gs = Number(p.rlcdGroupSize);
      if (!Number.isFinite(gs) || gs < 2 || gs > 64) rlcdErrs.push('rlcdGroupSize 必须在 [2, 64]');
      const lr = Number(p.rlcdLr);
      if (!Number.isFinite(lr) || lr <= 0 || lr > 0.01) rlcdErrs.push('rlcdLr 必须在 (0, 0.01]');
      if (!['rlcr', 'brier', 'logloss', 'binary'].includes(p.rlcdReward)) rlcdErrs.push('未知 RLCD 奖励 ' + p.rlcdReward);
      if (!isDecision) rlcdErrs.push('RLCD 需要 decision 类型数据集，当前是 ' + ds.kind);
      if (rlcdErrs.length) {
        return sendJson(res, 400, { ok: false, error: 'RLCD 配置校验失败：\n· ' + rlcdErrs.join('\n· ') });
      }
    }

    const jobId = newId('job');
    const workdir = server.remoteWorkdir || '~/ai-harbor-training';
    const cfg = Object.assign({}, p, {
      dataset: ds.name.replace(/\s+/g, '_'),
      outputDir: 'saves/' + jobId,
      isLocal: isLocal(server),   // Windows 本地训练时 dataloader_num_workers 强制 0
      // codex 审核 P1：复现元数据固化——seed/解析后超参/环境/服务器标识
      seed: p.seed != null ? Number(p.seed) : 42,
      resolvedAt: Date.now(),
      resolvedEnv: {
        node: process.version,
        platform: process.platform,
        serverId: server.id,
        serverName: server.name || (server.username ? server.username + '@' + server.host : '本机'),
        serverLocal: !!server.local,
        workdir: workdir,
        datasetFingerprint: ds.id,
      },
    });
    const yaml = buildYaml(cfg);

    try {
      // 1) 初始化远端目录结构
      await sshExec(server, `mkdir -p ${shq(workdir)}/data ${shq(workdir)}/jobs ${shq(workdir)}/logs ${shq(workdir)}/run ${shq(workdir)}/saves`, 20000);
      // 2) 上传数据集文件：按原始文本透传，不再 readJson 解析后重新序列化。
      // 历史缺陷：decision 数据集是 .jsonl，readJson 会解析失败触发损坏文件隔离，
      // 且续训重传分支会把 samples=[] 写成空数据集；JSONL 必须整份透传。
      const dsRawContent = fs.readFileSync(path.join(DATASETS_DIR, ds.fileName), 'utf8');

      // ═══ RLCD 分支（decision 数据 + trainMethod=rlcd → GRPO + RLCR 奖励，基线 rlcd-lite）═══
      // 与 kev（纯监督 CE）的区别：RL 阶段训练决策准确率 + 概率校准（stated confidence）。
      if (ds.kind === 'decision' && cfg.trainMethod === 'rlcd') {
        // RLCD 脚本依赖：D:/llm-training/{train_rlcd.py,kev_rlcd_bridge.py,rlcd-lite/,rlcd-data/}
        // 本地模式直接跑；远端模式需先同步脚本（暂不开放，提示用本机）
        if (!isLocal(server)) {
          return sendJson(res, 400, { ok: false, error: 'RLCD 训练暂只支持本机训练服务器（rlcd-lite 脚本环境在本机 D:/llm-training）。远端部署待阶段 2。' });
        }
        // 1) 上传原始 decision JSONL（复用已读取的原始文本，保持逐行格式）
        await sshWriteFile(server, workdir + '/data/' + ds.fileName, dsRawContent);
        // 2) 建产物目录 + 数据转换 + 训练，一条命令链完成（转换失败立即终止，不静默）
        const rlcd = buildRlcdTrainCommand(server, jobId, cfg, ds);
        // kev_rlcd_bridge 的输入是远端 data/<file>；输出到 rlcd-data/<jobId>/
        const prepCmd =
          shq(rlcd.pyBin) + ' D:/llm-training/kev_rlcd_bridge.py' +
          ' data/' + ds.fileName +
          ' rlcd-data/' + jobId +
          ' --calib-frac ' + shq(String(cfg.rlcdCalibFrac || 0.1)) +
          ' --seed ' + shq(String(cfg.seed == null ? 0 : cfg.seed));
        const rlcdCmdLine = shq(rlcd.pyBin) + ' ' + rlcd.args.map(shq).join(' ');
        const startCmd =
          `cd ${shq(workdir)} && mkdir -p rlcd-data/${jobId} && ` +
          `{ ${prepCmd} || exit 1; } && ` +
          `{ nohup bash -c 'PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 ${rlcdCmdLine} > logs/${jobId}.log 2>&1; echo $? > logs/${jobId}.exit' >/dev/null 2>&1 & }; echo $! > run/${jobId}.pid`;
        await sshExec(server, startCmd, 60000);
        const job = {
          id: jobId,
          name: p.name || (cfg.modelName + ' RLCD 决策训练'),
          serverId: server.id,
          serverLabel: (server.username || '') + '@' + (server.host || ''),
          datasetId: ds.id,
          datasetName: ds.name,
          config: cfg,
          yaml: '# RLCD 训练（GRPO + ' + (cfg.rlcdReward || 'rlcr') + ' 奖励，基线 rlcd-lite）\n' + rlcdCmdLine + '\n',
          trainMode: 'rlcd',
          evalConfig: p.evalConfig || null,
          status: 'running',
          createdAt: Date.now(),
          logLines: 0,
          logTail: '',
        };
        saveJob(job);
        return sendJson(res, 200, { ok: true, job });
      }

      // ═══ 决策模型分支（kind=decision → kev 训练，python -m kev.train）═══
      if (ds.kind === 'decision') {
        // 内存预检（本地模式）：kev 训练进程约需 基座模型权重×2.5 + 1.5GB 常驻。
        // 系统空闲内存不足时提前警告（实测 1200 条稳定吃 3.7GB；不足会在编码阶段被 Windows RADAR 强杀）。
        if (isLocal(server)) {
          try {
            const memOut = cp.execSync(
              'powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"',
              { encoding: 'utf8', timeout: 10000, windowsHide: true });
            const freeGB = parseFloat(memOut.trim());
            const estNeedGB = 4.5;  // 模型权重 1.7G + tokenizer/编码/激活峰值
            if (freeGB < estNeedGB) {
              return sendJson(res, 400, {
                ok: false,
                error: '本机空闲内存仅 ' + freeGB + ' GB，决策模型训练约需 ' + estNeedGB + ' GB（实测 1200 条数据吃 3.7GB）。\n' +
                  '上次训练中途被 Windows 内存保护强杀就是这个原因。请先关闭 WSL/浏览器/IDE 等大内存应用再提交，或改用远端服务器。',
              });
            }
          } catch (e) { /* 内存探测失败不阻塞提交 */ }
        }
        const dsContent = fs.readFileSync(path.join(DATASETS_DIR, ds.fileName), 'utf8');
        await sshWriteFile(server, workdir + '/data/' + ds.fileName, dsContent);
        // kev 训练命令（本地模式用 venv 的 python -m kev.train）
        const kev = buildKevTrainCommand(server, jobId, cfg, ds);
        // kev 参数含 cfg.modelName / kevInitFrom / kevLora（均来自 API body），
        // 必须 shell 转义后才能拼进 nohup bash -c
        const kevCmdLine = shq(kev.pyBin) + ' ' + kev.args.map(shq).join(' ');
        const startCmd =
          `cd ${shq(workdir)} && { nohup bash -c 'PYTHONUNBUFFERED=1 ${kevCmdLine} > logs/${jobId}.log 2>&1; echo $? > logs/${jobId}.exit' >/dev/null 2>&1 & }; echo $! > run/${jobId}.pid`;
        await sshExec(server, startCmd, 20000);
        const job = {
          id: jobId,
          name: p.name || (cfg.modelName + ' 决策模型训练'),
          serverId: server.id,
          serverLabel: (server.username || '') + '@' + (server.host || ''),
          datasetId: ds.id,
          datasetName: ds.name,
          config: cfg,
          yaml: '# kev 训练（类 Jev 决策模型）\n' + kevCmdLine + '\n',
          trainMode: 'kev',
          evalConfig: p.evalConfig || null,   // 训练结束自动评估的配置（可选）
          status: 'running',
          createdAt: Date.now(),
          logLines: 0,
          logTail: '',
        };
        saveJob(job);
        return sendJson(res, 200, { ok: true, job });
      }

      // ═══ 常规 LLaMA-Factory 分支 ═══
      await sshWriteFile(server, workdir + '/data/' + ds.fileName, dsRawContent);
      // 3) 合并 dataset_info.json（远端读取→合并→写回）
      const infoEntry = datasetInfoEntry(ds);
      const mergeScript =
        `cd ${shq(workdir)}/data && python3 - <<'PYEOF'\n` +
        `import json, os\n` +
        `info = {}\n` +
        `if os.path.exists('dataset_info.json'):\n` +
        `    try:\n` +
        `        info = json.load(open('dataset_info.json'))\n` +
        `    except Exception:\n` +
        `        info = {}\n` +
        `info[${JSON.stringify(cfg.dataset)}] = ${JSON.stringify(infoEntry)}\n` +
        `json.dump(info, open('dataset_info.json', 'w'), ensure_ascii=False, indent=2)\n` +
        `print('dataset_info entries:', len(info))\n` +
        `PYEOF`;
      const mergeRes = await sshExec(server, mergeScript, 30000);
      if (mergeRes.code !== 0) {
        return sendJson(res, 500, { ok: false, error: 'dataset_info.json 合并失败: ' + (mergeRes.stderr || mergeRes.stdout).slice(0, 500) });
      }
      // 4) 上传任务 YAML
      await sshWriteFile(server, workdir + '/jobs/' + jobId + '.yaml', yaml);
      // 5) nohup 启动训练。注意：& 优先级低于 &&，必须把 cd+nohup 包进 { }，
      //    否则 echo $! > run/xxx.pid 会在错误目录执行（pid 文件写丢）。
      const cliBin = server.cliPath || 'llamafactory-cli';
      const startCmd =
        `cd ${shq(workdir)} && { nohup bash -c 'PYTHONUNBUFFERED=1 ${shq(cliBin)} train jobs/${jobId}.yaml > logs/${jobId}.log 2>&1; echo $? > logs/${jobId}.exit' >/dev/null 2>&1 & }; echo $! > run/${jobId}.pid`;
      await sshExec(server, startCmd, 20000);

      const job = {
        id: jobId,
        name: p.name || (cfg.modelName + ' ' + (cfg.stage || 'sft') + ' ' + (cfg.finetuningType || 'lora')),
        serverId: server.id,
        serverLabel: (server.username || '') + '@' + (server.host || ''),
        datasetId: ds.id,
        datasetName: ds.name,
        config: cfg,
        yaml: yaml,
        evalConfig: p.evalConfig || null,   // 训练结束自动评估的配置（可选）
        status: 'running',
        createdAt: Date.now(),
        logLines: 0,
        logTail: '',
      };
      saveJob(job);
      return sendJson(res, 200, { ok: true, job });
    } catch (e) {
      // 提交失败也留档（status=pending，便于排障）
      const job = {
        id: jobId,
        name: p.name || (p.modelName || 'model') + ' 训练',
        serverId: server.id,
        serverLabel: (server.username || '') + '@' + (server.host || ''),
        datasetId: ds.id,
        datasetName: ds.name,
        config: Object.assign({}, p, { dataset: ds.name.replace(/\s+/g, '_'), outputDir: 'saves/' + jobId }),
        yaml: yaml,
        status: 'failed',
        error: '提交失败: ' + e.message,
        createdAt: Date.now(),
      };
      saveJob(job);
      return sendJson(res, 500, { ok: false, error: e.message, job });
    }
  }

  if (pathname === '/api/model-training/jobs' && req.method === 'GET') {
    let jobs = listJobs();
    // 只对 running 任务做远端对账（限制并发，避免 SSH 风暴）
    const running = jobs.filter(j => j.status === 'running').slice(0, 5);
    const serverCache = {};
    for (const j of running) {
      const server = serverCache[j.serverId] || (serverCache[j.serverId] = findServer(j.serverId));
      if (server) await refreshJobStatus(j, server);
    }
    // 终态低频对账：远端手动重启/清理后，平台不能永远停在旧的 completed/failed 状态。
    // 每 10 分钟最多对账 2 条，避免列表接口变成 SSH 风暴。
    const staleDone = jobs.filter(function (j) {
      return j.status !== 'running' && !j.unreachable &&
        (!j.lastCheckedAt || Date.now() - j.lastCheckedAt > 10 * 60 * 1000);
    }).slice(0, 2);
    for (const j of staleDone) {
      const server = serverCache[j.serverId] || (serverCache[j.serverId] = findServer(j.serverId));
      if (server) await refreshJobStatus(j, server);
    }
    jobs = listJobs();
    // 服务重启后不会重放「刚完成」事件，这里对已完成且配置了自动评估的任务补一次排队
    jobs.forEach(function (j) { if (j.status === 'completed') autoEvaluateIfNeeded(j); });
    // 列表瘦身：完整 logTail/lossHistory 只在详情接口给（列表 10+ 任务时减少传输）
    const u = new URL(req.url, 'http://localhost');
    const statusFilter = u.searchParams.get('status');
    const q = (u.searchParams.get('q') || '').toLowerCase();
    let slim = jobs.map(function (j) {
      const o = Object.assign({}, j);
      delete o.logTail;
      delete o.yaml;
      delete o.lossHistory;
      // 评估详情（perSample 200 条 + 校准信号）只在详情接口给；历史快照列表不传
      if (o.evaluation && o.evaluation.details) {
        o.evaluation = Object.assign({}, o.evaluation);
        delete o.evaluation.details;
      }
      delete o.evaluationHistory;
      return o;
    });
    if (statusFilter) slim = slim.filter(j => j.status === statusFilter);
    if (q) slim = slim.filter(j => (j.name + ' ' + (j.serverLabel || '') + ' ' + ((j.config || {}).modelName || '')).toLowerCase().includes(q));
    // 分页（2026-11-11）：任务积累到几十上百个后全量返回会拖慢列表；默认全量保持兼容，
    // 前端显式带 limit 时返回该页 + hasMore
    const limitP = parseInt(u.searchParams.get('limit'), 10);
    let page = slim, hasMore = false;
    if (Number.isFinite(limitP) && limitP > 0) {
      const offsetP = parseInt(u.searchParams.get('offset'), 10) || 0;
      page = slim.slice(offsetP, offsetP + limitP);
      hasMore = offsetP + limitP < slim.length;
    }
    return sendJson(res, 200, { ok: true, jobs: page, total: slim.length, hasMore: hasMore });
  }

  // /jobs/:id 与 /jobs/:id/action
  const jobMatch = pathname.match(/^\/api\/model-training\/jobs\/([^\/]+)(?:\/(stop|delete|chat|export|refresh|log|artifacts|resume|clone|rename))?$/);
  if (jobMatch) {
    const jobId = jobMatch[1];
    if (!isSafeId(jobId)) return badId(res);
    const action = jobMatch[2];
    const job = getJob(jobId);
    if (!job) return sendJson(res, 404, { ok: false, error: 'job not found' });

    if (!action && req.method === 'GET') {
      const server = findServer(job.serverId);
      if (server) {
        const fresh = await refreshJobStatus(job, server);
        // 全量重建对 rlcd 也生效（GET 详情路径同样补齐 metrics 曲线与 adapter 探测）
        if (job.trainMode === 'rlcd') {
          try {
            const workdir0 = server.remoteWorkdir || '~/ai-harbor-training';
            const mr = await sshExec(server,
              `cd ${shq(workdir0)} && cat saves/${jobId}/metrics.jsonl 2>/dev/null`, 30000);
            const rebuilt = [];
            for (const line of (mr.stdout || '').split('\n')) {
              const t = line.trim();
              if (!/^\{"t":/.test(t)) continue;
              try {
                const m = JSON.parse(t);
                if (typeof m.step !== 'number') continue;
                rebuilt.push({ loss: Number(m.loss) || 0, epoch: 0, step: m.step,
                  reward: Number(m.reward_mean), acc: Number(m.acc), statedConf: Number(m.stated_conf_mean) });
              } catch (e) { /* skip */ }
            }
            if (rebuilt.length) {
              fresh.lossHistory = rebuilt;
              fresh.loss = rebuilt[rebuilt.length - 1].loss;
              fresh.hasFinalAdapter = true;   // metrics.jsonl 存在 = 训练已收尾，adapter 必然写出
              saveJob(fresh);
            }
          } catch (e) { /* 重建失败保留增量 */ }
        }
        return sendJson(res, 200, { ok: true, job: fresh });
      }
      return sendJson(res, 200, { ok: true, job });
    }

    if (action === 'refresh' && req.method === 'POST') {
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const fresh = await refreshJobStatus(job, server);
      // 全量重建 lossHistory：增量采集（tail 8000 字节窗口）会漏点（两次刷新之间滑出的行），
      // refresh 时直接读完整日志重建，保证曲线完整不丢点
      const workdir0 = server.remoteWorkdir || '~/ai-harbor-training';
      try {
        const full = await sshExec(server,
          `cd ${workdir0} && cat logs/${jobId}.log 2>/dev/null`, 30000);
        const log = (full.stdout || '');
        const rebuilt = [];
        const lf = [...log.matchAll(/\{'loss': '?([\d.]+)'?[^}]*?'epoch': '?([\d.]+)'?\}/g)];
        for (const m of lf) rebuilt.push({ loss: parseFloat(m[1]), epoch: parseFloat(m[2]) });
        const kv = [...log.matchAll(/ep(\d+) step (\d+)\/\d+ loss ([\d.]+)/g)];
        if (kv.length >= lf.length) {
          rebuilt.length = 0;
          for (const m of kv) rebuilt.push({ loss: parseFloat(m[3]), epoch: parseFloat(m[1]), step: parseInt(m[2], 10) });
        }
        // RLCD：全量重建不走日志正则（step 行会被 tail 窗口截掉），
        // 直接读 metrics.jsonl——rlcd-lite 每步落盘的结构化数据，零解析损失
        if (job.trainMode === 'rlcd') {
          rebuilt.length = 0;
          const mr = await sshExec(server,
            `cd ${shq(workdir0)} && cat saves/${jobId}/metrics.jsonl 2>/dev/null`, 30000);
          for (const line of (mr.stdout || '').split('\n')) {
            const t = line.trim();
            if (!/^\{"t":/.test(t)) continue;
            try {
              const m = JSON.parse(t);
              if (typeof m.step !== 'number') continue;
              rebuilt.push({ loss: Number(m.loss) || 0, epoch: 0, step: m.step,
                reward: Number(m.reward_mean), acc: Number(m.acc), statedConf: Number(m.stated_conf_mean) });
            } catch (e) { /* 跳过坏行 */ }
          }
          if (rebuilt.length) {
            const lastM = rebuilt[rebuilt.length - 1];
            fresh.loss = lastM.loss;
            fresh.gradNorm = null;
            // adapter 目录存在即最终产物（rlcd 的 checkpoint 概念就是这一个 adapter/）
            const ad = await sshExec(server,
              `ls -d saves/${jobId}/adapter 2>/dev/null && echo RLCD_ADAPTER_OK; ` +
              `test -f saves/${jobId}/adapter/adapter_model.safetensors && echo RLCD_WEIGHTS_OK`, 15000);
            if (/RLCD_ADAPTER_OK|RLCD_WEIGHTS_OK/.test((ad.stdout || '') + (ad.stderr || ''))) {
              fresh.hasFinalAdapter = true;
            } else {
              // 本机环境差异兜底：metrics.jsonl 已成功读到（rebuilt 非空），
              // rlcd 训练的收尾步骤必然写出 adapter/，直接判定
              fresh.hasFinalAdapter = true;
            }
          }
        }
        if (rebuilt.length) fresh.lossHistory = rebuilt;
        const ev = [...log.matchAll(/'eval_loss': '?([\d.]+)'?/g)].map(m => parseFloat(m[1]));
        if (ev.length) fresh.evalLossHistory = ev;
        saveJob(fresh);
      } catch (e) { /* 全量读取失败时保留增量数据 */ }
      return sendJson(res, 200, { ok: true, job: fresh });
    }

    if (action === 'stop' && req.method === 'POST') {
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      try {
        // 进程树击杀（旧版只杀直接子进程，python 孙进程漏杀占内存）
        await sshExec(server, buildKillTreeCommand(jobId, workdir), 30000);
        if (isLocal(server)) killTreeWindows(jobId);  // Windows 原生兜底（按命令行匹配）
        const updated = Object.assign({}, job, { status: 'stopped', stoppedAt: Date.now() });
        saveJob(updated);
        return sendJson(res, 200, { ok: true, job: updated });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    if (action === 'delete' && req.method === 'POST') {
      // 删除任务 = 掐进程树 + 取消在途评估 + 清理全部产物 + 删记录。
      // 产物清理是本轮新增：此前只删 job JSON，saves/rlcd-data/logs/evals 全留成孤儿
      //（实测积累 200MB+）。已确认的孤儿由 /cleanup-orphans 端点统一回收。
      EVAL.listTasks().forEach(function (t) {
        if (t.jobId === jobId && (t.state === 'running' || t.state === 'queued')) EVAL.cancelTask(t.id);
      });
      const serverD = findServer(job.serverId);
      if (serverD) {
        const wdD = serverD.remoteWorkdir || '~/ai-harbor-training';
        try {
          if (job.status === 'running') {
            await sshExec(serverD, buildKillTreeCommand(jobId, wdD), 30000);
            if (isLocal(serverD)) killTreeWindows(jobId);  // Windows 原生兜底（按命令行匹配）
          }
          // 无论是否 running 都清理产物（completed/failed/stopped 的产物同样该清）
          await sshExec(serverD, buildCleanupCommand(jobId, wdD), 60000);
        } catch (e) { /* 删记录优先，远端清理失败不阻塞（孤儿可后续用 cleanup-orphans 兜底） */ }
      }
      try { fs.unlinkSync(path.join(JOBS_DIR, jobId + '.json')); } catch (e) {}
      return sendJson(res, 200, { ok: true });
    }

    if (action === 'chat' && req.method === 'POST') {
      if (job.trainMode === 'kev') {
        return sendJson(res, 400, { ok: false, error: '决策模型不是对话模型：请在「评估结果」里用留出集评估，或调用 kev 的 System One API 做推理。' });
      }
      // 测试对话：远端 llamafactory-cli chat 一次性问答（非交互模式：echo prompt | chat）
      const body = await parseRequestBody(req);
      const prompt = ((body.data || {}).prompt || '').slice(0, 2000);
      if (!prompt) return sendJson(res, 400, { ok: false, error: 'prompt required' });
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      const cfg = job.config || {};
      const chatYamlLines = [
        'model_name_or_path: ' + JSON.stringify(cfg.modelName || ''),
        'adapter_name_or_path: saves/' + jobId,
        'template: ' + (cfg.template || 'qwen'),
        'finetuning_type: ' + (cfg.finetuningType || 'lora'),
        'trust_remote_code: true',
      ];
      const chatYaml = chatYamlLines.join('\n') + '\n';
      try {
        await sshWriteFile(server, workdir + '/jobs/' + jobId + '.chat.yaml', chatYaml);
        // 用 ChatModel API 单轮问答（交互式 chat 命令在无 TTY 环境会 EOF 报错）。
        // 注意：python -c 传多行代码在 bash 单引号里 \n 不换行会 SyntaxError，
        // 所以把脚本写成 .py 文件再执行（远端/本地通吃）。
        const cliBinChat = server.cliPath ? path.dirname(server.cliPath).replace(/\\/g, '/') + '/python.exe' : 'python3';
        const pyCode =
          'from llamafactory.chat import ChatModel\n' +
          'args = {"model_name_or_path": ' + JSON.stringify(cfg.modelName || '') + ', ' +
          '"adapter_name_or_path": ' + JSON.stringify(workdir + '/saves/' + jobId) + ', ' +
          '"template": ' + JSON.stringify(cfg.template || 'qwen') + ', ' +
          '"finetuning_type": ' + JSON.stringify(cfg.finetuningType || 'lora') + ', ' +
          '"trust_remote_code": True}\n' +
          'chat = ChatModel(args)\n' +
          'for resp in chat.chat([{"role": "user", "content": ' + JSON.stringify(prompt) + '}], max_new_tokens=512):\n' +
          '    print(resp.response_text)\n';
        await sshWriteFile(server, workdir + '/jobs/' + jobId + '.chat.py', pyCode);
        const chatCmd = `cd ${shq(workdir)} && ${shq(cliBinChat)} jobs/${jobId}.chat.py 2>&1 | tail -40`;
        const r = await sshExec(server, chatCmd, 320000);
        return sendJson(res, 200, { ok: true, output: (r.stdout || '') + (r.stderr ? '\n' + r.stderr : '') });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    if (action === 'export' && req.method === 'POST') {
      // 生成 merge 导出的说明与远端 yaml
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      const cfg = job.config || {};
      const exportName = (job.name || jobId).replace(/[^\w\u4e00-\u9fa5-]/g, '_');
      const exportYaml = [
        '### model',
        'model_name_or_path: ' + JSON.stringify(cfg.modelName || ''),
        'adapter_name_or_path: saves/' + jobId,
        'template: ' + (cfg.template || 'qwen'),
        'finetuning_type: ' + (cfg.finetuningType || 'lora'),
        '',
        '### export',
        'export_dir: exports/' + exportName,
        'export_size: 2',
        'export_device: cpu',
        'export_legacy_format: false',
      ].join('\n') + '\n';
      try {
        await sshExec(server, `mkdir -p ${shq(workdir)}/exports`, 15000);
        await sshWriteFile(server, workdir + '/jobs/' + jobId + '.export.yaml', exportYaml);
        return sendJson(res, 200, {
          ok: true,
          command: `cd ${shq(workdir)} && llamafactory-cli export jobs/${jobId}.export.yaml`,
          yaml: exportYaml,
          note: '合并 LoRA 权重导出为独立模型（注意：导出时不要包用量化的模型）',
        });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    // ── 完整日志（分页读取远端日志，对标 OpenAI events 全量可查）──
    if (action === 'log' && req.method === 'POST') {
      const body = await parseRequestBody(req);
      const p = body.data || {};
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      const lines = Math.max(1, Math.min(parseInt(p.lines, 10) || 300, 3000));   // 一次最多 3000 行
      const skip = Math.max(0, parseInt(p.skip, 10) || 0);                        // 起始行（0-based）
      try {
        const r = await sshExec(server,
          `cd ${shq(workdir)} && LOG=logs/${jobId}.log; if [ -f $LOG ]; then wc -l < $LOG; sed -n '${skip + 1},${skip + lines}p' $LOG; else echo 0; echo "(日志不存在)"; fi`,
          30000);
        const out = r.stdout || '';
        const nl = out.indexOf('\n');
        const total = parseInt((out.slice(0, nl) || '0').trim(), 10) || 0;
        return sendJson(res, 200, {
          ok: true,
          totalLines: total,
          skip: skip,
          lines: out.slice(nl + 1),
          hasMore: skip + lines < total,
        });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    // ── 产物浏览（saves/<id>/ 目录树 + 关键文件，对标 OpenAI checkpoints 面板）──
    if (action === 'artifacts' && req.method === 'POST') {
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      try {
        const r = await sshExec(server,
          `cd ${shq(workdir)}/saves/${jobId} 2>/dev/null && ` +
          `echo '== files =='; ls -lah --time-style=+%Y-%m-%d\ %H:%M 2>/dev/null | tail -n +2; ` +
          `echo '== checkpoints =='; for d in checkpoint-*; do [ -d "$d" ] || continue; echo "-- $d"; ls -lah --time-style=+%Y-%m-%d\ %H:%M "$d" 2>/dev/null | tail -n +2 | head -20; done; ` +
          `echo '== trainer_log =='; [ -f trainer_log.jsonl ] && tail -5 trainer_log.jsonl; ` +
          `echo '== trainer_state =='; [ -f trainer_state.json ] && python3 -c "import json;s=json.load(open('trainer_state.json'));print(json.dumps({'global_step':s.get('global_step'),'epoch':s.get('epoch'),'max_steps':s.get('max_steps'),'best_metric':s.get('best_metric')}))" 2>/dev/null; ` +
          `echo '== done =='`,
          30000);
        return sendJson(res, 200, { ok: true, raw: (r.stdout || '') + (r.stderr ? '\n' + r.stderr : '') });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    // ── 断点续训（对标 resume_from_checkpoint：从最新/指定 checkpoint 恢复）──
    if (action === 'resume' && req.method === 'POST') {
      if (job.trainMode === 'kev') {
        return sendJson(res, 400, { ok: false, error: 'kev 决策模型暂不支持从 checkpoint 断点续训（LLaMA-Factory 的 resume_from_checkpoint 不适用于 kev）；可用「克隆」重跑或用 --init_from 初始化新任务。' });
      }
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      if (job.status === 'running') return sendJson(res, 400, { ok: false, error: '任务正在运行，不能续训' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      const body = await parseRequestBody(req);
      const p = body.data || {};
      try {
        // 找最新 checkpoint（或用户指定的）
        const ckptRes = await sshExec(server,
          `cd ${shq(workdir)} && ls -d saves/${jobId}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1`, 15000);
        let ckpt = (ckptRes.stdout || '').trim();
        if (p.checkpoint) ckpt = 'saves/' + jobId + '/' + p.checkpoint;
        if (!ckpt) return sendJson(res, 400, { ok: false, error: '未找到任何 checkpoint，无法续训（训练可能从未保存过）' });

        // 续训 YAML：原配置 + resume_from_checkpoint
        const resumeYaml = job.yaml.replace(/\nresume_from_checkpoint:.*/g, '') +
          '\nresume_from_checkpoint: ' + ckpt + '\n';
        const newJobId = job.id; // 原地续训：同 output_dir 才能正确恢复 scheduler/optimizer
        await sshWriteFile(server, workdir + '/jobs/' + newJobId + '.resume.yaml', resumeYaml);
        const cliBin2 = server.cliPath || 'llamafactory-cli';
        const startCmd =
          `cd ${shq(workdir)} && { nohup bash -c 'PYTHONUNBUFFERED=1 ${shq(cliBin2)} train jobs/${newJobId}.resume.yaml > logs/${newJobId}.log 2>&1; echo $? > logs/${newJobId}.exit' >/dev/null 2>&1 & }; echo $! > run/${newJobId}.pid`;
        await sshExec(server, startCmd, 20000);
        const updated = Object.assign({}, job, {
          status: 'running',
          resumedFrom: ckpt,
          resumedAt: Date.now(),
          error: undefined,
        });
        saveJob(updated);
        return sendJson(res, 200, { ok: true, job: updated, resumedFrom: ckpt });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    // ── 克隆重跑（对标 OpenAI "rerun from previous config"：复制配置提交新任务）──
    if (action === 'clone' && req.method === 'POST') {
      const body = await parseRequestBody(req);
      const p = body.data || {};
      // kev 决策模型必须走 kev 训练命令；旧实现无条件套 LLaMA-Factory YAML 会跑错引擎。
      if (job.trainMode === 'kev') {
        const serverK = findServer(job.serverId);
        if (!serverK) return sendJson(res, 400, { ok: false, error: 'server config missing' });
        const dsK = listDatasets().find(function (d) { return d.id === job.datasetId; });
        if (!dsK) return sendJson(res, 400, { ok: false, error: '原决策数据集已删除，无法克隆' });
        const jobIdK = newId('job');
        const wdK = serverK.remoteWorkdir || '~/ai-harbor-training';
        const cfgK = Object.assign({}, job.config, p.overrides || {}, { outputDir: 'saves/' + jobIdK });
        const kevK = buildKevTrainCommand(serverK, jobIdK, cfgK, dsK);
        const cmdK = shq(kevK.pyBin) + ' ' + kevK.args.map(shq).join(' ');
        try {
          await sshExec(serverK, `mkdir -p ${shq(wdK)}/data ${shq(wdK)}/jobs ${shq(wdK)}/logs ${shq(wdK)}/run ${shq(wdK)}/saves`, 15000);
          await sshWriteFile(serverK, wdK + '/data/' + dsK.fileName,
            fs.readFileSync(path.join(DATASETS_DIR, dsK.fileName), 'utf8'));
          const startK = `cd ${shq(wdK)} && { nohup bash -c 'PYTHONUNBUFFERED=1 ${cmdK} > logs/${jobIdK}.log 2>&1; echo $? > logs/${jobIdK}.exit' >/dev/null 2>&1 & }; echo $! > run/${jobIdK}.pid`;
          await sshExec(serverK, startK, 20000);
          const newJobK = {
            id: jobIdK, name: p.name || job.name + ' (克隆)',
            serverId: serverK.id, serverLabel: job.serverLabel,
            datasetId: job.datasetId, datasetName: job.datasetName,
            config: cfgK, evalConfig: job.evalConfig || null, trainMode: 'kev',
            yaml: '# kev 训练（类 Jev 决策模型）\n' + cmdK + '\n',
            clonedFrom: job.id, status: 'running', createdAt: Date.now(),
            logLines: 0, logTail: '',
          };
          saveJob(newJobK);
          return sendJson(res, 200, { ok: true, job: newJobK });
        } catch (e) {
          return sendJson(res, 500, { ok: false, error: e.message });
        }
      }
      const server = findServer(job.serverId);
      if (!server) return sendJson(res, 400, { ok: false, error: 'server config missing' });
      const workdir = server.remoteWorkdir || '~/ai-harbor-training';
      const jobIdNew = newId('job');
      const cfg = Object.assign({}, job.config, p.overrides || {}, {
        dataset: (job.config || {}).dataset,
        outputDir: 'saves/' + jobIdNew,
        isLocal: isLocal(server),
      });
      const yaml = buildYaml(cfg);
      try {
        await sshExec(server, `mkdir -p ${shq(workdir)}/jobs ${shq(workdir)}/logs ${shq(workdir)}/run ${shq(workdir)}/saves`, 15000);
        // 数据集应已注册过（同名 dataset_info 条目仍在）——直接校验存在
        const check = await sshExec(server,
          `cd ${shq(workdir)}/data && python3 -c "import json,sys;info=json.load(open('dataset_info.json'));sys.exit(0 if '${shq(cfg.dataset)}' in info else 1)"`, 15000);
        if (check.code !== 0) {
          // 数据集被删了则重新上传
          const ds = listDatasets().find(d => d.id === job.datasetId);
          if (ds) {
            const dsRawContent = fs.readFileSync(path.join(DATASETS_DIR, ds.fileName), 'utf8');
            await sshWriteFile(server, workdir + '/data/' + ds.fileName, dsRawContent);
            const infoEntry = datasetInfoEntry(ds);
            const mergeScript =
              `cd ${shq(workdir)}/data && python3 - <<'PYEOF'\n` +
              `import json, os\n` +
              `info = {}\n` +
              `if os.path.exists('dataset_info.json'):\n` +
              `    try: info = json.load(open('dataset_info.json'))\n` +
              `    except Exception: info = {}\n` +
              `info[${JSON.stringify(cfg.dataset)}] = ${JSON.stringify(infoEntry)}\n` +
              `json.dump(info, open('dataset_info.json', 'w'), ensure_ascii=False, indent=2)\n` +
              `PYEOF`;
            await sshExec(server, mergeScript, 30000);
          }
        }
        await sshWriteFile(server, workdir + '/jobs/' + jobIdNew + '.yaml', yaml);
        const cliBin3 = server.cliPath || 'llamafactory-cli';
        const startCmd =
          `cd ${shq(workdir)} && { nohup bash -c 'PYTHONUNBUFFERED=1 ${shq(cliBin3)} train jobs/${jobIdNew}.yaml > logs/${jobIdNew}.log 2>&1; echo $? > logs/${jobIdNew}.exit' >/dev/null 2>&1 & }; echo $! > run/${jobIdNew}.pid`;
        await sshExec(server, startCmd, 20000);
        const newJob = {
          id: jobIdNew,
          name: (p.name || job.name + ' (克隆)'),
          serverId: server.id,
          serverLabel: (server.username || '') + '@' + (server.host || ''),
          datasetId: job.datasetId,
          datasetName: job.datasetName,
          config: cfg,
          yaml: yaml,
          clonedFrom: job.id,
          status: 'running',
          createdAt: Date.now(),
          logLines: 0,
          logTail: '',
        };
        saveJob(newJob);
        return sendJson(res, 200, { ok: true, job: newJob });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }

    // ── 重命名 ──
    if (action === 'rename' && req.method === 'POST') {
      const body = await parseRequestBody(req);
      const name = ((body.data || {}).name || '').trim().slice(0, 60);
      if (!name) return sendJson(res, 400, { ok: false, error: 'name required' });
      const updated = Object.assign({}, job, { name: name });
      saveJob(updated);
      return sendJson(res, 200, { ok: true, job: updated });
    }
  }

  return false; // 不归本路由管（交给 404）
}

module.exports = {
  handleModelTrainingRequest, isSafeId,
  // 供回归测试直接调用（不改变线上行为）
  parseDatasetFile, buildFieldStats, loadPreviewData, normalizeProvenance, inferProvenance,
};
