'use strict';
/**
 * Model-Builder — 模型训练工作台（独立版）
 *
 * 从 AI Harbor 分割出的独立服务：只含训练工作台（训练/数据集/评估/实验报告）
 * 及其依赖闭包，一个进程即可运行，不再背负 Harbor 的其他工具页。
 *
 * 启动：
 *   npm install
 *   npm start            （默认 http://127.0.0.1:3000）
 *   PORT=8080 npm start
 *
 * 环境变量（均可选）：
 *   HOST          监听地址（默认 127.0.0.1；对外部署务必先配鉴权）
 *   PORT          端口（默认 3000）
 *   QWEN_KEY 等   云 LLM 密钥（数据合成 / AI 报告用；见 config/api-keys.js）
 *
 * 本版本与 Harbor 主仓的差异：
 *   - 免登录（独立部署在个人机器/内网；如需公网暴露请自行加反向代理鉴权）
 *   - 训练执行依赖本机或 SSH 远端的 Python 训练环境（LLaMA-Factory / kev / rlcd-lite）
 */

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '127.0.0.1';

const { handleModelTrainingRequest } = require('./routes/model-training');
const { handleDatasetToolsRequest } = require('./routes/dataset-tools-router');
const { handleExperimentReportRequest } = require('./routes/experiment-report');

// ── 极简静态文件服务（tools/ 与前端页）──
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
};
function serveStatic(req, res) {
  const u = new URL(req.url, 'http://localhost');
  let p = path.normalize(u.pathname).replace(/^([/\\])+/, '');
  if (p === '' || p === '.') p = 'tools/model-training.html';   // 根路径直达工作台
  // 静态文件只允许 tools/ 内（白名单）：
  // .env（密钥）、data/（含 servers.json SSH 凭据与数据集）、源码与测试一律不可经 HTTP 读取。
  const rel = path.relative(path.join(__dirname, 'tools'), path.join(__dirname, p));
  if (p !== 'tools/model-training.html' && (rel.startsWith('..') || path.isAbsolute(rel))) {
    res.writeHead(403); return res.end('forbidden');
  }
  const file = path.join(__dirname, p);
  fs.readFile(file, function (err, buf) {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('404: ' + p); }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(buf);
  });
}

const server = http.createServer(async function (req, res) {
  const u = new URL(req.url, 'http://localhost');
  const pathname = u.pathname;
  try {
    if (pathname.startsWith('/api/model-training/')) return await handleModelTrainingRequest(req, res);
    if (pathname.startsWith('/api/dataset-tools/')) return await handleDatasetToolsRequest(req, res);
    if (pathname.startsWith('/api/experiment-report/')) return await handleExperimentReportRequest(req, res);
    return serveStatic(req, res);
  } catch (e) {
    console.error('[model-builder] unhandled:', e);
    if (!res.headersSent) { res.writeHead(500, { 'Content-Type': 'application/json' }); }
    try { res.end(JSON.stringify({ ok: false, error: String(e && e.message || e) })); } catch (_) {}
  }
});

// 训练命令动辄几十分钟：不限制请求时长，只约束首字节
server.headersTimeout = 65 * 1000;
server.requestTimeout = 0;

server.listen(PORT, HOST, () => {
  console.log(`[model-builder] listening on http://${HOST}:${PORT}`);
  console.log(`[model-builder] 工作台页面: http://${HOST}:${PORT}/`);
  if (HOST === '0.0.0.0') {
    console.warn('[model-builder] ⚠️ 监听全部网卡且本服务无鉴权——仅建议内网使用；公网部署请加反向代理鉴权');
  }
});

// 进程级兜底日志（定位静默退出）
process.on('uncaughtException', function (e) {
  console.error('[model-builder] uncaught:', e);
  try { fs.appendFileSync(path.join(__dirname, 'data', 'model-builder.log'), '[' + new Date().toISOString() + '] uncaught: ' + (e && e.stack || e) + '\n'); } catch (_) {}
});
process.on('unhandledRejection', function (r) {
  console.error('[model-builder] unhandled rejection:', r);
});
