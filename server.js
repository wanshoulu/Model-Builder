'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const { handleModelTrainingRequest } = require('./routes/model-training');
const { handleDatasetToolsRequest } = require('./routes/dataset-tools-router');
const { handleExperimentReportRequest } = require('./routes/experiment-report');

const MIME = {
  '.html': 'text/html;charset=utf-8',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

function sendJSON(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json;charset=utf-8',
    'Cache-Control': 'no-store, no-cache, must-revalidate',
  });
  res.end(JSON.stringify(data));
}

// ---------- 访问令牌 ----------
const DATA_DIR = path.join(__dirname, 'data', 'model-training');
const AUTH_FILE = path.join(DATA_DIR, 'auth.json');
let authToken = process.env.MT_TOKEN || '';

function loadOrCreateToken() {
  if (!authToken) {
    try {
      const saved = JSON.parse(fs.readFileSync(AUTH_FILE, 'utf8'));
      if (saved && typeof saved.token === 'string' && saved.token.length >= 16) {
        authToken = saved.token;
        console.log(`[ai-harbor] 访问令牌已加载自 ${AUTH_FILE}（遗忘可查看该文件，删除后重启将重新生成）`);
        return;
      }
    } catch (e) { /* 文件不存在则生成 */ }
    authToken = crypto.randomBytes(24).toString('base64url');
    try {
      fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(AUTH_FILE, JSON.stringify({ token: authToken, createdAt: new Date().toISOString() }, null, 2));
    } catch (e) {
      console.error('[ai-harbor] 令牌持久化失败（本次仅内存有效）:', e.message);
    }
    console.log('='.repeat(64));
    console.log('[ai-harbor] 首次启动已生成访问令牌: ' + authToken);
    console.log('[ai-harbor] 请在登录页输入该令牌；也可通过环境变量 MT_TOKEN 固定令牌');
    console.log('='.repeat(64));
  } else {
    console.log('[ai-harbor] 访问令牌来自环境变量 MT_TOKEN');
  }
}
loadOrCreateToken();

// ---------- 用户名 + 密码登录（2026-11-13：人性化登录层）----------
// MT_USERNAME / MT_PASSWORD 由 ai-harbor.env 提供；未配置时用户名默认 admin，
// 密码则复用 MT_TOKEN（保证升级后至少有一种方式能登录）。校验用 timingSafeEqual 防时序侧信道。
// 登录成功后由服务端下发 HttpOnly + SameSite=Lax cookie（防 XSS 窃取）。
let loginAttempts = [];  // 简单防爆破：5 分钟窗口内同一 IP 最多 10 次失败
function tooManyAttempts(ip) {
  const cutoff = Date.now() - 5 * 60 * 1000;
  loginAttempts = loginAttempts.filter(a => a.at > cutoff);
  return loginAttempts.filter(a => a.ip === ip).length >= 10;
}
function recordFailedAttempt(ip) { loginAttempts.push({ ip, at: Date.now() }); }

function safeEq(a, b) {
  const ba = Buffer.from(String(a || ''));
  const bb = Buffer.from(String(b || ''));
  if (ba.length !== bb.length) return false;
  return crypto.timingSafeEqual(ba, bb);
}
function checkCredentials(username, password) {
  const u = process.env.MT_USERNAME || 'admin';
  const p = process.env.MT_PASSWORD || authToken;   // 未设密码时复用访问令牌
  return safeEq(username, u) && safeEq(password, p);
}

function handleLogin(req, res, url) {
  let body = '';
  req.on('data', c => { body += c; if (body.length > 8192) req.destroy(); });
  req.on('end', () => {
    let username = '', password = '';
    try {
      const j = JSON.parse(body);
      username = String(j.username || '');
      password = String(j.password || '');
    } catch (e) { /* fallthrough */ }
    const ip = (req.headers['x-forwarded-for'] || req.socket.remoteAddress || '?').split(',')[0].trim();
    if (tooManyAttempts(ip)) {
      res.writeHead(429, { 'Content-Type': 'application/json;charset=utf-8' });
      res.end(JSON.stringify({ ok: false, error: '失败次数过多，请 5 分钟后重试' }));
      return;
    }
    if (checkCredentials(username, password)) {
      // 登录成功：HttpOnly cookie（JS 不可读，XSS 拿不到）；180 天免登
      res.writeHead(200, {
        'Content-Type': 'application/json;charset=utf-8',
        'Set-Cookie': 'mt_token=' + encodeURIComponent(authToken) + '; Path=/; Max-Age=15552000; HttpOnly; SameSite=Lax',
      });
      res.end(JSON.stringify({ ok: true }));
    } else {
      recordFailedAttempt(ip);
      res.writeHead(401, { 'Content-Type': 'application/json;charset=utf-8' });
      res.end(JSON.stringify({ ok: false, error: '用户名或密码错误' }));
    }
  });
}

function tokenEquals(candidate) {
  if (!candidate || typeof candidate !== 'string') return false;
  const a = Buffer.from(candidate);
  const b = Buffer.from(authToken);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

function extractToken(req, url) {
  const h = req.headers['x-auth-token'];
  if (typeof h === 'string' && h) return h;
  const authz = req.headers['authorization'];
  if (typeof authz === 'string' && authz.startsWith('Bearer ')) return authz.slice(7).trim();
  const cookie = req.headers.cookie || '';
  const m = /(?:^|;\s*)mt_token=([^;]+)/.exec(cookie);
  if (m) { try { return decodeURIComponent(m[1]); } catch (e) { return m[1]; } }
  const q = url.searchParams.get('token');
  if (q) return q;
  return '';
}

// 无需令牌即可访问的静态资源（登录页与其依赖）
function isPublicPath(p) {
  return p === '/tools/login.html' || p.startsWith('/tools/vendor/');
}

function serveStatic(req, res) {
  if (res.headersSent) return;
  let filePath = path.join(__dirname, req.pathname === '/' ? 'tools/model-training.html' : req.pathname);
  // 防目录穿越
  if (!filePath.startsWith(__dirname + path.sep)) {
    res.writeHead(403, { 'Content-Type': 'text/plain;charset=utf-8' });
    res.end('403 Forbidden');
    return;
  }
  const ext = path.extname(filePath);
  fs.readFile(filePath, (err, data) => {
    if (res.headersSent) return;
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain;charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
    const headers = { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'no-cache' };
    if (ext === '.html') headers['Cache-Control'] = 'no-cache, no-store, must-revalidate';
    res.writeHead(200, headers);
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://' + req.headers.host);
  req.pathname = url.pathname;

  if (req.pathname === '/') { res.writeHead(302, { Location: '/tools/model-training.html' }); res.end(); return; }
  if (req.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }

  // CORS preflight（预检不携带自定义头，放行）。
  // 收紧（2026-11-13）：不再回 Allow-Origin: * —— 与凭据 cookie 并存时任意网页
  // 都能带用户 cookie 打 API。仅回显同源请求的 Origin（跨站请求拿不到授权头）。
  if (req.method === 'OPTIONS') {
    const origin = req.headers.origin;
    const headers = {
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Auth-Token',
    };
    if (origin) {
      // 同源校验：Origin 的 host 必须与 Host 头一致（浏览器同源场景必然成立）
      try {
        const o = new URL(origin);
        if (req.headers.host && o.host === req.headers.host) {
          headers['Access-Control-Allow-Origin'] = origin;
          headers['Access-Control-Allow-Credentials'] = 'true';
          headers['Vary'] = 'Origin';
        }
      } catch (e) { /* 非法 Origin 不回授权头 */ }
    }
    res.writeHead(204, headers);
    res.end();
    return;
  }

  // ---- 用户名密码登录端点（无需令牌）----
  if (req.pathname === '/api/auth/login' && req.method === 'POST') {
    handleLogin(req, res, url);
    return;
  }

  // ---- 鉴权网关：除登录页/vendor 外全部需要令牌 ----
  if (!isPublicPath(req.pathname)) {
    const queryToken = url.searchParams.get('token');
    if (!tokenEquals(extractToken(req, url))) {
      if (req.pathname.startsWith('/api/')) {
        sendJSON(res, 401, { ok: false, error: '未授权：访问令牌缺失或无效' });
      } else {
        res.writeHead(302, { Location: '/tools/login.html?next=' + encodeURIComponent(req.pathname) });
        res.end();
      }
      return;
    }
    // 通过 ?token= 访问时下发 cookie（兼容自动化脚本）；HttpOnly 防脚本读取
    if (queryToken && tokenEquals(queryToken) && !req.headers.cookie) {
      res.setHeader('Set-Cookie', 'mt_token=' + encodeURIComponent(queryToken) + '; Path=/; Max-Age=15552000; HttpOnly; SameSite=Lax');
    }
  }

  // 数据集工具
  if (req.pathname === '/api/dataset-tools' || req.pathname.startsWith('/api/dataset-tools/')) {
    handleDatasetToolsRequest(req, res);
    return;
  }

  // 模型训练工作台
  if (req.pathname === '/api/model-training' || req.pathname.startsWith('/api/model-training/')) {
    handleModelTrainingRequest(req, res);
    return;
  }

  // 实验报告
  if (req.pathname === '/api/experiment-report' || req.pathname.startsWith('/api/experiment-report/')) {
    handleExperimentReportRequest(req, res);
    return;
  }

  // 静态文件
  serveStatic(req, res);
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`[ai-harbor] 模型训练工作台已启动: http://0.0.0.0:${PORT}（已启用令牌鉴权）`);
});
