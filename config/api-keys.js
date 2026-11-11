'use strict';

const fs = require('fs');

/**
 * API Keys — 模型路由注册表（单一事实来源）
 *
 * 所有 LLM 访问（tools/llm-client.js、services/intent-draft-trial.js）都从这里取路由与密钥。
 *
 * 密钥一律通过环境变量注入，本仓库不提交任何真实密钥：
 *   DEEPSEEK_KEY  — DeepSeek API key
 *   QWEN_KEY      — Qwen / DashScope API key
 *   KIMI_KEY      — Kimi / Moonshot API key
 *   GLM_KEY       — Zhipu GLM API key
 *   VOLCANO_KEY   — Volcano (ByteDance) API key (format: id:secret)
 *
 * 配置方式：复制 .env.example → .env（或直接 export），按需填入至少一个 key。
 * 未配置任何 key 时服务可启动，但 AI 辅助功能（AI 映射/抽取/试算）会在调用时
 * 返回明确的 MISSING_KEY 错误；模板校验、草稿管理、应用/回滚等离线能力不受影响。
 */

const API_ROUTES = {
  'deepseek-chat': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-chat',
    getKey: () => process.env.DEEPSEEK_KEY,
    provider: 'deepseek',
  },
  'deepseek-v4-pro': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-v4-pro',
    getKey: () => process.env.DEEPSEEK_KEY,
    provider: 'deepseek',
  },
  'deepseek-v4-flash': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-v4-flash',
    getKey: () => process.env.DEEPSEEK_KEY,
    provider: 'deepseek',
  },
  'qwen-plus': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/chat/completions',
    getKey: () => process.env.QWEN_KEY,
    provider: 'qwen',
  },
  'qwen-vl': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/chat/completions',
    getKey: () => process.env.QWEN_KEY,
    provider: 'qwen',
  },
  'qwen3.7-flash': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/chat/completions',
    model: 'qwen3.7-flash',
    getKey: () => process.env.QWEN_KEY,
    provider: 'qwen',
  },
  'kimi-vl': {
    host: 'api.moonshot.cn',
    path: '/v1/chat/completions',
    model: 'kimi-k2.6',
    getKey: () => process.env.KIMI_KEY,
    provider: 'kimi',
  },
  'kimi-latest': {
    host: 'api.moonshot.cn',
    path: '/v1/chat/completions',
    model: 'kimi-latest',
    getKey: () => process.env.KIMI_KEY,
    provider: 'kimi',
  },
  // GLM（订阅制）— 注意必须用 coding 专属端点，通用端点会扣账户余额
  'glm-coding': {
    host: 'open.bigmodel.cn',
    path: '/api/coding/paas/v4/chat/completions',
    model: 'GLM-4.6',
    getKey: () => process.env.GLM_KEY,
    provider: 'glm',
  },
  'glm-4.6': {
    host: 'open.bigmodel.cn',
    path: '/api/paas/v4/chat/completions',
    model: 'GLM-4.6',
    getKey: () => process.env.GLM_KEY,
    provider: 'glm',
  },
  'glm-5.2': {
    host: 'open.bigmodel.cn',
    path: '/api/paas/v4/chat/completions',
    model: 'glm-5.2',
    getKey: () => process.env.GLM_KEY,
    provider: 'glm',
  },
  'text-embedding-v4': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/embeddings',
    model: 'qwen3.7-text-embedding-flash',
    getKey: () => process.env.QWEN_KEY,
    provider: 'qwen',
    dim: 1024,
    batchSize: 10,
  },
  'qwen3.7-text-embedding-flash': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/embeddings',
    model: 'qwen3.7-text-embedding-flash',
    getKey: () => process.env.QWEN_KEY,
    provider: 'qwen',
    dim: 1024,
    batchSize: 10,
  },
  // Volcano (ByteDance) API — OpenAI compatible
  'doubao-pro-32k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-pro-32k',
    getKey: () => process.env.VOLCANO_KEY,
    provider: 'volcano',
  },
  'doubao-pro-128k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-pro-128k',
    getKey: () => process.env.VOLCANO_KEY,
    provider: 'volcano',
  },
  'doubao-lite-32k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-lite-32k',
    getKey: () => process.env.VOLCANO_KEY,
    provider: 'volcano',
  },
  // 内部模型网关 — Docker / 生产环境使用，通过 ai-harbor 模型网关调用 LLM
  // 容器内 dashscope.aliyuncs.com 不可达（TLS 断开），需走内部 HTTP 网关
  'internal-gateway-chat': {
    host: process.env.AI_HARBOR_INTERNAL_URL || 'http://localhost:3000',
    path: '/api/ai-bank/v1/internal/model/v1/chat/completions',
    model: 'qwen3.7-flash',
    getKey: () => {
      const token = process.env.AI_BANK_AGENT_RUNTIME_TOKEN;
      if (token) return token;
      const tokenFile = process.env.AI_BANK_AGENT_RUNTIME_TOKEN_FILE;
      if (tokenFile) {
        try { return fs.readFileSync(tokenFile, 'utf-8').trim(); } catch (e) { /* fallthrough */ }
      }
      return process.env.QWEN_KEY || '';
    },
    provider: 'internal-gateway',
  },
  'internal-gateway-embedding': {
    host: process.env.AI_HARBOR_INTERNAL_URL || 'http://localhost:3000',
    path: '/api/ai-bank/v1/internal/model/v1/embeddings',
    model: 'qwen3.7-text-embedding-flash',
    getKey: () => {
      const token = process.env.AI_BANK_AGENT_RUNTIME_TOKEN;
      if (token) return token;
      const tokenFile = process.env.AI_BANK_AGENT_RUNTIME_TOKEN_FILE;
      if (tokenFile) {
        try { return fs.readFileSync(tokenFile, 'utf-8').trim(); } catch (e) { /* fallthrough */ }
      }
      return process.env.QWEN_KEY || '';
    },
    provider: 'internal-gateway',
    dim: 1024,
    batchSize: 10,
  },
};

/** Return array of model IDs that have a usable key */
function getAvailableModels() {
  return Object.keys(API_ROUTES).filter(m => API_ROUTES[m].getKey());
}

/** Check which keys are missing (for startup logging) */
function getMissingKeys() {
  const missing = [];
  if (!process.env.DEEPSEEK_KEY) missing.push('DEEPSEEK_KEY');
  if (!process.env.QWEN_KEY) missing.push('QWEN_KEY');
  if (!process.env.KIMI_KEY) missing.push('KIMI_KEY');
  if (!process.env.GLM_KEY) missing.push('GLM_KEY');
  if (!process.env.VOLCANO_KEY) missing.push('VOLCANO_KEY');
  return missing;
}

/** 知识管线默认对话模型 — 优先使用 DeepSeek V4 Pro */
const DEFAULT_CHAT_MODEL = 'deepseek-v4-pro';

module.exports = {
  API_ROUTES,
  DEFAULT_CHAT_MODEL,
  getAvailableModels,
  getMissingKeys,
};
