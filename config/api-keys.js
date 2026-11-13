'use strict';

/**
 * API Keys — Single source of truth
 *
 * All AI model routes and keys are configured HERE.
 * Every tool (server, graph-generator, ppt-converter, etc.) imports from this file.
 *
 * Environment variables override dev defaults:
 *   DEEPSEEK_KEY  — DeepSeek API key
 *   QWEN_KEY      — Qwen / DashScope API key
 *   KIMI_KEY      — Kimi / Moonshot API key
 *   GLM_KEY       — Zhipu GLM coding plan API key
 *   VOLCANO_KEY   — Volcano (ByteDance) API key (format: id:secret)
 *
 * Optional web search providers used by tools/web-search-providers.js:
 *   WEB_SEARCH_PROVIDERS   — comma order, default: searxng,gdelt,brave,tavily,exa,serper,jina
 *   SEARXNG_URL            — free/self-hosted SearXNG endpoint, e.g. http://localhost:8888
 *   gdelt                  — free no-key news/public-media provider, configured by provider order only
 *   BRAVE_SEARCH_API_KEY   — Brave Search API
 *   JINA_API_KEY           — optional higher-rate Jina Search key; s.jina.ai can work without it
 *   TAVILY_API_KEY         — Tavily Search API
 *   EXA_API_KEY            — Exa Search API
 *   SERPER_API_KEY         — Serper Google SERP API
 *   llm_search_assist      — optional model-assisted lead generation; not a trusted retrieval source
 *
 * Dev defaults kick in when DEV_MODE=true or NODE_ENV=development.
 * In production, missing keys cause clear errors.
 */

// 密钥一律通过环境变量注入，本仓库不提交任何真实密钥（历史版本曾把开发密钥烤进
// DEV_KEYS——2026-11-13 已彻底移除；部署时用 --env-file 或 .env 注入，见 .env.example）
const DEV_MODE = process.env.DEV_MODE === 'true' || process.env.NODE_ENV === 'development' || !process.env.NODE_ENV;


const API_ROUTES = {
  'deepseek-chat': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-chat',
    getKey: () => process.env.DEEPSEEK_KEY ,
    provider: 'deepseek',
  },
  'deepseek-v4-pro': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-v4-pro',
    getKey: () => process.env.DEEPSEEK_KEY ,
    provider: 'deepseek',
  },
  'deepseek-v4-flash': {
    host: 'api.deepseek.com',
    path: '/v1/chat/completions',
    model: 'deepseek-v4-flash',
    getKey: () => process.env.DEEPSEEK_KEY ,
    provider: 'deepseek',
  },
  'qwen-plus': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/chat/completions',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
  },
  'qwen3.8-flash': {
    // 千问AI平台 MaaS 端点（与百炼同 key 体系）。2026-11 平台最新 flash，
    // 默认优先调用（PRIORITY 首位）。
    host: 'maas.qianwenaiapi.com',
    path: '/compatible-mode/v1/chat/completions',
    model: 'qwen3.8-flash',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
  },
  'qwen3.7-flash': {
    // 千问AI平台 MaaS 端点（与百炼同 key 体系）。思考开关同时接受
    // enable_thinking:boolean 与 thinking:{type:...} 两种方言，网关的
    // DeepSeek 风格透传可直接命中。
    host: 'maas.qianwenaiapi.com',
    path: '/compatible-mode/v1/chat/completions',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
  },
  'qwen-vl': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/chat/completions',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
  },
  'kimi-vl': {
    host: 'api.moonshot.cn',
    path: '/v1/chat/completions',
    model: 'kimi-k2.6',
    getKey: () => process.env.KIMI_KEY ,
    provider: 'kimi',
  },
  'kimi-latest': {
    host: 'api.moonshot.cn',
    path: '/v1/chat/completions',
    model: 'kimi-latest',
    getKey: () => process.env.KIMI_KEY ,
    provider: 'kimi',
  },
  // GLM Coding Plan（订阅制）— 注意必须用 coding 专属端点，通用端点会扣账户余额
  'glm-coding': {
    host: 'open.bigmodel.cn',
    path: '/api/coding/paas/v4/chat/completions',
    model: 'GLM-4.6',
    getKey: () => process.env.GLM_KEY ,
    provider: 'glm',
  },
  'glm-4.6': {
    host: 'open.bigmodel.cn',
    path: '/api/coding/paas/v4/chat/completions',
    model: 'GLM-4.6',
    getKey: () => process.env.GLM_KEY ,
    provider: 'glm',
  },
  'glm-5.2': {
    host: 'open.bigmodel.cn',
    path: '/api/paas/v4/chat/completions',
    model: 'glm-5.2',
    getKey: () => process.env.GLM_KEY ,
    provider: 'glm',
  },
  'text-embedding-v4': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/embeddings',
    model: 'qwen3.7-text-embedding-flash',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
    dim: 1024,
    batchSize: 10,
  },
  'qwen3.7-text-embedding-flash': {
    host: 'dashscope.aliyuncs.com',
    path: '/compatible-mode/v1/embeddings',
    model: 'qwen3.7-text-embedding-flash',
    getKey: () => process.env.QWEN_KEY ,
    provider: 'qwen',
    dim: 1024,
    batchSize: 10,
  },
  // Volcano (ByteDance) API — OpenAI compatible
  'doubao-pro-32k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-pro-32k',
    getKey: () => process.env.VOLCANO_KEY ,
    provider: 'volcano',
  },
  'doubao-pro-128k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-pro-128k',
    getKey: () => process.env.VOLCANO_KEY ,
    provider: 'volcano',
  },
  'doubao-lite-32k': {
    host: 'st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com',
    path: '/v1/chat/completions',
    model: 'doubao-lite-32k',
    getKey: () => process.env.VOLCANO_KEY ,
    provider: 'volcano',
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
  DEV_MODE,
  DEFAULT_CHAT_MODEL,
  getAvailableModels,
  getMissingKeys,
};
