/**
 * 路径沙箱：本地导入类端点（import-path / read-doc）共用。
 *
 * 背景：这两个端点原本接受任意绝对路径，等价于把整台机器的文件系统
 * 暴露给任何能访问 API 的人（服务历史上默认 SKIP_AUTH=true + 0.0.0.0）。
 * 收紧方式：只允许「允许根目录」内的路径，realpath 解析后二次校验，
 * 符号链接 / .. / 大小写变体都逃不出去。
 *
 * 允许根目录来自环境变量 DATASET_IMPORT_ROOTS（分号分隔，Windows 路径
 * 可含空格），默认覆盖训练工作目录与项目数据目录——正常数据集导入不受影响。
 */
const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_ROOTS = [path.join(__dirname, '..', 'data')]
  .concat(process.env.TRAINING_ROOT ? [process.env.TRAINING_ROOT] : []);  // TRAINING_ROOT 未配置则只含项目 data

function configuredRoots() {
  const raw = (process.env.DATASET_IMPORT_ROOTS || '').trim();
  if (!raw) return DEFAULT_ROOTS;
  return raw.split(';').map(function (s) { return s.trim(); }).filter(Boolean);
}

function realpathSafe(p) {
  try { return fs.realpathSync(p); } catch (e) { return null; }
}

/**
 * 校验 rawPath 是否落在允许根目录内。
 * 返回 { ok: true, abs, roots } 或 { ok: false, error }。
 * 空白名单（DATASET_IMPORT_ROOTS="" 且未配默认）= 完全禁用导入。
 */
function guardImportPath(rawPath) {
  const s = String(rawPath || '').trim();
  if (!s) return { ok: false, error: 'path 必填' };
  // 拒绝明显穿越意图（即使 resolve 能消掉 ..，也先挡一层，日志更清晰）
  if (/(^|[\\\/])\.\.($|[\\\/])/.test(s)) return { ok: false, error: '路径不允许包含 ..' };

  const roots = configuredRoots().map(function (r) {
    const rp = realpathSafe(r) || path.resolve(r);
    return path.resolve(rp).toLowerCase();
  });
  if (!roots.length) return { ok: false, error: '路径导入已禁用（未配置 DATASET_IMPORT_ROOTS）' };

  const abs = path.resolve(s);
  const real = realpathSafe(abs);
  if (!real) return { ok: false, error: '路径不存在：' + abs };

  const realLc = path.resolve(real).toLowerCase();
  const inside = roots.some(function (root) {
    return realLc === root || realLc.startsWith(root + path.sep);
  });
  if (!inside) {
    return {
      ok: false,
      error: '路径不在允许的导入目录内。允许的根目录（可用环境变量 DATASET_IMPORT_ROOTS 调整，分号分隔）：'
        + configuredRoots().join(' ; ')
    };
  }
  return { ok: true, abs: real, roots: configuredRoots() };
}

module.exports = { guardImportPath, configuredRoots, DEFAULT_ROOTS };
