/**
 * 统一 JSON 存储层：原子写 + 同文件串行化 + 损坏文件隔离。
 *
 * 为什么需要：平台此前各路由都用 fs.writeFileSync(file, JSON.stringify(...)) 直接覆盖。
 * 两个并发的 refresh/轮询会 lost update；进程在写入中途被杀会留下半截 JSON，
 * 而旧 readJson 的 catch 会把损坏文件静默当成空库，下一次写入就把历史数据抹掉。
 *
 * 做法：
 *   - writeJson：同目录写 .tmp-<pid>-<rand> → fsync → rename 覆盖（Windows 下 EPERM 时
 *     先删目标再重试一次）。同目录保证 rename 原子且不跨卷。
 *   - 串行队列：同一文件的写操作排队，避免交错。
 *   - updateJson：read → mutate → write 在一个队列里完成（真正的 read-modify-write 事务），
 *     用于 servers / metrics / reports 这类「读出来改几条再整体写回」的场景。
 *   - readJson：解析失败时把损坏文件另存为 .corrupt-<时间戳>（不删除，便于人工恢复），
 *     并返回 fallback + 显式 corrupt 标记，绝不假装文件是空的。
 *
 * 只保证单进程内的串行；多进程写同一文件仍需外部文件锁，本平台是单 Node 服务。
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const queues = new Map(); // file -> Promise chain

function enqueue(file, fn) {
  const key = path.resolve(file);
  const prev = queues.get(key) || Promise.resolve();
  const next = prev.catch(function () {}).then(fn);
  queues.set(key, next);
  next.then(function () { if (queues.get(key) === next) queues.delete(key); },
            function () { if (queues.get(key) === next) queues.delete(key); });
  return next;
}

/** 损坏文件的隔离备份：保留原始字节，避免「静默当空库 → 下次写入抹掉历史」 */
function quarantine(file, raw, err) {
  try {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const dest = file + '.corrupt-' + stamp + '-' + crypto.randomBytes(3).toString('hex');
    fs.writeFileSync(dest, raw, 'utf8');
    try {
      const dir = path.dirname(file);
      const log = path.join(dir, 'json-store-corrupt.log');
      fs.appendFileSync(log, JSON.stringify({
        at: new Date().toISOString(), file: path.basename(file), backup: path.basename(dest),
        error: String((err && err.message) || err).slice(0, 300)
      }) + '\n', 'utf8');
    } catch (e) { /* 日志写失败不影响主流程 */ }
    return dest;
  } catch (e) { return null; }
}

/** 同步读取（解析失败会隔离备份）。返回 { value, corrupt, backup } */
function readJsonDetailed(file, fallback) {
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); }
  catch (e) {
    if (e && (e.code === 'ENOENT' || e.code === 'ENOTDIR')) return { value: fallback, corrupt: false, backup: null };
    throw e;
  }
  if (!raw.trim()) return { value: fallback, corrupt: false, backup: null };  // 空文件按空库处理（初始化常见）
  try {
    return { value: JSON.parse(raw), corrupt: false, backup: null };
  } catch (e) {
    return { value: fallback, corrupt: true, backup: quarantine(file, raw, e) };
  }
}

/** 兼容旧调用签名：readJson(file, fallback) 直接返回值 */
function readJson(file, fallback) { return readJsonDetailed(file, fallback).value; }

function atomicWriteSync(file, text) {
  const dir = path.dirname(file);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, '.' + path.basename(file) + '.tmp-' + process.pid + '-' + crypto.randomBytes(4).toString('hex'));
  let fd = null;
  try {
    fd = fs.openSync(tmp, 'wx', 0o600);
    fs.writeFileSync(fd, text, 'utf8');
    fs.fsyncSync(fd);
    fs.closeSync(fd); fd = null;
    try { fs.renameSync(tmp, file); }
    catch (e) {
      // Windows 上目标被占用时偶发 EPERM/EACCES：删目标后重试一次，仍失败就抛出（不吞错误）
      if (e && (e.code === 'EPERM' || e.code === 'EACCES' || e.code === 'EEXIST')) {
        try { fs.unlinkSync(file); } catch (e2) { /* 不存在则忽略 */ }
        fs.renameSync(tmp, file);
      } else throw e;
    }
  } finally {
    if (fd != null) { try { fs.closeSync(fd); } catch (e) {} }
    try { if (fs.existsSync(tmp)) fs.unlinkSync(tmp); } catch (e) {}
  }
}

function serialize(obj) { return JSON.stringify(obj, null, 2); }

/** 异步原子写（同一文件排队）。返回值 Promise<void> */
function writeJson(file, obj) {
  const text = serialize(obj);
  return enqueue(file, function () { atomicWriteSync(file, text); });
}

/** 同步原子写：给已有大量同步调用路径直接使用，避免改造成 async 的连锁改动 */
function writeJsonSync(file, obj) { atomicWriteSync(file, serialize(obj)); }

/**
 * 读改写事务：mutator(value) 返回新值（或抛错中止，不写入）。
 * 同一个文件的所有 updateJson / writeJson 排队，避免 lost update。
 */
function updateJson(file, fallback, mutator) {
  return enqueue(file, function () {
    const cur = readJsonSyncRaw(file, fallback);
    const next = mutator(cur);
    if (next === undefined) return cur;   // mutator 选择不改动
    atomicWriteSync(file, serialize(next));
    return next;
  });
}

/** 同步读改写（同步路径用；仍走同一队列的串行保护由调用方保证，这里直接同步执行） */
function updateJsonSync(file, fallback, mutator) {
  const cur = readJsonSyncRaw(file, fallback);
  const next = mutator(cur);
  if (next === undefined) return cur;
  atomicWriteSync(file, serialize(next));
  return next;
}

function readJsonSyncRaw(file, fallback) {
  return readJsonDetailed(file, fallback).value;
}

module.exports = {
  readJson, readJsonDetailed, writeJson, writeJsonSync,
  updateJson, updateJsonSync, atomicWriteSync, enqueue, quarantine
};
