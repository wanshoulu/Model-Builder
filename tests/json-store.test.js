/**
 * JSON 原子写存储回归测试：并发 lost update / 半写损坏 / 隔离备份。
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert');
const STORE = require('../routes/json-store');

const dir = path.join(__dirname, '..', 'data', 'model-training', '__test_json_store');
fs.mkdirSync(dir, { recursive: true });
const file = path.join(dir, 'counter.json');

let passed = 0, failed = 0;
async function check(name, fn) {
  try { await fn(); passed++; console.log('  PASS ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + ' → ' + e.message); }
}
function cleanup() {
  for (const f of fs.readdirSync(dir)) {
    try { fs.unlinkSync(path.join(dir, f)); } catch (e) {}
  }
  try { fs.rmdirSync(dir); } catch (e) {}
}

(async function () {
  console.log('=== 1. 原子写：写出的文件永远可解析，无 tmp 残留 ===');
  await check('writeJson 后可正常读回，且目录里没有临时文件', async function () {
    await STORE.writeJson(file, { a: 1, nested: { x: '中文' } });
    assert.deepStrictEqual(STORE.readJson(file, null), { a: 1, nested: { x: '中文' } });
    assert.ok(!fs.readdirSync(dir).some(f => f.includes('.tmp-')), '存在 tmp 残留');
  });

  console.log('\n=== 2. 并发读改写事务：50 次自增不丢更新 ===');
  await check('50 个并发 updateJson 后计数为 50', async function () {
    await STORE.writeJson(file, { n: 0 });
    await Promise.all(Array.from({ length: 50 }, function () {
      return STORE.updateJson(file, { n: 0 }, function (cur) { cur.n = (cur.n || 0) + 1; return cur; });
    }));
    assert.strictEqual(STORE.readJson(file, {}).n, 50);
  });
  await check('updateJson 抛错不会写坏文件，后续写入仍可用', async function () {
    try {
      await STORE.updateJson(file, {}, function () { throw new Error('boom'); });
      assert.fail('应当抛出');
    } catch (e) { assert.ok(/boom/.test(e.message)); }
    assert.strictEqual(STORE.readJson(file, {}).n, 50);
  });

  console.log('\n=== 3. 损坏文件：隔离备份 + 显式 corrupt 标记，绝不静默当空库 ===');
  await check('半截 JSON 被隔离为备份并写日志，readJsonDetailed 报 corrupt', function () {
    fs.writeFileSync(file, '{"n": 50, "broken": ');
    const r = STORE.readJsonDetailed(file, { n: -1 });
    assert.strictEqual(r.corrupt, true);
    assert.strictEqual(r.value.n, -1);
    assert.ok(r.backup && fs.existsSync(r.backup), '损坏文件应被隔离备份');
    assert.ok(fs.existsSync(path.join(dir, 'json-store-corrupt.log')));
    assert.strictEqual(fs.readFileSync(r.backup, 'utf8'), '{"n": 50, "broken": ');
  });
  await check('正常文件不触发 corrupt，备份不会污染主文件', async function () {
    await STORE.writeJson(file, { n: 51 });
    const r = STORE.readJsonDetailed(file, null);
    assert.strictEqual(r.corrupt, false);
    assert.strictEqual(r.value.n, 51);
  });

  console.log('\n=== 4. 目录不存在时自动创建 ===');
  await check('writeJson 自动 mkdir -p', async function () {
    const deep = path.join(dir, 'a', 'b', 'c.json');
    await STORE.writeJson(deep, { ok: true });
    assert.strictEqual(STORE.readJson(deep, null).ok, true);
  });

  cleanup();
  console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
  process.exit(failed ? 1 : 0);
})().catch(function (e) { console.error(e); cleanup(); process.exit(1); });
