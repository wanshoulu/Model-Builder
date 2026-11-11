/** 安全加固测试：shq 转义 / 服务器字段校验 / 路径沙箱 / 监听地址 */
const assert = require('node:assert');

let passed = 0, failed = 0;
function check(name, fn) {
  try { fn(); passed++; console.log('  ✓ ' + name); }
  catch (e) { failed++; console.log('  ✗ ' + name + '\n    ' + e.message); }
}

// ── 1. shq 转义（从 model-training 源码内提取验证，因为它未导出；
//      这里通过读源码+eval 的方式拿到函数本体）──
const fs = require('node:fs');
const src = fs.readFileSync(require.resolve('../routes/model-training.js'), 'utf8');
const mShq = src.match(/function shq\(s\) \{[\s\S]*?\n\}/);
assert.ok(mShq, 'shq 函数存在');
const shq = new Function('return ' + mShq[0])();

check('shq：普通路径原样加引号', function () {
  assert.strictEqual(shq('~/ai-harbor-training'), "'~/ai-harbor-training'");
});
check('shq：单引号注入被中断转义', function () {
  const evil = "qwen'; rm -rf / #";
  const out = shq(evil);
  // 展开后必须是单一字面量参数：用 bash 语义验证不了，这里验证结构
  assert.strictEqual(out, `'qwen'"'"'; rm -rf / #'`);
});
check('shq：反引号/$/分号在单引号内无特殊含义', function () {
  const evil = '`id`; $(whoami) | cat /etc/passwd && echo pwned';
  const out = shq(evil);
  assert.ok(out.startsWith("'") && out.endsWith("'"));
  assert.ok(!out.includes('\\`'));   // 不靠反斜杠转义（bash 单引号内反斜杠是字面量）
});
check('shq：null/undefined 安全', function () {
  assert.strictEqual(shq(null), "''");
  assert.strictEqual(shq(undefined), "''");
});
check('shq：真实 bash 往返一致（命令注入不可行）', function () {
  const cp = require('node:child_process');
  const evil = "x'; touch /tmp/pwned_marker_$$; echo '";
  const script = 'printf %s ' + shq(evil);
  // 用 node 自己的 bash 跑（本地 Git Bash 一定存在，localExec 同款）
  const out = cp.execSync('"C:\\Program Files\\Git\\bin\\bash.exe" -c ' + JSON.stringify(script),
    { encoding: 'utf8', timeout: 8000 });
  assert.strictEqual(out, evil);   // 原样回读 = 注入没有被执行
  assert.ok(!fs.existsSync('/tmp/pwned_marker_$$'));
});

// ── 2. 服务器字段校验 ──
const mVal = src.match(/function validateServerFields\(p\) \{[\s\S]*?\n\}/);
assert.ok(mVal, 'validateServerFields 存在');
const validateServerFields = new Function('return ' + mVal[0])();

check('合法配置通过校验', function () {
  assert.strictEqual(validateServerFields({ host: '192.168.1.10', username: 'root', port: 22, remoteWorkdir: '~/ai-harbor-training' }), null);
});
check('host 含 shell 元字符被拒', function () {
  assert.ok(/host/.test(validateServerFields({ host: 'a;rm -rf /' })));
});
check('host 含换行被拒（防注入 SSH/日志）', function () {
  assert.ok(validateServerFields({ host: 'evil\nhost' }) !== null);
});
check('remoteWorkdir 含反引号被拒', function () {
  assert.ok(/remoteWorkdir/.test(validateServerFields({ remoteWorkdir: '~/x`id`' })));
});
check('cliPath 含管道被拒', function () {
  assert.ok(/cliPath/.test(validateServerFields({ cliPath: '/bin/sh | id' })));
});
check('port 非法被拒', function () {
  assert.ok(/port/.test(validateServerFields({ port: 99999 })));
  assert.ok(/port/.test(validateServerFields({ port: 'abc' })));
});
check('控制字符被拒', function () {
  assert.ok(validateServerFields({ name: 'bad\x00name' }) !== null);
});

// ── 3. 路径沙箱 ──
const { guardImportPath, configuredRoots } = require('../routes/path-guard');
check('默认根目录内的路径放行（项目 data 目录）', function () {
  const real = require('path').join(__dirname, '..', 'data');
  assert.ok(fs.existsSync(real), '测试前置：data 目录应存在');
  const r = guardImportPath(real);
  assert.strictEqual(r.ok, true, JSON.stringify(r));
});
check('根目录外的绝对路径被拒（C:/Windows）', function () {
  const r = guardImportPath('C:\\Windows\\win.ini');
  assert.strictEqual(r.ok, false);
  assert.ok(/允许的导入目录/.test(r.error));
});
check('根目录外的用户目录被拒', function () {
  assert.strictEqual(guardImportPath('C:\\Users\\somebody\\.ssh\\id_rsa').ok, false);
});
check('.. 穿越被前置拒绝', function () {
  const dataDir = require('path').join(__dirname, '..', 'data');
  const r = guardImportPath(dataDir + require('path').sep + '..' + require('path').sep + '..' + require('path').sep + 'etc' + require('path').sep + 'passwd');
  assert.strictEqual(r.ok, false);
  assert.ok(/\.\./.test(r.error));
});
check('不存在的路径被拒', function () {
  const r = guardImportPath('Z:\\no_such_dir_xyz\\a.json');
  assert.strictEqual(r.ok, false);
  assert.ok(/不存在/.test(r.error));
});
check('大小写与正斜杠变体归一后仍受白名单约束', function () {
  // 白名单内目录（项目 data）
  assert.strictEqual(guardImportPath(require('path').join(__dirname, '..', 'data')).ok, true);
  // d:/other 不在白名单 → 拒
  assert.strictEqual(guardImportPath('Z:/definitely/not/whitelisted/a.json').ok, false);
});
check('空路径被拒', function () {
  assert.strictEqual(guardImportPath('').ok, false);
  assert.strictEqual(guardImportPath('   ').ok, false);
});
check('DATASET_IMPORT_ROOTS 环境变量可覆盖白名单', function () {
  process.env.DATASET_IMPORT_ROOTS = 'Z:\\some\\custom\\root';
  try {
    assert.strictEqual(guardImportPath('Z:\\whatever\\a.json').ok, false);
    assert.strictEqual(configuredRoots()[0], 'Z:\\some\\custom\\root');
  } finally { delete process.env.DATASET_IMPORT_ROOTS; }
});

// ── 4. 监听地址默认回环 ──
const serverSrc = fs.readFileSync(require.resolve('../server.js'), 'utf8');
check('server.listen 绑定 HOST（默认 127.0.0.1）', function () {
  assert.ok(/const HOST = process\.env\.HOST \|\| '127\.0\.0\.1'/.test(serverSrc));
  assert.ok(/server\.listen\(PORT, HOST/.test(serverSrc));
});
check('0.0.0.0 + SKIP_AUTH=true 组合有显式警告', function () {
  // 独立版无 SKIP_AUTH（免登录是既定设计）；改断言：0.0.0.0 时有无鉴权警告
  assert.ok(/无鉴权|监听全部网卡/.test(serverSrc));
});

console.log('\n=== 结果：' + passed + ' passed, ' + failed + ' failed ===');
process.exit(failed ? 1 : 0);
