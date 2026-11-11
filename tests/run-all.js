// 测试启动器：依次跑全部测试套件，汇总通过数
const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');

const suites = fs.readdirSync(__dirname)
  .filter(f => f.endsWith('.test.js'))
  .sort();

let totalPass = 0, totalFail = 0, failedSuites = [];
for (const s of suites) {
  console.log('\n════ ' + s + ' ════');
  const out = cp.spawnSync('node', [path.join(__dirname, s)], { encoding: 'utf8', timeout: 120000 });
  process.stdout.write(out.stdout || '');
  if (out.stderr) process.stderr.write(out.stderr);
  const m = (out.stdout || '').match(/(\d+) passed, (\d+) failed/);
  if (m) {
    totalPass += parseInt(m[1], 10);
    totalFail += parseInt(m[2], 10);
    if (parseInt(m[2], 10) > 0) failedSuites.push(s);
  } else {
    totalFail += 1;
    failedSuites.push(s + ' (no summary)');
  }
}
console.log('\n══════════════════════════════');
console.log(`总计: ${totalPass} passed, ${totalFail} failed (${suites.length} 套件)`);
if (failedSuites.length) {
  console.log('失败套件:', failedSuites.join(', '));
  process.exit(1);
}
