# Model-Builder 模型训练工作台

一句话：本机/SSH 远端 GPU 上完成「数据集合成 → 训练 → 留出集评估 → 实验报告」全流程的独立工作台，支持 LLaMA-Factory（SFT/LoRA/偏好）、kev（类 Jev 决策模型）、RLCD（GRPO + RLCR 校准奖励）三类训练引擎。

> 从 AI Harbor 分割出的独立版：一个 Node 进程 + 一个前端页 + 训练机上的 Python 环境，不再依赖 Harbor 的其他模块。

## 它解决什么问题

跑对照实验（如「LoRA vs RLCD 谁的意图识别更准」）时，散落的脚本和数据有四个坑：

| 坑 | 本工具的做法 |
|---|---|
| 训练只产 loss 曲线，精度靠手抄 | 训练完成自动在留出集推理打分（accuracy/F1/ECE/Brier + 逐样本留痕），指标带来源标签自动进报告 |
| 合成数据无质检等于投毒 | 7 段质检漏斗（去重/长度/语言/语义去重/结构/评分/LLM 裁判）+ 评测集泄漏隔离 |
| 两个引擎各一套口径，数字不可比 | 统一评估协议：Wilson + record-level bootstrap 双置信区间、平台自算与引擎官方数字交叉校验 |
| 结果散在聊天记录里 | 实验报告页：Plotly 本地图表（分组曲线/直方图/雷达/平行坐标）+ AI 叙述（事实包引用回锚，数字白名单校验） |

## 2 分钟跑起来

```bash
git clone https://github.com/wanshoulu/Model-Builder
cd Model-Builder
npm install          # 2 个依赖：formidable, ssh2
npm start            # http://127.0.0.1:3000
```

浏览器打开后看到的就是工作台首页（训练任务列表 + 新建训练任务向导）。

训练执行需要一台有 Python 训练环境的机器（本机或 SSH 远端）：

- **LLaMA-Factory**（SFT/LoRA/DPO 路线）：`llamafactory-cli` 可用
- **kev / RLCD**（决策模型路线）：`kev` 包 + `rlcd-lite`（Apache-2.0，github.com/arnabgho/rlcd-lite）+ 训练底座模型（如 Qwen3-0.6B）

在「训练服务器」页签把 `python` / `llamafactory-cli` 路径填进去即可（本机模式直接用本机路径）。

## 名词表（先名词后机制）

| 术语 | 含义 |
|---|---|
| LoRA | 低秩适配微调：冻结底座只训小矩阵，显存友好 |
| kev | 类 Jev 决策模型训练框架：监督 CE + 指针头读出选项 |
| RLCD | 校准决策的强化学习：GRPO + RLCR 奖励，训出的模型「说多准就有多准」 |
| RLCR 奖励 | `正确性 − (置信度−正确性)²`：兼顾答对与置信度诚实 |
| 留出集（held-out） | 训练时从未见过的评估数据；与训练集重复叫泄漏 |
| 准确率 Accuracy | 预测正确的比例 |
| 宏平均 F1（Macro-F1） | 每个类别等权平均的 F1，少数类与多数类同样重要 |
| ECE | 期望校准误差：自报置信度与真实正确率的偏差，越小越诚实 |
| Brier 评分 | 预测概率与实际结果的均方差，概率质量综合指标 |
| Wilson CI / bootstrap CI | 置信区间的两种算法；一条记录多个问题时 bootstrap 更严谨 |

## 功能地图

```
首页        训练任务列表 · 统计卡 · 新建任务向导（4 步分组配置 + 兼容性硬校验）
训练服务器   本机/SSH 远端配置（凭据脱敏存储）· GPU/环境探测
数据集       导入（文件/夹/路径）· 质检漏斗 · 转换（8 种布局互转）· 云 LLM 合成
            （文档驱动/Persona/Evol/按标签扩写/决策合成/规则映射/偏好对）
评估        训练后自动评估 · 3 引擎（kev/LLaMA-Factory/import）· 校准曲线大图
            · 泄漏检测 · record-level bootstrap CI
实验报告     运行对比 · Plotly 图表（曲线/直方图/雷达/平行坐标）
            · 统计检验（t 检验/Cohen's d）· AI 综合叙述（引用回锚校验）
```

## 已验证的用法（真实数据）

- Qwen3-0.6B + 银行 QA LoRA（A1000 6GB）
- kev 决策模型训练 + kev.benchmark 评估（平台自算 acc 与官方逐位一致）
- RLCD（rlcd-lite 基线）：GRPO 75 步 + RLCR 奖励，Brier 比监督 LoRA 低约一半
- 建行手机银行 378 意图：637 种子云 LLM 扩写 2342 条双格式数据集（见下）

## 限制（如实声明）

- 单机单用户设计：无多租户/鉴权；公网部署需自行加反向代理鉴权
- 本机训练执行链路在 Windows（Git Bash）/典型 Linux 下实测；SSH 远端未真机验证过
- 378 类这种高基数分类对 0.8B 底座是边界场景——决策臂用域路由（10 选项）而非裸 378-way
- 大数据集（>200MB）会整读进内存；任务列表默认全量返回（可带 `?limit=&offset=` 分页）

## 仓库结构

```
server.js                 独立服务入口（静态 + 3 组 API 路由）
routes/model-training.js  训练任务：提交/进度/曲线/断点续训/克隆/进程树击杀/产物清理
routes/model-evaluation.js评估编排：3 引擎 · 指标回写 · 泄漏检测 · bootstrap CI
routes/eval-metrics.js    统计内核：accuracy/F1/EM/token-F1/Wilson/bootstrap
routes/dataset-tools*.js  数据集：嗅探/质检漏斗/转换/合成（5+2 模式）
routes/experiment-report*.js 实验报告：聚合/统计检验/AI 叙述（事实包+回锚）
tools/model-training.html 单页前端（Plotly 本地化，无外网依赖）
tools/vendor/plotly.min.js Plotly.js（MIT，见 PLOTLY-LICENSE.txt）
tests/                    12 套 209 项回归测试（npm test）
```

## License

MIT（Plotly.js 以 MIT 随仓库分发；rlcd-lite/kev 训练环境为 Apache-2.0，需另行安装）
