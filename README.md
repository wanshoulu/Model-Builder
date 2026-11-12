<p align="center">
<img src="docs/images/hero-dashboard.png" alt="Model-Builder 工作台首页" width="880">
</p>

# Model-Builder

从数据集合成到训练任务、自动评估、实验对比报告的一站式本地工作台。单进程 Node 服务 + 单页前端，两个 npm 依赖即可跑起来。

> [!TIP]
> 界面截图全部来自真实运行（`docs/images/`）。

## 是什么

面向个人与小团队的**本地大模型训练全流程工具**。它把训练 LLaMA-Factory / kev / RLCD 作业时的手工环节做成了界面：造数据（合成 + 七段质检漏斗）→ 提交训练（进程树管理）→ 训练完自动评估（Wilson + record-level bootstrap 双置信区间）→ 多任务实验对比（Plotly 图表 + 可选 AI 叙述）。

不依赖云服务：训练跑在本机或 SSH 远端（如一台有 GPU 的 Linux 机器），图表用本地化的 Plotly（无 CDN），密钥只进环境变量。

## 它解决什么问题

跑对照实验（如「LoRA vs RLCD 谁的意图识别更准」）时，散落的脚本和数据有四个坑：

| 坑 | 本工具的做法 |
|---|---|
| 训练只产 loss 曲线，精度靠手抄 | 训练完成自动在留出集推理打分（accuracy/F1/ECE/Brier + 逐样本留痕），指标带来源标签自动进报告 |
| 合成数据无质检等于投毒 | 7 段质检漏斗 + 评测集泄漏隔离 |
| 两个引擎各一套口径，数字不可比 | 统一评估协议：Wilson + record-level bootstrap 双置信区间，平台自算与引擎官方数字交叉校验 |
| 结果散在聊天记录里 | 实验报告页：Plotly 本地图表（分组曲线/直方图/雷达/平行坐标）+ AI 叙述（事实包引用回锚，数字白名单校验） |

## 特性

- **7 段质检漏斗**：空值 / 布局归一 / 精确去重 / 语义去重（3-gram Jaccard）/ 长度截断 / 语言过滤 / 评测集泄漏隔离，每段独立计数
- **7 种合成模式**：evol / rewrite / label-variant（按标签扩写）/ rule-map（零 LLM 成本）/ decision few-shot 等，支持种子注入与「禁止编造」约束
- **8 种数据布局**：alpaca / sharegpt / openai_messages / qa_pairs / dpo_standard / alpaca_preference / pretrain_text / decision_kev，互转 fail-closed（二次归一化直接报错）
- **三训练引擎**：LLaMA-Factory（yaml 生成 + 断点续训 + 克隆）、kev benchmark、RLCD-GRPO（rlcd-lite）
- **进程树击杀**：Windows / Linux 下递归杀子进程不留孤儿；任务删除连带清理 data/ logs/ saves/ 全部产物与评估临时件
- **自动评估**：训练完成即跑留出集，accuracy / macro-F1 / EM / token-F1 / ROUGE-L / 校准（ECE + Brier），指标回写任务并进实验报告
- **双置信区间**：Wilson（快）与 record-level bootstrap（1000 次按记录重采样，处理多问题记录的单元相关性）
- **实验报告**：多任务指标聚合 + 统计检验 + Plotly 本地图表；AI 叙述仅引用事实包内数字并回锚到图表
- **路径沙箱**：数据导入白名单（`DATASET_IMPORT_ROOTS`），realpath 二次校验，`..` / 符号链接 / 大小写变体均逃不出去

## 快速开始

```bash
git clone https://github.com/wanshoulu/Model-Builder.git
cd Model-Builder
npm install && npm start     # → http://127.0.0.1:3000
```

打开浏览器就是工作台首页（任务列表 + 新建训练任务向导）。训练执行需要一台 Python 训练环境（本机或 SSH 远端），全部环境变量见 `.env.example`。

## 训练环境准备（提交真实训练才需要）

工作台本体是 Node 单进程，只做编排；真正跑 LoRA / GRPO 的是 `training/` 里的两个 Python 脚本（`train_rlcd.py`、`kev_rlcd_bridge.py`）。仓库已自带它们，第三方引擎、模型权重与 venv 需自行安装：

```bash
cd training
git clone --depth 1 https://github.com/arnabgho/rlcd-lite.git rlcd-lite
python -m venv venv && venv/bin/pip install -r requirements.txt
```

然后在项目根目录的 `.env` 里设 `TRAINING_ROOT=./training`。三个训练引擎（LLaMA-Factory / kev / rlcd-lite）各自的安装项、目录约定与已知限制见 [training/README.md](training/README.md)。只想体验数据集合成、质检与实验报告，不需要装这些。

## 渐进式使用

**第一步：体验界面与质检（不需要训练环境）**

把一份 alpaca 格式数据粘贴进「数据集」页签导入，质检漏斗立即给出 7 段剔除计数与通过率：

<p align="center">
<img src="docs/images/tab-datasets.png" alt="数据集管理" width="760">
</p>

**第二步：接上训练环境**

「训练服务器」页签注册本机或 SSH 远端（workdir / cli 路径 / 凭据）后，「新建训练任务」向导即可选数据集与模型提交。任务列表实时刷新，点「详情」看 Loss 曲线与产物：

<p align="center">
<img src="docs/images/job-detail.png" alt="任务详情" width="760">
</p>

**第三步：训练后评估与实验对比**

建任务时勾选「训练后评估」与评估集，完成后自动打分并回写；「实验报告」页签勾选多个任务生成对比图表：

<p align="center">
<img src="docs/images/tab-report.png" alt="实验报告" width="760">
</p>

## 架构

```mermaid
flowchart LR
    UI[单页前端 tools/model-training.html<br/>Plotly 本地化] -->|/api/model-training| MT[routes/model-training.js<br/>任务编排 · 进程树 · SSH]
    UI -->|/api/dataset-tools| DT[routes/dataset-tools*.js<br/>嗅探 · 漏斗 · 转换 · 合成]
    UI -->|/api/experiment-report| ER[routes/experiment-report*.js<br/>聚合 · 统计检验 · AI 叙述]
    MT --> EV[routes/model-evaluation.js<br/>三引擎评估编排]
    EV --> EM[eval-metrics.js<br/>Wilson · bootstrap · 泄漏检测]
    MT <-->|SSH / 本机| TR[训练环境<br/>LLaMA-Factory · kev · rlcd-lite]
    DT --> PG[path-guard.js<br/>导入路径沙箱]
```

## 名词表

| 名词 | 含义 |
|---|---|
| SFT | Supervised Fine-Tuning，指令微调（alpaca 格式 instruction/output） |
| decision 数据 | 决策格式样本：`state`（用户话语）+ `questions`（追问选项树），kev 引擎的多问题输出用 |
| RLCD | RL for Calibrated Decisions，面向校准决策的强化学习（[rlcd-lite](https://github.com/arnabgho/rlcd-lite) 实现）。rlcd-lite 官方声明：它既不是 TypeSafe 未公开的 RLCD 算法，也与 Yang et al. 2023 的 Contrastive Distillation 无关 |
| 质检漏斗 | 七段串行过滤器，任一段剔除即出局，输出每段计数与通过率 |
| 语义去重 | 3-gram Jaccard ≥ 0.85 判为重复（decision 样本指纹只用 state，避免模板字段误杀） |
| 泄漏隔离 | 训练集与评测集做指纹比对，重复样本自动剔除并计数 |
| Wilson CI | 单比例置信区间，快但不处理预测单元间的相关性 |
| record-level bootstrap | 按独立记录重采样 1000 次求区间，一条记录多问题时区间更诚实（更宽） |
| 断点续训 | LLaMA-Factory resume：从上次 checkpoint 的 optimizer 状态继续 |

训练环境侧名词（LLaMA-Factory / kev / rlcd-lite / GRPO / TRAINING_ROOT）见 [training/README.md 名词表](training/README.md#名词表)。

## 测试

```bash
npm test    # 12 套回归；干净 clone 基线 173 项，带本地 data/ 历史任务时 190 项（2026-11-12 实测）

项数差异来自 `experiment-report-ui.test.js`：该套件依赖本地任务记录生成 loss 曲线，干净 clone 会跳过，见「已知限制」。
```

## 目录

```
server.js                  独立入口（静态 tools/ 白名单 + 3 组 API）
routes/                    11 个路由模块（训练/数据集/评估/报告/沙箱）
tools/model-training.html  单页前端（内嵌样式与脚本，零构建）
tools/vendor/plotly.min.js Plotly 本地化（MIT，随仓库分发）
tests/                     12 套测试 + run-all 启动器
training/                  训练环境：2 个 Python 运行时脚本 + 依赖/模板/样例（见 training/README.md）
```

## 已知限制

- 免登录设计，默认只听 127.0.0.1；公网部署需自行加反向代理鉴权
- RLCD 引擎暂只支持本机训练环境（脚本依赖 `TRAINING_ROOT` 布局，见 [training/README.md 已知限制](training/README.md#已知限制)）
- 语义去重是 n-gram 近似，不等于语义级查重（同义不同词的重复检不出）
- AI 叙述需要云 LLM 密钥（`QWEN_KEY` 等）；不配置时统计与图表完全可用
- 干净 clone 缺少 `data/`（被 `.gitignore` 排除），`experiment-report-ui.test.js` 因无历史任务生成不了 loss 曲线，17 项断言会跳过；`npm test` 此时显示 173 项而非全量 190 项
- `training/` 的 Python 脚本不在 `npm test` 覆盖范围内，补入后只做过无 GPU 冒烟验证，未在 GPU 上重跑完整训练

## License

MIT（Plotly.js 以 MIT 随仓库分发；LLaMA-Factory / kev / rlcd-lite 为外部训练环境，需另行安装）
