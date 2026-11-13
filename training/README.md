# training/ — 训练环境

把 `training/` 当作 `TRAINING_ROOT`，Model-Builder 才能提交真实训练任务：工作台只生成 YAML 与命令行，真正跑 LoRA / GRPO 训练的 Python 进程住在这个目录里。

## 是什么

Model-Builder 分两层。`server.js` + `routes/` + `tools/model-training.html` 是**编排层**：数据集合成与七段质检、任务表单、进程树击杀、训练后评估、实验报告。本目录是**执行层**：`train_rlcd.py` 提供 RLCD/GRPO 训练与 `eval` 子命令，`kev_rlcd_bridge.py` 把 decision JSONL 转成 rlcd-lite 的 Example 格式。两层之间走本机 bash 或 SSH，连接参数是 `TRAINING_ROOT` / `TRAINING_WORKDIR` / `TRAINING_CLI` 三个环境变量。

交付物：2 个运行时脚本、1 份依赖清单、2 个配置模板、2 组小样例数据。第三方引擎、模型权重、venv 与任务产物都不入库。

状态：2026-11-12 从本机训练环境补入。已通过无 GPU 冒烟验证（脚本导入 + 样例解析，18 条 train / 2 条 calib）；**补入后未在 GPU 上重跑完整训练**。

## 名词表

主仓 [README 名词表](../README.md#名词表) 定义了 SFT、decision 数据、RLCD、质检漏斗等，这里不重复，只补编排层文档里没有的词。

| 名词 | 含义与出处 |
|---|---|
| LLaMA-Factory | 提供零代码 CLI 与 Web UI、可微调 100+ 大模型的框架；Apache-2.0，要求 Python ≥ 3.11。出处：[官方 README](https://github.com/hiyouga/LLaMA-Factory#installation) |
| kev | 基于 Qwen3.5 / Qwen3.8 的小型决策模型族，输出带校准概率的 typed JSON 决策；Apache-2.0。出处：[官方 README](https://github.com/jaredpalmer/kev) |
| rlcd-lite | "RL for Calibrated Decisions" 的可复现轻量实现，用 GRPO 加 proper-scoring-rule 奖励训练校准决策；Apache-2.0。官方 README 明确声明它**不是** TypeSafe 未公开的 RLCD 算法，也与 Yang et al. 2023 的 Contrastive Distillation 无关。出处：[官方 README](https://github.com/arnabgho/rlcd-lite) |
| GRPO | Group Relative Policy Optimization，PPO 的变体：去掉 critic 模型，改用组内分数估计基线以降低显存。出处：[DeepSeekMath, arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| TRAINING_ROOT | 训练环境根目录。`routes/model-training.js` 用它拼出 `TRAINING_ROOT + '/train_rlcd.py'` 与 `'/kev_rlcd_bridge.py'`，也用来推导 `venv/.../llamafactory-cli` |
| TRAINING_WORKDIR | 任务工作目录，默认 `TRAINING_ROOT/ai-harbor-training`，下含 `data/ jobs/ logs/ run/ saves/ evals/` |

## 最短上手

在 `training/` 目录执行（Windows 把 `venv/bin/` 换成 `venv\Scripts\`）：

```bash
git clone --depth 1 https://github.com/arnabgho/rlcd-lite.git rlcd-lite
python -m venv venv && venv/bin/pip install -r requirements.txt
python -c "from kev_rlcd_bridge import load_examples_jsonl; print(len(load_examples_jsonl('sample-data/bank-intent-train.jsonl')))"   # -> 18
```

第三条是无 GPU 冒烟验证，输出 `18` 说明脚本与样例数据能配合。要跑真实训练，还需按下面「接上工作台」装 LLaMA-Factory / kev 与模型权重。

## 目录约定

```
training/
├── train_rlcd.py              # 运行时入口：默认 train / eval 子命令
├── kev_rlcd_bridge.py         # kev decision JSONL -> rlcd-lite Example
├── requirements.txt           # Python >= 3.11 的最小依赖
├── templates/
│   ├── dataset_info.json      # LLaMA-Factory 数据集注册模板
│   └── job-sft.yaml           # SFT YAML 模板（路径为占位符）
├── sample-data/
│   ├── bank-intent-train.jsonl / -calib.jsonl   # rlcd-lite Example 格式，20 条
│   └── bank_qa_demo.json      # alpaca 格式，12 条
├── rlcd-lite/                 # 需自行 clone，不入库
├── venv/                      # 需自建，不入库
├── models/                    # 权重，GB 级，不入库
└── ai-harbor-training/        # 任务工作目录，运行时生成，不入库
```

两个脚本都用 `Path(__file__).resolve().parent / "rlcd-lite" / "src"` 加进 `sys.path`，因此 `rlcd-lite/` 必须与它们**同级**；用 pip 把 rlcd-lite 装到别的环境不能替代这一步。

## 接上工作台

项目根目录复制 `.env.example` 为 `.env`，指向本目录：

```dotenv
TRAINING_ROOT=./training
TRAINING_WORKDIR=./training/ai-harbor-training
TRAINING_CLI=./training/venv/bin/llamafactory-cli
```

Windows 的 `TRAINING_CLI` 用 `./training/venv/Scripts/llamafactory-cli.exe`。三条训练路径各自需要的东西：

| 引擎 | 工作台调用 | 额外安装 |
|---|---|---|
| LLaMA-Factory (SFT) | `llamafactory-cli train jobs/<id>.yaml` | 官方 `git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git && pip install -e .`；在 `TRAINING_WORKDIR/data/dataset_info.json` 注册数据集 |
| kev (decision) | `python -m kev.train` / `python -m kev.benchmark` | [kev](https://github.com/jaredpalmer/kev) 源码与 `evals/decision-v1` 评测套件 |
| RLCD / GRPO | `python train_rlcd.py ...` + `python kev_rlcd_bridge.py ...` | 本目录 `rlcd-lite/`；仅本机训练环境，暂不支持 SSH 远端 |

任务 YAML 由工作台按表单生成（`POST /api/model-training/config/preview`），`templates/job-sft.yaml` 只用于离线排查。

## 设计决策与代价

- **放进同一个仓库，而不是独立仓**：clone 一次即可把 `TRAINING_ROOT` 指到确定路径，脚本与调用它的 `routes/model-training.js` 保持版本同步。代价是仓库里多了一层不属于 Node 运行时的 Python 代码，`npm test` 不覆盖它们。
- **只收 2 个运行时脚本**：本机训练目录另有 9 个脚本（`ccb_synth.py` / `ccb_judge.py` 数据合成与 LLM 裁判、`build-test-subset.py`、`find-labeled-kev.py`、`inspect-kev-data.py`、`inspect-onecycle.py`、`mem-profile.py`、`download-qwen35*.py`）。它们不在工作台调用链上，且全部写死了本机绝对路径，部分还直读外部项目的 `.env` 取密钥——放进 public 仓库既不可复现，也会暴露本机目录布局。代价是这些实验脚本仍只存在于本机，没有版本管理与备份。
- **不提交 rlcd-lite / venv / models / 任务产物**：第三方代码由各自上游按 Apache-2.0 维护，权重与 venv 达 GB 级，任务产物含真实数据。代价是 clone 后必须多执行一条 `git clone` 才能冒烟验证。

## 已知限制

- 只做了无 GPU 验证；补入后未重跑完整训练，模型权重、CUDA 与 torch 版本差异都可能导致失败。
- ~~RLCD 分支只支持本机训练环境~~ **2026-11 更新**：SSH 远端 RLCD 已放开——在平台「训练服务器」配置里填 `rlcdRoot`（目标机器上的脚本根目录，Windows 训练机如 `D:/llm-training`）即可；本地模式仍走 `TRAINING_ROOT`/`RLCD_ROOT`。
- Windows SSH 训练机（Git Bash sshd）：训练/评估进程用 PowerShell 脱离会话启动；`dataloader_num_workers` 强制 0（transformers 闭包 pickle 兼容）；击杀按 jobId 匹配命令行。首次接入实测：Win11 + RTX A1000 全链路（提交→断连存活→adapter 产出→自动评估）通过。
- `sample-data/` 是 20 条冒烟样例，不是可训练的数据集；真实数据集由工作台「数据集」页签生成并落在 `TRAINING_WORKDIR/data/`。
- `requirements.txt` 不含 LLaMA-Factory 本体，只覆盖 `rlcd-lite` 源码的最小依赖。
- 两个脚本依赖 `os._exit` 规避 Windows + torch 2.9 退出阶段的 CUDA 清理误报（`train_rlcd.py` 尾部），换平台时需重新验证退出码语义。
