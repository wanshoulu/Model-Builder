"""平台 RLCD 训练入口（A1000 / 本机 Git Bash 均可调用）。

平台（routes/model-training.js trainMode=rlcd）会这样调用：
    python train_rlcd.py
        --data-dir <workdir>/rlcd-data/<jobId>
        --base Qwen/Qwen3-0.6B-Base
        --steps 100 --group-size 8 --reward rlcr
        --lr 1e-5 --rollout-temp 1.0 --conf-grid 5
        --out saves/<jobId> --logs <workdir>/logs/<jobId>.log

输出（与 kev 引擎同构，平台直接解析）:
    <out>/adapter/            LoRA 适配器（训练产物）
    <out>/metrics.jsonl       每步指标（step/loss/reward_mean/acc/stated_conf_mean/...）
    <out>/rows.json           kev benchmark 兼容的逐样本评估行（含 logits）——平台评估闭环直接吃
    <out>/report.json         汇总指标（acc/ece/brier/温度）
    <out>/training_done.json  退出码副证（平台 .exit 文件机制之外的双保险）

设计要点：
- 训练后立即在 calib 集上做 held-out 评估 + 温度标定（RL 训后 softmax 必坍缩，
  不标定的概率不可用——rlcd-lite Finding 4）
- metrics.jsonl 每步 flush：平台可实时画 reward/acc/confidence 三条曲线
- 复用 rlcd_lite 原生 trainer/evaluate，不复制算法代码；本文件只做胶水
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rlcd-lite" / "src"))
sys.path.insert(0, str(HERE))

from kev_rlcd_bridge import load_examples_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RLCD (GRPO + RLCR) training for the platform")
    sub = ap.add_subparsers(dest="mode")
    # 默认模式 = train（兼容旧调用：无子命令时按训练解析）
    ap.add_argument("--data-dir", required=False, help="kev_rlcd_bridge.py 的输出目录（train.jsonl/calib.jsonl）")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2, help="examples per step")
    ap.add_argument("--reward", choices=("rlcr", "brier", "logloss", "binary"), default="rlcr")
    ap.add_argument("--honesty-lambda", type=float, default=0.0,
                    help="诚实惩罚系数 λ：reward -= λ*|stated_conf - batch_acc|。"
                         "qwen 建议项：置信度膨胀（acc 77%% 但 stated 90%%）时用它压回真实水平。"
                         "推荐起点 0.3；0 = 关闭（标准 RLCR）")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Early Stopping：验证集 accuracy 连续 N 个评估点无提升则提前停止"
                         "（并回滚到最优 step 的 adapter）。0 = 关闭。qwen 建议项。")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--conf-grid", type=int, default=5)
    ap.add_argument("--decouple", type=int, choices=(0, 1), default=0,
                    help="choice/confidence 分开算 advantage（纯 Brier 奖励的修复之一）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="产物目录（adapter/metrics/rows/report）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", help="bf16 建议（CUDA）；fp32 回退")
    # eval 模式：--run <saves/jobId> --eval-data <decision JSONL>，输出 rows.json/report.json 到 --out
    ev = sub.add_parser("eval", help="用已训 adapter 评估 kev decision JSONL（平台评估闭环入口）")
    ev.add_argument("--run", required=True, help="训练产物目录（含 adapter/ 与 config.json）")
    ev.add_argument("--eval-data", required=True, help="kev decision JSONL（原始格式，内部转换）")
    ev.add_argument("--out", required=True)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--dtype", default="bf16")
    a = ap.parse_args()
    if a.mode != "eval":
        if not a.out:
            ap.error("--out is required for training mode")
        if not a.data_dir:
            ap.error("--data-dir is required for training mode")
    return a


def run_eval(a) -> int:
    """评估模式：加载 adapter → kev decision JSONL → rows.json/report.json（平台评估闭环直接读）。"""
    from pathlib import Path as _P
    out = _P(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        from rlcd_lite.engine import DecisionEngine
        from rlcd_lite import evaluate as EV
        from kev_rlcd_bridge import load_kev_decision, dump_examples, load_examples_jsonl

        run = _P(a.run)
        # adapter 的底座记录在 config.json（训练时写入）
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        base = cfg.get("model") or a.base
        print(f"[rlcd-eval] base={base} adapter={run / 'adapter'}", flush=True)
        engine = DecisionEngine.from_pretrained(base, device=a.device, dtype=a.dtype,
                                                adapter=str(run / "adapter"))

        examples = load_kev_decision(a.eval_data)
        # 转成 JSONL 再回读，保证与训练同一转换口径
        tmp = out / "_eval_examples.jsonl"
        dump_examples(examples, tmp)
        examples = load_examples_jsonl(tmp)
        tmp.unlink(missing_ok=True)

        records = EV.collect(engine, examples, None, log_every=max(1, len(examples) // 4))
        evaluation = EV.evaluate(records)
        rows = EV.records_to_rows(records)
        (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        report = {
            "engine": "rlcd",
            "n_examples": len(examples),
            "n_records": evaluation.get("n_records"),
            "accuracy": evaluation.get("accuracy"),
            "per_field_accuracy": evaluation.get("per_field_accuracy"),
            "signals": {
                name: {k: v for k, v in s.items() if isinstance(v, (int, float, str, list, dict))}
                for name, s in (evaluation.get("signals") or {}).items()
            },
            "duration_sec": round(time.time() - t0, 1),
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rlcd-eval] acc={report['accuracy']} rows={len(rows)} -> {out}", flush=True)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            (out / "report.json").write_text(json.dumps({"ok": False, "error": str(e)[:500]}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return 1


def main() -> int:
    a = parse_args()
    if getattr(a, "mode", None) == "eval":
        return run_eval(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        import torch
        from rlcd_lite.data import Example  # noqa: F401
        from rlcd_lite.engine import pick_device, pick_dtype
        from rlcd_lite.trainer import GRPOConfig, GRPOTrainer

        train_examples = load_examples_jsonl(Path(a.data_dir) / "train.jsonl")
        calib_examples = load_examples_jsonl(Path(a.data_dir) / "calib.jsonl")
        print(f"[rlcd] {len(train_examples)} train / {len(calib_examples)} calib examples", flush=True)

        device = pick_device(a.device)
        # GRPOConfig.dtype 期望字符串（'fp32'/'bf16'），GRPOTrainer 内部自己调 pick_dtype
        dtype = a.dtype  # 'bf16'（默认）或 'fp32'
        cfg = GRPOConfig(
            model=a.base,
            steps=a.steps,
            batch_examples=a.batch,
            group_size=a.group_size,
            reward=a.reward,
            lr=a.lr,
            lora_r=a.lora,
            rollout_temp=a.rollout_temp,
            conf_grid=a.conf_grid,
            decouple=bool(a.decouple),
            normalize_std=False,          # Dr. GRPO（rlcd-lite 默认）
            seed=a.seed,
            # 注意：cfg.device/dtype 必须保持字符串——trainer 会把整个 config
            # asdict() 后写 config.json，传 torch.device 对象会触发循环引用崩溃
            device=a.device if a.device != 'auto' else None,
            dtype=dtype,
            run_name=out.name,
            runs_root=str(out.parent),    # runs/<jobId>/ 即 --out
            # qwen 建议的细粒度日志/评估：默认密度提高（10 步一评 → 最多 30 步一评）
            eval_every=max(5, min(30, a.steps // 10)),
            eval_n=min(len(calib_examples), 100),
            save_every=max(20, a.steps // 3),
        )
        trainer = GRPOTrainer(cfg, train_examples=train_examples, eval_examples=calib_examples)
        # qwen 建议的诚实惩罚：包装 reward，在标准 RLCR 上叠加 λ*|stated - batch_acc|。
        # 实现：monkeypatch trainer.reward.per_field—— advantage 仍按组内相对值计算，
        # 该惩罚项压低「自报置信度系统性偏离批次真实准确率」的 rollout。
        if a.honesty_lambda > 0:
            base_reward = trainer.reward
            lam = a.honesty_lambda
            class HonestRLCR:
                """RLCR - λ*|stated_conf - batch_acc|：对「盲目自信」的额外压力。
                batch_acc 是该 rollout 组内的实际正确率（组基线），使惩罚在
                组间可比、组内相对（与 GRPO advantage 语义一致）。"""
                name = base_reward.name + f'+honest{lam}'
                def per_field(self, correct, conf):
                    base = base_reward.per_field(correct.float(), conf.float())
                    batch_acc = correct.float().mean()
                    penalty = (conf.float().mean() - batch_acc).abs()
                    return base - lam * penalty
                def __call__(self, correct, conf):
                    return self.per_field(correct, conf).mean(dim=-1)
            trainer.reward = HonestRLCR()
            print(f"[rlcd] honesty-lambda={lam} 启用：reward = {base_reward.name} - {lam}*|stated - batch_acc|", flush=True)

        # ── 训练循环 ──
        # rlcd-lite 的 train() 是自有循环（print 进度 + 自己写 runs/<name>/metrics.jsonl），
        # runs_root 已指到 out.parent，产物路径天然就是 <out>/metrics.jsonl。
        # Early Stopping（qwen 建议）：patience>0 时复刻 trainer.train 循环并监控
        # 验证集 accuracy——连续 patience 个评估点无提升则停止，并回滚最优 step 的
        # adapter（save 每 eval 点都覆盖 adapter/，故「最优 step 后未再 save」即最优）。
        if a.early_stop_patience > 0 and calib_examples and a.steps >= 30:
            cfg = trainer.cfg
            t0 = time.time()
            best_acc, best_step, bad_streak, stopped_at = -1.0, 0, 0, None
            if cfg.eval_every and trainer.eval_examples:
                rep0 = trainer.evaluate(0)
                best_acc = rep0.get("accuracy", 0.0)
            for s in range(1, cfg.steps + 1):
                batch = trainer.rng.sample(trainer.train_examples, min(cfg.batch_examples, len(trainer.train_examples)))
                m = trainer.step(batch, s)
                if s % 5 == 0 or s == 1:
                    print(f"[rlcd-lite] step {s}/{cfg.steps} loss={m.loss:+.4f} R={m.reward_mean:+.3f} acc={m.acc:.3f} "
                          f"stated={m.stated_conf_mean:.2f} ({time.time() - t0:.0f}s)", flush=True)
                if cfg.eval_every and s % cfg.eval_every == 0 and trainer.eval_examples:
                    rep = trainer.evaluate(s)
                    acc = rep.get("accuracy", 0.0)
                    if acc > best_acc + 1e-6:
                        best_acc, best_step, bad_streak = acc, s, 0
                        trainer.save()   # 最优点快照
                    else:
                        bad_streak += 1
                        if bad_streak >= a.early_stop_patience:
                            stopped_at = s
                            print(f"[rlcd] early-stop @ step {s}：验证集 acc 连续 {bad_streak} 个评估点无提升"
                                  f"（best acc={best_acc:.3f} @ step {best_step}），回滚最优 adapter", flush=True)
                            break
                if cfg.save_every and s % cfg.save_every == 0:
                    trainer.save()
            if stopped_at is None:
                trainer.save()
            print(f"[rlcd] 训练结束（early_stop={'是@'+str(stopped_at) if stopped_at else '未触发'}，best_acc={best_acc:.3f}@{best_step}）", flush=True)
        else:
            trainer.train()

        # ── 训练后评估 + 温度标定（复用 rlcd-lite evaluate 原生实现） ──
        # 一半 calib 做温度拟合集，一半做 held-out 报告集（避免 in-sample 温度）
        from rlcd_lite import evaluate as EV

        n_half = max(1, len(calib_examples) // 2)
        fit_set, report_set = calib_examples[:n_half], calib_examples[n_half:]
        calib_records = EV.collect(trainer.engine, fit_set, trainer.grid, log_every=0)
        report_records = EV.collect(trainer.engine, report_set, trainer.grid, log_every=0)
        evaluation = EV.evaluate(report_records, calib_records=calib_records)

        rows = EV.records_to_rows(report_records + calib_records)
        rows_path = out / "rows.json"
        rows_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

        report = {
            "n_examples": len(report_set),
            "accuracy": evaluation.get("accuracy"),
            "per_field_accuracy": evaluation.get("per_field_accuracy"),
            "temperatures": evaluation.get("temperatures"),
            "signals": {
                name: {k: v for k, v in s.items() if isinstance(v, (int, float, str, list, dict))}
                for name, s in (evaluation.get("signals") or {}).items()
            },
            "duration_sec": round(time.time() - t0, 1),
        }

        report["config"] = {
            "base": a.base, "steps": a.steps, "group_size": a.group_size,
            "reward": a.reward, "lr": a.lr, "lora_r": a.lora,
            "rollout_temp": a.rollout_temp, "conf_grid": a.conf_grid,
            "decouple": bool(a.decouple), "seed": a.seed,
            "honesty_lambda": a.honesty_lambda,
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "training_done.json").write_text(json.dumps(
            {"ok": True, "finished_at": time.time(), "out": str(out)}, ensure_ascii=False), encoding="utf-8")
        print(f"[rlcd] done in {round(time.time() - t0, 1)}s -> {out}", flush=True)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            Path(a.out).joinpath("training_done.json").write_text(json.dumps(
                {"ok": False, "error": str(e)[:500], "finished_at": time.time()}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    _code = main()
    # Windows + torch 2.9：解释器退出阶段的 CUDA/线程清理偶发抛错，导致结果完整
    # （rows/report 已落盘、return 0）但进程退出码是 1——平台把评估误判为失败。
    # os._exit 跳过解释器清理阶段，直接以 main() 的返回码退出。
    try:
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(_code if isinstance(_code, int) else 0)
