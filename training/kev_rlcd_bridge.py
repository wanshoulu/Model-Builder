"""kev decision JSONL → rlcd-lite Example 转换器（平台 RLCD 训练的数据桥）。

kev decision 格式（平台 datasets/*.jsonl，kind=decision）:
    {"state": "...", "questions": {"qname": {"type": "choice|noul|score",
                                             "instructions": "...",
                                             "options"/"criteria": [...],
                                             "label": str|bool|int}}}

rlcd-lite Example:
    {"context": str, "schema": Schema, "gold": {field_name: option_string}}

映射规则（与 schema.py 的 Primitive 对齐）:
    choice → Field.choice(options 保持原序；criteria 是 dict 时用其键序)
    noul   → Field.bool_（options=("false","true")；label bool→"true"/"false"）
    score  → Field.score(criteria 列表为有序档位；label int→档位字符串)
              注意 kev 的 score label 是 0-based 档位索引（benchmark.py 的 label_to_index 同口径），
              转换时 label i → levels[i]，与 kev 官方评估一致。

用法:
    from kev_rlcd_bridge import load_kev_decision
    examples = load_kev_decision("data/ds-xxx.jsonl")        # 全量
    train, calib = split_examples(examples, calib_frac=0.1, seed=0)
    # 或命令行（平台训练命令会先调它做转换落盘）:
    python kev_rlcd_bridge.py in.jsonl out_dir/ [--calib-frac 0.1 --seed 0]
    # -> out_dir/train.jsonl + calib.jsonl（rlcd-lite JSONL Example 格式）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "rlcd-lite" / "src"))

from rlcd_lite.schema import Field, Primitive, Schema  # noqa: E402


class BridgeError(ValueError):
    """数据格式问题（缺 label / 空选项 / 不支持的类型），fail-fast 不静默丢样本。"""


def _options_of(spec: dict) -> list[str]:
    """kev 的选项列表：options 是 list[str]；criteria 可能是 list[str] 或 dict（键=选项名）。"""
    opts = spec.get("options")
    if opts is None:
        crit = spec.get("criteria")
        if isinstance(crit, dict):
            opts = list(crit.keys())
        elif isinstance(crit, list):
            opts = crit
        else:
            opts = None
    if not opts or not isinstance(opts, (list, tuple)) or len(opts) < 2:
        raise BridgeError(f"问题 {spec.get('type')} 选项为空或不足 2 个：{opts!r}")
    return [str(o) for o in opts]


def _label_to_option(spec: dict, options: list[str]) -> str:
    """kev label → rlcd-lite gold 选项字符串。三种类型的 label 形式不同。"""
    label = spec.get("label", spec.get("target"))
    qtype = spec.get("type")
    if qtype == "choice":
        if not isinstance(label, str) or label not in options:
            raise BridgeError(f"choice label {label!r} 不在选项 {options[:5]}{'...' if len(options) > 5 else ''} 中")
        return label
    if qtype == "noul":
        # bool → "true"/"false"（rlcd-lite Field.bool_ 固定选项序）
        if isinstance(label, bool):
            return "true" if label else "false"
        if isinstance(label, str) and label.lower() in ("true", "false", "yes", "no"):
            return "true" if label.lower() in ("true", "yes") else "false"
        raise BridgeError(f"noul label 应为 bool，得到 {label!r}")
    if qtype == "score":
        # int → 0-based 档位索引（kev benchmark.py label_to_index 同口径）
        if isinstance(label, bool) or not isinstance(label, int):
            # 容忍 "1"/1.0 形式
            try:
                label = int(label)
            except (TypeError, ValueError):
                raise BridgeError(f"score label 应为 int 档位索引，得到 {label!r}") from None
        if not 0 <= label < len(options):
            raise BridgeError(f"score label {label} 超出档位范围 0..{len(options) - 1}")
        return options[label]
    raise BridgeError(f"不支持的问题类型 {qtype!r}（仅 choice/noul/score）")


def _flatten_state(state) -> str:
    """kev 的 state 可能是 str，也可能是结构化对象（{'document': ...} /
    {'ticket': {'channel': ..., 'body': ...}}——kev/data.py 的多源混合格式）。
    kev 官方 prompt 把它序列化为 JSON 文本，这里保持同口径：结构化 state
    → ensure_ascii=False 的紧凑 JSON 字符串，模型读到的语义与 kev 训练一致。"""
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)) and state:
        return json.dumps(state, ensure_ascii=False)
    raise BridgeError(f"state 应为非空字符串或结构化对象，得到 {type(state).__name__}: {state!r}")


def record_to_example(rec: dict) -> "object":  # -> rlcd_lite.data.Example
    """单条 kev decision 记录 → rlcd-lite Example。字段顺序保持 questions 的原序。"""
    from rlcd_lite.data import Example

    context = _flatten_state(rec.get("state"))
    questions = rec.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BridgeError("questions 缺失或为空")

    fields = []
    gold = {}
    for name, spec in questions.items():
        qtype = spec.get("type")
        desc = str(spec.get("instructions") or name)
        if qtype == "noul":
            fields.append(Field.bool_(name, desc))
            gold[name] = _label_to_option(spec, ["false", "true"])
        elif qtype == "score":
            levels = _options_of(spec)
            fields.append(Field.score(name, levels, desc))
            gold[name] = _label_to_option(spec, levels)
        elif qtype == "choice":
            opts = _options_of(spec)
            fields.append(Field.choice(name, opts, desc))
            gold[name] = _label_to_option(spec, opts)
        else:
            raise BridgeError(f"不支持的问题类型 {qtype!r}")
    return Example(context=context, schema=Schema(tuple(fields)), gold=gold)


def load_kev_decision(path: str | Path, limit: int | None = None) -> list:
    """读取 kev decision JSONL 并整体转换。任何一条坏记录都报错（fail-fast，
    带行号——平台质检漏斗已经挡了大部分格式问题，这里防御剩余的）。"""
    from rlcd_lite.data import Example  # noqa: F401  (类型引用)

    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(record_to_example(json.loads(line)))
            except (BridgeError, json.JSONDecodeError) as e:
                raise BridgeError(f"第 {i} 行: {e}") from None
            if limit and len(records) >= limit:
                break
    if not records:
        raise BridgeError("文件里没有可用的 decision 记录")
    return records


def split_examples(examples: list, calib_frac: float = 0.1, seed: int = 0):
    """按比例切 train/calib（rlcd-lite 的 GRPOTrainer 用 calib 做训练中评估）。"""
    if not 0 < calib_frac < 1:
        raise ValueError(f"calib_frac 应在 (0,1)，得到 {calib_frac}")
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_calib = max(1, round(len(examples) * calib_frac))
    calib = [examples[i] for i in idx[:n_calib]]
    train = [examples[i] for i in idx[n_calib:]]
    return train, calib


# ── rlcd-lite JSONL Example 格式的落盘/回读（平台命令行转换用） ──────────

def example_to_json(ex) -> dict:
    return {
        "context": ex.context,
        "fields": [
            {"name": f.name, "type": f.primitive.value, "options": list(f.options),
             "description": f.description}
            for f in ex.schema.fields
        ],
        "gold": ex.gold,
    }


def example_from_json(d: dict):
    from rlcd_lite.data import Example

    fields = []
    for f in d["fields"]:
        t = f["type"]
        if t == "bool":
            fields.append(Field.bool_(f["name"], f.get("description", "")))
        elif t == "score":
            fields.append(Field.score(f["name"], f["options"], f.get("description", "")))
        elif t == "choice":
            fields.append(Field.choice(f["name"], f["options"], f.get("description", "")))
        else:
            raise BridgeError(f"未知字段类型 {t!r}")
    return Example(context=d["context"], schema=Schema(tuple(fields)), gold=dict(d["gold"]))


def dump_examples(examples: list, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(example_to_json(ex), ensure_ascii=False) + "\n")
    return len(examples)


def load_examples_jsonl(path: str | Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(example_from_json(json.loads(line)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="kev decision JSONL → rlcd-lite train/calib JSONL")
    ap.add_argument("input", help="kev decision JSONL 路径")
    ap.add_argument("out_dir", help="输出目录（train.jsonl / calib.jsonl）")
    ap.add_argument("--calib-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    try:
        examples = load_kev_decision(a.input)
        train, calib = split_examples(examples, a.calib_frac, a.seed)
        n1 = dump_examples(train, Path(a.out_dir) / "train.jsonl")
        n2 = dump_examples(calib, Path(a.out_dir) / "calib.jsonl")
        print(f"[bridge] {a.input}: {n1} train + {n2} calib -> {a.out_dir}")
        return 0
    except BridgeError as e:
        print(f"[bridge] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
