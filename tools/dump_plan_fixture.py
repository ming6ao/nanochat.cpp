#!/usr/bin/env python3
"""Generate the planning fixture, ``tests/data/plan_fixture.bin``.

The fixture is *data*: it is produced once, offline, from the reference PyTorch
nanochat checkout and committed to the tree. The runtime test never imports
torch; it reads the fixture and compares the C-based planning layer
(``python/nanochat_cpp/plan.py``) against it (docs/python.md sections 4.3 and
11).

For each case the fixture records:

* the planning inputs (the arguments of ``compute_plan``);
* the reference parameter counts from ``GPT.num_scaling_params`` and the
  reference ``GPT.num_matmul_params`` for the same config;
* the reference ``GPT.estimate_flops`` value for one token;
* the reference plan outputs, computed with the ``scripts/base_train.py``
  arithmetic.

The whole file is little-endian JSON inside a small container::

    header : magic[8] = b"NANOPLAN", uint32 version
    body   : UTF-8 JSON

The generation is CPU-only:

    /path/to/reference/.venv/bin/python tools/dump_plan_fixture.py \
        --out tests/data/plan_fixture.bin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

# Force the CPU reference before importing torch. The fixture is the oracle and
# the host has one shared GPU that must stay free for the candidate run.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

MAGIC = b"NANOPLAN"
VERSION = 1

# Policy constants copied from scripts/base_train.py.
B_REF = 2**19

#: The planning cases. Each set of inputs is a ``compute_plan`` call.
CASES = [
    {
        "depth": 8,
        "aspect_ratio": 64,
        "head_dim": 128,
        "seq_len": 512,
        "vocab_size": 32768,
        "window_pattern": "SSSL",
        "device_batch_size": 8,
        "total_batch_size": -1,
        "num_iterations": 0,
        "target_flops": 0,
        "target_param_data_ratio": 12,
    },
    {
        "depth": 12,
        "aspect_ratio": 64,
        "head_dim": 128,
        "seq_len": 2048,
        "vocab_size": 32768,
        "window_pattern": "SSSL",
        "device_batch_size": 32,
        "total_batch_size": 524288,
        "num_iterations": 100,
        "target_flops": 0,
        "target_param_data_ratio": 12,
    },
    {
        "depth": 20,
        "aspect_ratio": 64,
        "head_dim": 128,
        "seq_len": 2048,
        "vocab_size": 50304,
        "window_pattern": "SSL",
        "device_batch_size": 16,
        "total_batch_size": -1,
        "num_iterations": 0,
        "target_flops": 0,
        "target_param_data_ratio": 12,
    },
]

#: Default base rates and schedule, copied from scripts/base_train.py.
RATES = {
    "embedding_lr": 0.3,
    "unembedding_lr": 0.008,
    "weight_decay": 0.28,
    "matrix_lr": 0.02,
    "scalar_lr": 0.5,
    "warmup_steps": 40,
    "warmdown_ratio": 0.65,
    "final_lr_frac": 0.05,
}


def _reference_model(depth: int, aspect_ratio: int, head_dim: int,
                     seq_len: int, vocab_size: int, window_pattern: str):
    """A reference ``GPT`` on the meta device for the requested config."""
    import torch

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size,
                       n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
                       n_embd=model_dim, window_pattern=window_pattern)
    with torch.device("meta"):
        return GPT(config)


def _reference_counts(depth: int, aspect_ratio: int, head_dim: int,
                      seq_len: int, vocab_size: int,
                      window_pattern: str) -> dict:
    model = _reference_model(depth, aspect_ratio, head_dim, seq_len,
                             vocab_size, window_pattern)
    counts = model.num_scaling_params()
    return {
        "total": int(counts["total"]),
        "transformer_matrices": int(counts["transformer_matrices"]),
        "lm_head": int(counts["lm_head"]),
        "embeddings": int(counts["wte"] + counts["value_embeds"]),
        "scalars": int(counts["scalars"]),
        "num_matmul_params": int(model.num_matmul_params()),
        "flops_per_token": float(model.estimate_flops()),
    }


def _reference_plan(case: dict) -> dict:
    """The reference plan outputs for one case, from base_train.py."""
    counts = _reference_counts(
        case["depth"], case["aspect_ratio"], case["head_dim"], case["seq_len"],
        case["vocab_size"], case["window_pattern"])
    reference = _reference_counts(
        12, case["aspect_ratio"], case["head_dim"], case["seq_len"],
        case["vocab_size"], case["window_pattern"])

    scaling = counts["transformer_matrices"] + counts["lm_head"]
    target_tokens = int(case["target_param_data_ratio"] * scaling)
    d_ref = case["target_param_data_ratio"] * (
        reference["transformer_matrices"] + reference["lm_head"])

    total_batch_size = case["total_batch_size"]
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / d_ref
        predicted = B_REF * batch_size_ratio**0.383
        total_batch_size = 2 ** round(math.log2(predicted))

    batch_ratio = total_batch_size / B_REF
    batch_lr_scale = batch_ratio**0.5 if batch_ratio != 1.0 else 1.0
    weight_decay_scaled = (RATES["weight_decay"]
                           * math.sqrt(total_batch_size / B_REF)
                           * (d_ref / target_tokens))

    if case["num_iterations"] > 0:
        iterations = case["num_iterations"]
    elif case["target_flops"] > 0:
        iterations = round(
            case["target_flops"]
            / (counts["flops_per_token"] * total_batch_size))
    else:
        iterations = target_tokens // total_batch_size

    tokens_per_micro = case["device_batch_size"] * case["seq_len"]
    grad_accum = total_batch_size // tokens_per_micro

    padded_vocab = ((case["vocab_size"] + 63) // 64) * 64
    optimizer = {
        "embedding_lr": RATES["embedding_lr"] * batch_lr_scale,
        "unembedding_lr": RATES["unembedding_lr"] * batch_lr_scale,
        "matrix_lr": RATES["matrix_lr"] * batch_lr_scale,
        "scalar_lr": RATES["scalar_lr"] * batch_lr_scale,
        "weight_decay": weight_decay_scaled,
    }
    scheduler = {
        "warmup_steps": RATES["warmup_steps"],
        "warmdown_ratio": RATES["warmdown_ratio"],
        "final_lr_frac": RATES["final_lr_frac"],
        "weight_decay_base": weight_decay_scaled,
    }
    return {
        "padded_vocab_size": padded_vocab,
        "total_batch_size": int(total_batch_size),
        "grad_accum": int(grad_accum),
        "num_iterations": int(iterations),
        "target_tokens": int(target_tokens),
        "scaling_params": int(scaling),
        "batch_lr_scale": float(batch_lr_scale),
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/data/plan_fixture.bin")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "python"))
    from nanochat_cpp.reference import find_reference_repo  # noqa: E402

    reference = find_reference_repo()
    sys.path.insert(0, str(reference))
    global GPT, GPTConfig
    from nanochat.gpt import GPT, GPTConfig  # noqa: E402

    cases = []
    for case in CASES:
        counts = _reference_counts(
            case["depth"], case["aspect_ratio"], case["head_dim"],
            case["seq_len"], case["vocab_size"], case["window_pattern"])
        cases.append({
            "inputs": dict(case),
            "counts": {key: counts[key] for key in (
                "total", "transformer_matrices", "lm_head", "embeddings",
                "scalars", "num_matmul_params")},
            "flops_per_token": counts["flops_per_token"],
            "plan": _reference_plan(case),
        })

    body = json.dumps({"version": VERSION, "cases": cases},
                      sort_keys=True, indent=2).encode("utf-8")
    payload = MAGIC + struct.pack("<I", VERSION) + body
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"wrote {out} ({len(cases)} cases, {len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
