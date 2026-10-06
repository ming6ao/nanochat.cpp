"""``python -m nanochat_cpp.base_train``: a nanochat-compatible training CLI.

Accepts the same flags and defaults as ``scripts/base_train.py``, derives the
same model configuration and horizon, passes the parquet dataset and the
``NCTOKEN1`` artifact to the C++ ``train_main`` (which tokenizes during the
run; docs/parquet-native.md), and launches it under the sandbox and GPU
broker.

    python -m nanochat_cpp.base_train --depth=8 --window-pattern=L \\
        --max-seq-len=512 --device-batch-size=8 --total-batch-size=4096 \\
        --num-iterations=50 --warmup-steps=3 --eval-every=-1 --save-every=-1 \\
        --model-tag=demo50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config, data, launcher, reference

DEFAULT_EVAL_TOKENS = 80 * 524288


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanochat_cpp.base_train",
        description="Pretrain a base model with nanochat.cpp (nanochat CLI)")
    # Logging.
    parser.add_argument("--run", type=str, default="dummy",
                        help="run name (accepted for compatibility; no wandb)")
    # Runtime.
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--fp8", action="store_true",
                        help="unsupported by the C++ backend; errors if set")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise",
                        choices=["rowwise", "tensorwise"])
    # Model architecture.
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--aspect-ratio", type=int,
                        default=config.ASPECT_RATIO)
    parser.add_argument("--head-dim", type=int, default=config.HEAD_DIM)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=4096,
                        help="accepted for compatibility; the C++ loader packs")
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    # Training horizon.
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--target-flops", type=float, default=-1.0)
    parser.add_argument("--target-param-data-ratio", type=float,
                        default=config.PARAM_DATA_RATIO)
    # Optimization.
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--total-batch-size", type=int, default=-1)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--resume-from-step", type=int, default=-1)
    # Evaluation.
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--core-metric-every", type=int, default=2000,
                        help="accepted for compatibility; not implemented")
    parser.add_argument("--core-metric-max-per-task", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=2000,
                        help="accepted for compatibility; not implemented")
    parser.add_argument("--save-every", type=int, default=-1)
    # Output.
    parser.add_argument("--model-tag", type=str, default=None)
    # Bridge-specific.
    parser.add_argument("--data-dir", type=str, default=None,
                        help="parquet directory (default: reference dataset)")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--profile", type=str, default="t2-parity",
                        help="sandbox profile for the C++ launch")
    parser.add_argument("--binary", type=str, default=None,
                        help="explicit train_main path")
    parser.add_argument("--no-build", action="store_true",
                        help="skip the Bazel build and use the current binary")
    parser.add_argument("--force-data", action="store_true",
                        help="re-tokenize even if a cached shard exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the mapped train_main command and exit")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _autodetect_backend(device_type: str) -> str:
    if device_type == "cuda":
        return "cuda"
    if device_type in ("cpu", "mps"):
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference.ensure_reference_on_path()

    if args.fp8:
        raise SystemExit("--fp8 is not supported by the C++ backend")

    if args.data_dir:
        os.environ["NANOCHAT_BASE_DIR"] = str(Path(args.data_dir).resolve().parent)

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    plan = config.compute_plan(args, vocab_size)
    backend = args.backend or _autodetect_backend(args.device_type)
    cuda = backend == "cuda"

    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR",
                                   str(Path.home() / ".cache" / "nanochat")))
    tag = args.model_tag or f"d{args.depth}"
    checkpoint_dir = base_dir / "base_checkpoints" / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "model.ckpt"
    log_path = checkpoint_dir / "train.log"

    print(f"[nanochat_cpp] vocab size: {vocab_size:,}")
    print(f"[nanochat_cpp] model: depth={plan.depth} dim={plan.model_dim} "
          f"heads={plan.num_heads} head_dim={plan.head_dim} "
          f"seq={plan.seq_len} window={plan.window_pattern}")
    print(f"[nanochat_cpp] horizon: target_tokens={plan.target_tokens:,} "
          f"total_batch={plan.total_batch_size:,} "
          f"grad_accum={plan.grad_accum} iterations={plan.num_iterations:,} "
          f"lr_scale={plan.batch_lr_scale:.4f} "
          f"weight_decay={plan.weight_decay_scaled:.6f}")

    eval_steps = 0
    if args.eval_every > 0:
        eval_steps = max(1, args.eval_tokens // (plan.device_batch_size * plan.seq_len))

    core_args = [
        "--layers", str(plan.depth),
        "--heads", str(plan.num_heads),
        "--kv-heads", str(plan.num_heads),
        "--hidden", str(plan.model_dim),
        "--seq", str(plan.seq_len),
        "--vocab", str(vocab_size),
        "--padded-vocab", str(plan.padded_vocab_size),
        "--window-pattern", plan.window_pattern,
        "--batch", str(plan.device_batch_size),
        "--grad-accum", str(plan.grad_accum),
        "--num-iterations", str(plan.num_iterations),
        "--log-every", "1",
        "--save-every", str(args.save_every if args.save_every > 0 else 0),
        "--seed", str(args.seed),
        "--checkpoint", str(checkpoint),
        "--log", str(log_path),
        "--embedding-lr", repr(plan.optimizer["embedding_lr"]),
        "--unembedding-lr", repr(plan.optimizer["unembedding_lr"]),
        "--matrix-lr", repr(plan.optimizer["matrix_lr"]),
        "--scalar-lr", repr(plan.optimizer["scalar_lr"]),
        "--weight-decay", repr(plan.optimizer["weight_decay"]),
        "--weight-decay-base", repr(plan.scheduler["weight_decay_base"]),
        "--clip", "0",
        "--warmup-steps", str(plan.scheduler["warmup_steps"]),
        "--warmdown-ratio", repr(plan.scheduler["warmdown_ratio"]),
        "--final-lr-frac", repr(plan.scheduler["final_lr_frac"]),
    ]
    if eval_steps > 0:
        core_args += ["--eval-every", str(args.eval_every),
                      "--eval-steps", str(eval_steps)]
    if args.resume_from_step != -1:
        core_args += ["--resume", str(checkpoint)]

    artifact = data.nctoken1_path()
    if artifact is None:
        raise SystemExit(
            "training needs the NCTOKEN1 artifact; run "
            "tools/convert_tokenizer.py first")
    if args.dry_run:
        preview = ["--train-parquet", "<train-parquet>",
                   "--tokenizer", str(artifact)] + core_args
        if eval_steps > 0:
            preview[2:2] = ["--val-parquet", "<val-parquet>"]
        print("[nanochat_cpp] dry run; train_main arguments:")
        print("  " + " ".join(preview))
        return 0

    # The C++ runtime reads parquet and tokenizes on the fly
    # (docs/parquet-native.md). No shard is materialized.
    train_parquet = ",".join(data.parquet_files("train"))
    cpp_args = ["--train-parquet", train_parquet,
                "--tokenizer", str(artifact)] + core_args
    if eval_steps > 0:
        val_parquet = ",".join(data.parquet_files("val"))
        cpp_args[2:2] = ["--val-parquet", val_parquet]

    device = launcher.gpu_name() if cuda else ""
    if device:
        cpp_args += ["--device", device]

    repo_root = reference.repository_root()
    binary = Path(args.binary) if args.binary else launcher.ensure_built(
        repo_root, cuda, build=not args.no_build)
    print(f"[nanochat_cpp] launching {binary} ({backend})")
    sys.stdout.flush()
    return launcher.launch(repo_root, binary, cpp_args, cuda, args.profile)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
