"""Model configuration and training-horizon math.

Mirrors ``scripts/base_train.py`` exactly. The horizon, batch-size, learning
rate, and weight-decay scaling are all derived from the reference model's
parameter counts, so the reference ``nanochat.gpt.GPT`` is imported rather than
reimplemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Constants copied verbatim from scripts/base_train.py.
ASPECT_RATIO = 64
HEAD_DIM = 128
B_REF = 2**19  # optimal batch size at d12 (~524,288 tokens), measured
PARAM_DATA_RATIO = 12
PAD_VOCAB_TO = 64


def model_shape(depth: int, aspect_ratio: int, head_dim: int) -> tuple[int, int]:
    """(model_dim, num_heads) exactly as ``build_model_meta`` computes them."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    return model_dim, model_dim // head_dim


def padded_vocab(vocab_size: int) -> int:
    return ((vocab_size + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO) * PAD_VOCAB_TO


def scaling_params(depth: int, aspect_ratio: int, head_dim: int, seq_len: int,
                   vocab_size: int, window_pattern: str) -> int:
    """``transformer_matrices + lm_head`` for a depth, from the reference GPT."""
    from nanochat.gpt import GPT, GPTConfig  # imported after reference on path
    import torch

    model_dim, num_heads = model_shape(depth, aspect_ratio, head_dim)
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def flops_per_token(depth: int, aspect_ratio: int, head_dim: int, seq_len: int,
                    vocab_size: int, window_pattern: str) -> float:
    """Reference ``GPT.estimate_flops()`` for the model shape."""
    from nanochat.gpt import GPT, GPTConfig
    import torch

    model_dim, num_heads = model_shape(depth, aspect_ratio, head_dim)
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    return float(model.estimate_flops())


@dataclass
class TrainPlan:
    depth: int
    model_dim: int
    num_heads: int
    head_dim: int
    seq_len: int
    vocab_size: int
    padded_vocab_size: int
    window_pattern: str
    device_batch_size: int
    total_batch_size: int
    grad_accum: int
    num_iterations: int
    target_tokens: int
    batch_lr_scale: float
    weight_decay_scaled: float
    optimizer: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)


def compute_plan(args, vocab_size: int) -> TrainPlan:
    """Reproduce the horizon/batch/LR/weight-decay derivation in base_train.py."""
    model_dim, num_heads = model_shape(args.depth, args.aspect_ratio,
                                       args.head_dim)
    padded = padded_vocab(vocab_size)

    num_scaling_params = scaling_params(
        args.depth, args.aspect_ratio, args.head_dim, args.max_seq_len,
        vocab_size, args.window_pattern)
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)

    d_ref_params = scaling_params(
        12, args.aspect_ratio, args.head_dim, args.max_seq_len, vocab_size,
        args.window_pattern)
    d_ref = args.target_param_data_ratio * d_ref_params

    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / d_ref
        predicted = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted))

    batch_ratio = total_batch_size / B_REF
    batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0

    weight_decay_scaled = (args.weight_decay * math.sqrt(total_batch_size / B_REF)
                           * (d_ref / target_tokens))

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    elif args.target_flops > 0:
        per_token = flops_per_token(args.depth, args.aspect_ratio,
                                    args.head_dim, args.max_seq_len, vocab_size,
                                    args.window_pattern)
        num_iterations = round(args.target_flops / (per_token * total_batch_size))
    else:
        num_iterations = target_tokens // total_batch_size

    tokens_per_micro = args.device_batch_size * args.max_seq_len
    if total_batch_size % tokens_per_micro != 0:
        raise SystemExit(
            f"--total-batch-size ({total_batch_size}) must be a multiple of "
            f"--device-batch-size * --max-seq-len ({tokens_per_micro})")
    grad_accum = total_batch_size // tokens_per_micro

    optimizer = {
        "embedding_lr": args.embedding_lr * batch_lr_scale,
        "unembedding_lr": args.unembedding_lr * batch_lr_scale,
        "matrix_lr": args.matrix_lr * batch_lr_scale,
        "scalar_lr": args.scalar_lr * batch_lr_scale,
        "weight_decay": weight_decay_scaled,
    }
    scheduler = {
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "weight_decay_base": weight_decay_scaled,
    }

    return TrainPlan(
        depth=args.depth,
        model_dim=model_dim,
        num_heads=num_heads,
        head_dim=args.head_dim,
        seq_len=args.max_seq_len,
        vocab_size=vocab_size,
        padded_vocab_size=padded,
        window_pattern=args.window_pattern,
        device_batch_size=args.device_batch_size,
        total_batch_size=total_batch_size,
        grad_accum=grad_accum,
        num_iterations=num_iterations,
        target_tokens=target_tokens,
        batch_lr_scale=batch_lr_scale,
        weight_decay_scaled=weight_decay_scaled,
        optimizer=optimizer,
        scheduler=scheduler,
    )
