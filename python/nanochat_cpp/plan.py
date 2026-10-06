"""The planning layer for ``nanochat.cpp`` (docs/python.md section 4).

The reference training plan needs two model-derived numbers: the scaling
parameter count and the FLOP estimate for one token. The reference computes
both in PyTorch. This package must not import PyTorch, so the two numbers come
from the C library through :func:`params`. The rest of the plan is arithmetic
with no torch, so it stays here.

:func:`compute_plan` mirrors ``scripts/base_train.py``: it builds the
:class:`nanochat_cpp.Config`, reads the two C numbers for the target depth and
for depth 12, and derives the horizon, the batch size, the learning-rate
scale, and the weight-decay scale.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
from typing import Mapping

from . import _core, _lib
from .api import Config

__all__ = [
    "ASPECT_RATIO",
    "B_REF",
    "HEAD_DIM",
    "PARAM_DATA_RATIO",
    "Params",
    "TrainPlan",
    "compute_plan",
    "params",
]

#: The model aspect ratio, from scripts/base_train.py.
ASPECT_RATIO = 64
#: The reference head width, from scripts/base_train.py.
HEAD_DIM = 128
#: The optimal batch size at depth 12, in tokens, from scripts/base_train.py.
B_REF = 2**19
#: The reference tokens-to-scaling-parameters ratio.
PARAM_DATA_RATIO = 12

#: Default base rates and schedule, copied from scripts/base_train.py.
_DEFAULT_RATES = {
    "embedding_lr": 0.3,
    "unembedding_lr": 0.008,
    "weight_decay": 0.28,
    "matrix_lr": 0.02,
    "scalar_lr": 0.5,
    "warmup_steps": 40,
    "warmdown_ratio": 0.65,
    "final_lr_frac": 0.05,
}


@dataclasses.dataclass(frozen=True)
class Params:
    """The two model-derived plan inputs from the C library.

    ``total`` counts every allocated parameter. ``transformer_matrices`` and
    ``lm_head`` mirror the reference ``GPT.num_scaling_params``.
    """

    total: int
    transformer_matrices: int
    lm_head: int
    embeddings: int
    scalars: int
    flops_per_token: float

    @property
    def scaling_params(self) -> int:
        """``transformer_matrices + lm_head``, the horizon-setting count."""
        return self.transformer_matrices + self.lm_head


@dataclasses.dataclass(frozen=True)
class TrainPlan:
    """A model configuration plus its training horizon and rates."""

    config: Config
    padded_vocab_size: int
    device_batch_size: int
    total_batch_size: int
    grad_accum: int
    num_iterations: int
    target_tokens: int
    scaling_params: int
    flops_per_token: float
    batch_lr_scale: float
    optimizer: Mapping[str, float]
    scheduler: Mapping[str, float]


def params(config: Config) -> Params:
    """Call ``nanochat_params_get`` and return the model-derived numbers.

    The call allocates no model, so it is cheap enough for the two configs the
    plan needs.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be a nanochat_cpp.Config")
    library = _lib.load()
    c_config = config._to_c()
    out = _core.Params()
    _core.check_status(
        library,
        library.nanochat_params_get(
            ctypes.byref(c_config), ctypes.byref(out)),
        "nanochat_params_get")
    return Params(
        total=int(out.total),
        transformer_matrices=int(out.transformer_matrices),
        lm_head=int(out.lm_head),
        embeddings=int(out.embeddings),
        scalars=int(out.scalars),
        flops_per_token=float(out.flops_per_token),
    )


def _reference_config(depth: int, aspect_ratio: int, head_dim: int,
                      seq_len: int, vocab_size: int, window_pattern: str,
                      num_kv_heads: int | None,
                      rope_base: float) -> Config:
    """Build a :class:`Config` from the plan arguments.

    The reference plan forces ``n_kv_head = num_heads``. The C++ ``Config``
    carries ``num_kv_heads``, so an explicit value here is a deliberate
    difference from the reference; leave it None to match.
    """
    return Config(depth=depth, aspect_ratio=aspect_ratio, head_dim=head_dim,
                  seq_len=seq_len, vocab_size=vocab_size,
                  window_pattern=window_pattern, num_kv_heads=num_kv_heads,
                  rope_base=rope_base)


def compute_plan(depth: int, *, aspect_ratio: int = ASPECT_RATIO,
                 head_dim: int = HEAD_DIM, seq_len: int = 2048,
                 vocab_size: int, device_batch_size: int,
                 window_pattern: str = "SSSL",
                 num_kv_heads: int | None = None, rope_base: float = 100000.0,
                 total_batch_size: int = -1, num_iterations: int = 0,
                 target_flops: int = 0, target_param_data_ratio: int = 12,
                 **rates) -> TrainPlan:
    """Build the ``Config``, read the two C numbers, and derive the plan.

    The horizon, batch, learning-rate, and weight-decay arithmetic mirror
    ``scripts/base_train.py``. ``**rates`` overrides the base rates and the
    schedule; see ``_DEFAULT_RATES``.
    """
    selected = dict(_DEFAULT_RATES)
    selected.update(rates)
    unknown = set(selected) - set(_DEFAULT_RATES)
    if unknown:
        raise TypeError(f"unknown rate(s): {sorted(unknown)}")

    config = _reference_config(depth, aspect_ratio, head_dim, seq_len,
                               vocab_size, window_pattern, num_kv_heads,
                               rope_base)
    target = params(config)
    reference = params(_reference_config(
        12, aspect_ratio, head_dim, seq_len, vocab_size, window_pattern,
        num_kv_heads, rope_base))

    scaling_params = target.scaling_params
    target_tokens = int(target_param_data_ratio * scaling_params)
    d_ref = target_param_data_ratio * reference.scaling_params

    if total_batch_size == -1:
        batch_size_ratio = target_tokens / d_ref
        predicted = B_REF * batch_size_ratio**0.383
        total_batch_size = 2 ** round(math.log2(predicted))

    batch_ratio = total_batch_size / B_REF
    batch_lr_scale = batch_ratio**0.5 if batch_ratio != 1.0 else 1.0
    weight_decay_scaled = (selected["weight_decay"]
                           * math.sqrt(total_batch_size / B_REF)
                           * (d_ref / target_tokens))

    if num_iterations > 0:
        iterations = int(num_iterations)
    elif target_flops > 0:
        iterations = round(
            target_flops / (target.flops_per_token * total_batch_size))
    else:
        iterations = target_tokens // total_batch_size

    tokens_per_micro = device_batch_size * seq_len
    if tokens_per_micro <= 0:
        raise ValueError("device_batch_size and seq_len must be positive")
    if total_batch_size % tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_size ({total_batch_size}) must be a multiple of "
            f"device_batch_size * seq_len ({tokens_per_micro})")
    grad_accum = total_batch_size // tokens_per_micro

    optimizer = {
        "embedding_lr": selected["embedding_lr"] * batch_lr_scale,
        "unembedding_lr": selected["unembedding_lr"] * batch_lr_scale,
        "matrix_lr": selected["matrix_lr"] * batch_lr_scale,
        "scalar_lr": selected["scalar_lr"] * batch_lr_scale,
        "weight_decay": weight_decay_scaled,
    }
    scheduler = {
        "warmup_steps": selected["warmup_steps"],
        "warmdown_ratio": selected["warmdown_ratio"],
        "final_lr_frac": selected["final_lr_frac"],
        "weight_decay_base": weight_decay_scaled,
    }

    return TrainPlan(
        config=config,
        padded_vocab_size=config.padded_vocab_size,
        device_batch_size=int(device_batch_size),
        total_batch_size=int(total_batch_size),
        grad_accum=int(grad_accum),
        num_iterations=int(iterations),
        target_tokens=int(target_tokens),
        scaling_params=int(scaling_params),
        flops_per_token=float(target.flops_per_token),
        batch_lr_scale=float(batch_lr_scale),
        optimizer=optimizer,
        scheduler=scheduler,
    )
