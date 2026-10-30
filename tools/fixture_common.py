#!/usr/bin/env python3
"""Shared helpers for the offline fixture generators.

The merged generator ``tools/dump_fixtures.py`` keeps one dispatch and one
usage line for every fixture kind (``docs/script-consolidation-plan.md``
section 7, P4). This module holds the code that the kinds used to repeat:
the subcommand selection and the small value helpers.

The module imports no third-party package at import time. Each helper imports
the reference package it needs, so a kind that does not touch PyTorch stays
fast.
"""

from __future__ import annotations

import sys

__all__ = [
    "COMMANDS",
    "USAGE",
    "as_f32",
    "as_i32",
    "build_config",
    "scalar",
    "selected_command",
]

#: Every fixture subcommand, in help order.
COMMANDS = (
    "oracle",
    "train",
    "rl",
    "generate",
    "eval",
    "plan",
    "tokenizer",
    "loader",
    "chat",
    "api",
    "numerics",
    "checkpoint",
)

#: The usage line for the merged generator.
USAGE = "usage: dump_fixtures.py {" + "|".join(COMMANDS) + "} [options]"


def selected_command(argv: list[str]) -> str:
    """The fixture kind from ``argv``, or ``SystemExit`` with the usage line."""
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        raise SystemExit(0 if argv else 2)
    command = argv[0]
    if command not in COMMANDS:
        print("unknown fixture kind: " + command, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    return command


def as_f32(tensor):
    """The tensor as a contiguous fp32 NumPy array."""
    import torch
    return tensor.detach().to(torch.float32).cpu().numpy()


def as_i32(tensor):
    """The tensor as a contiguous int32 NumPy array."""
    import torch
    return tensor.detach().to(torch.int32).cpu().numpy()


def scalar(value: float):
    """One float as a one-element fp32 NumPy array."""
    import numpy as np
    return np.asarray([value], dtype=np.float32)


def build_config(args, config_class):
    """The reference ``GPTConfig`` from the parsed arguments."""
    return config_class(
        sequence_len=args.seq,
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
        window_pattern=args.window,
    )
