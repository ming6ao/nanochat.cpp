"""The torch half of the reference-checkpoint conversion (Z2).

``nanochat_cpp.checkpoint`` (Z1) owns the checkpoint container: the magic, the
record framing, the dtype table, the name remap, the metadata block, and the
path resolution. This module owns the half that needs PyTorch: reading a
reference ``torch.save`` file, casting its tensors to the container dtype, and
reassembling the AdamW and Muon optimizer state.

The split is the boundary of ``docs/dependency-layering-plan.md`` section 4.
A conversion is an adapter in both directions, so the torch code lives here
and the core stays a standard-library package (section 11). The container
helpers are imported from the core rather than copied, so the byte layout is
single-sourced in ``src/data.cc`` through the C surface and a change lands in
one place.

Nothing here imports ``torch`` at import time. Each function that touches a
tensor imports it, so ``import nanochat_cpp_bridge.torch_checkpoint`` needs no
extra, ``--help`` works on a host with no torch, and ``--print-path`` resolves
a checkpoint without loading one.

Entry points
------------
``main(argv=None)``
    The converter command line. ``tools/convert_checkpoint.py`` forwards to it
    and the distribution declares it as a console script, so the two spellings
    share one implementation.
``build_records(state_dict, target)``
    A loaded ``state_dict`` to container records.
``build_optimizer_records(optimizer_state_dict)``
    A loaded optimizer ``state_dict`` to container records.
``build_parser()``
    The argument parser, for a host that wants to add its own options.

Run it directly::

    python3 tools/convert_checkpoint.py --source base --model-tag d20 \
        --dtype fp32 --verify

or import it::

    from nanochat_cpp_bridge import torch_checkpoint
    records = torch_checkpoint.build_records(state_dict, "float32")

Extras: ``reference`` (torch and numpy).
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nanochat_cpp.checkpoint import (
    CHECKPOINT_SOURCES,
    CheckpointError,
    ResolvedCheckpoint,
    TensorRecord,
    UnknownParameter,
    container_dtype,
    dtype_code,
    normalize_dtype,
    order_key,
    read_checkpoint_full,
    remap_name,
    resolve,
    write_checkpoint,
)

from . import _extras

__all__ = [
    "ADAMW_KINDS",
    "build_optimizer_records",
    "build_parser",
    "build_records",
    "main",
    "muon_state_name",
    "optimizer_order_key",
    "optimizer_state_records",
]


def _torch():
    """The ``torch`` module, imported where it is used.

    Kept in one place so the module imports with no extra installed and a
    missing wheel names the extras line instead of raising a bare
    ``ModuleNotFoundError``.
    """
    try:
        import torch  # noqa: PLC0415 - the lazy import of section 2.1
    except ImportError as error:
        raise _extras.missing("torch") from error
    return torch


# ---------------------------------------------------------------------------
# Conversion (lazy torch/numpy)
# ---------------------------------------------------------------------------


def _unwrap_state_dict(obj: Any) -> Mapping[str, Any]:
    """Peel common wrappers (``{"model": {...}}``, DDP, ``torch.compile``)."""
    depth = 0
    while isinstance(obj, Mapping) and obj and all(
            isinstance(value, Mapping) for value in obj.values()):
        wrapper = next(
            (key for key in ("model", "state_dict", "model_state_dict", "module")
             if key in obj),
            next(iter(obj)),
        )
        obj = obj[wrapper]
        depth += 1
        if depth > 8:
            break
    if not isinstance(obj, Mapping):
        raise CheckpointError(
            f"checkpoint does not contain a state dict ({type(obj).__name__})")
    return obj


def _load_state_dict(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load a ``state_dict`` from a ``.pt`` file on the CPU."""
    # The conversion is host-only; never initialise or touch a GPU for it.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch = _torch()

    last_error: Exception | None = None
    for kwargs in (
        {"map_location": "cpu", "mmap": True, "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ):
        try:
            return _unwrap_state_dict(torch.load(path, **kwargs))
        except Exception as error:  # noqa: BLE001 - try the next loader form
            last_error = error
    raise CheckpointError(f"cannot load {path}: {last_error}")


def _load_optimizer_state(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load a reference optimizer ``state_dict`` from a ``.pt`` file."""
    # The conversion is host-only; never initialise or touch a GPU for it.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch = _torch()

    last_error: Exception | None = None
    for kwargs in (
        {"map_location": "cpu", "mmap": True, "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ):
        try:
            loaded = torch.load(path, **kwargs)
        except Exception as error:  # noqa: BLE001 - try the next loader form
            last_error = error
            continue
        if not isinstance(loaded, Mapping):
            last_error = CheckpointError(
                f"optimizer shard is not a mapping ({type(loaded).__name__})")
            continue
        return loaded
    raise CheckpointError(
        f"cannot load optimizer state {path}: {last_error}")


def _tensor_payload(tensor: Any, target: str) -> bytes:
    """Cast a torch tensor to the container dtype and return little-endian bytes."""
    torch = _torch()

    cpu = tensor.detach().to(dtype=torch.float32, device="cpu")
    if target == "bfloat16":
        cpu = cpu.to(dtype=torch.bfloat16)
    array = cpu.contiguous().numpy()
    # The container is little-endian regardless of the host.
    return array.view(array.dtype.newbyteorder("<")).tobytes()


def build_records(state_dict: Mapping[str, Any], target: Any = "float32", *,
                  skip_unknown: bool = False,
                  skip_non_float: bool = False) -> list[TensorRecord]:
    """Convert a loaded ``state_dict`` into checkpoint-container records.

    ``target`` is the container dtype (``fp32``/``bf16``). Unknown keys raise
    (:class:`nanochat_cpp.checkpoint.UnknownParameter`) unless
    ``skip_unknown``; non-floating tensors raise unless ``skip_non_float``.
    """
    torch = _torch()

    target_name = container_dtype(target)
    code = dtype_code(target_name)

    unknown: list[str] = []
    non_float: list[str] = []
    records: list[TensorRecord] = []
    for key, tensor in state_dict.items():
        canonical = remap_name(key)
        if canonical is None:
            if skip_unknown:
                continue
            unknown.append(key)
            continue
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            if skip_non_float:
                continue
            non_float.append(key)
            continue
        records.append(TensorRecord(
            name=canonical,
            dtype=code,
            shape=tuple(int(dim) for dim in tensor.shape),
            data=_tensor_payload(tensor, target_name),
        ))
    if unknown:
        raise UnknownParameter(
            "state dict has parameters the C++ model does not define: "
            + ", ".join(sorted(unknown)))
    if non_float:
        raise CheckpointError(
            "state dict has non-floating tensors: "
            + ", ".join(sorted(non_float)))
    return records


# ---------------------------------------------------------------------------
# Optimizer state (docs/post-training.md section 7)
# ---------------------------------------------------------------------------
#
# The reference ``MuonAdamW`` saves a standard ``torch.optim`` state dict.
# ``GPT.setup_optimizer`` builds six AdamW groups and one Muon group per matrix
# shape, in a fixed order. The C++ optimizer names each AdamW group
# ``adamw.<kind>.m``/``adamw.<kind>.v`` and each stacked Muon group
# ``muon.<rows>x<cols>.buf1``/``muon.<rows>x<cols>.buf2``. The converter
# reassembles the per-parameter moments into the group buffers the C++ side
# restores, so the group names live with the builder that emits them.

#: The AdamW group tokens, in the reference ``setup_optimizer`` order.
ADAMW_KINDS = ("lm_head", "embedding", "value_embedding", "resid", "x0",
               "smear")

_MUON_NAME_RE = re.compile(r"^muon\.(\d+)x(\d+)\.(buf1|buf2)$")


def muon_state_name(rows: int, cols: int, buffer: str) -> str:
    """The C++ record name for one stacked Muon buffer."""
    return f"muon.{int(rows)}x{int(cols)}.{buffer}"


def optimizer_order_key(name: str) -> tuple:
    """A sort key that reproduces the C++ optimizer-state write order."""
    if name.startswith("adamw."):
        token = name[len("adamw."):].split(".", 1)[0]
        index = (ADAMW_KINDS.index(token) if token in ADAMW_KINDS
                 else len(ADAMW_KINDS))
        suffix = 0 if name.endswith(".m") else 1
        return (0, index, suffix)
    match = _MUON_NAME_RE.match(name)
    if match is not None:
        rows, cols = int(match.group(1)), int(match.group(2))
        suffix = 0 if match.group(3) == "buf1" else 1
        return (1, rows, cols, suffix)
    return (2, name)


def _optimizer_payload(tensor: Any) -> bytes:
    """fp32 little-endian bytes of one reference optimizer tensor."""
    # The C++ optimizer restores every moment buffer as fp32, so an fp16 or
    # bfloat16 reference buffer is upcast here.
    return _tensor_payload(tensor, "float32")


def _concat_optimizer_state(state: Mapping[Any, Any], params: Sequence[Any],
                            key: str) -> bytes | None:
    """Concatenate one moment key over the parameters of a group."""
    chunks: list[bytes] = []
    for index in params:
        entry = state.get(index)
        if not entry:
            continue
        tensor = entry.get(key)
        if tensor is None:
            continue
        chunks.append(_optimizer_payload(tensor))
    if not chunks:
        return None
    return b"".join(chunks)


def _muon_records(state: Mapping[Any, Any],
                  params: Sequence[Any]) -> list[TensorRecord]:
    """Build the two stacked Muon-state records for one shape group."""
    if not params:
        return []
    entry = state.get(params[0])
    if not entry:
        return []
    momentum = entry.get("momentum_buffer")
    second = entry.get("second_momentum_buffer")
    if momentum is None or second is None:
        return []
    shape = tuple(int(dim) for dim in momentum.shape)
    if len(shape) < 2:
        raise CheckpointError(
            "the reference Muon momentum buffer has a rank below two")
    if shape[0] != len(params):
        raise CheckpointError(
            "the reference Muon group stacks "
            f"{shape[0]} buffers for {len(params)} parameters; this is a "
            "sharded rank shard, not a single-rank shard")
    rows, cols = shape[-2], shape[-1]
    first = _optimizer_payload(momentum)
    last = _optimizer_payload(second)
    return [
        TensorRecord(muon_state_name(rows, cols, "buf1"),
                     dtype_code("float32"),
                     (len(first) // 4,), first),
        TensorRecord(muon_state_name(rows, cols, "buf2"),
                     dtype_code("float32"),
                     (len(last) // 4,), last),
    ]


def build_optimizer_records(
        optimizer_state_dict: Mapping[str, Any]) -> list[TensorRecord]:
    """Convert a reference optimizer ``state_dict`` into container records.

    The input is a ``MuonAdamW.state_dict()``: ``{"state": {index: {...}},
    "param_groups": [...]}``. The group order and ``kind`` match
    ``GPT.setup_optimizer``, so the AdamW groups map onto :data:`ADAMW_KINDS`
    in order and each Muon group maps onto its ``rows``x``cols`` name. Every
    record is fp32. A group without saved state is skipped.
    """
    state = optimizer_state_dict.get("state") or {}
    groups = optimizer_state_dict.get("param_groups") or ()
    records: list[TensorRecord] = []
    adamw_index = 0
    for group in groups:
        kind = str(group.get("kind", "adamw"))
        params = list(group.get("params") or ())
        if kind == "muon":
            records.extend(_muon_records(state, params))
            continue
        if adamw_index >= len(ADAMW_KINDS):
            raise CheckpointError(
                "the optimizer state has more AdamW groups than the C++ "
                f"optimizer defines ({len(ADAMW_KINDS)})")
        name = ADAMW_KINDS[adamw_index]
        adamw_index += 1
        first = _concat_optimizer_state(state, params, "exp_avg")
        second = _concat_optimizer_state(state, params, "exp_avg_sq")
        if first is None or second is None:
            continue
        records.append(TensorRecord(f"adamw.{name}.m",
                                    dtype_code("float32"),
                                    (len(first) // 4,), first))
        records.append(TensorRecord(f"adamw.{name}.v",
                                    dtype_code("float32"),
                                    (len(second) // 4,), second))
    return records


#: An alias that reads well at a call site that already says "optimizer".
optimizer_state_records = build_optimizer_records


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

_MODEL_STEP_RE = re.compile(r"model_(\d+)\.pt$")


def _step_from_model_path(path: str | os.PathLike[str]) -> int | None:
    """The step in a ``model_<step>.pt`` file name, or ``None``."""
    match = _MODEL_STEP_RE.search(Path(path).name)
    return int(match.group(1)) if match is not None else None


def _resolve_optimizer_path(args: argparse.Namespace,
                            resolved: ResolvedCheckpoint | None,
                            input_path: Path) -> Path | None:
    """Resolve the optimizer shard beside the model, or ``None`` when absent."""
    if args.optimizer is not None:
        return Path(args.optimizer)
    step = (resolved.step if resolved is not None
            else _step_from_model_path(input_path))
    if step is None:
        return None
    candidate = input_path.with_name(
        f"optim_{int(step):06d}_rank{int(args.rank)}.pt")
    return candidate if candidate.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    """The converter's argument parser."""
    parser = argparse.ArgumentParser(
        prog="convert_checkpoint",
        description="Convert a reference nanochat .pt state dict into the "
                    "checkpoint container the C++ runtime loads.")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="explicit model_<step>.pt; overrides --source/--model-tag/--step")
    parser.add_argument(
        "--source", choices=sorted(CHECKPOINT_SOURCES), default="base",
        help="checkpoint family (default: base)")
    parser.add_argument(
        "--model-tag", default=None,
        help="model tag such as d20 (default: the largest under the source)")
    parser.add_argument(
        "--step", type=int, default=None,
        help="training step (default: the last one present)")
    parser.add_argument(
        "--base-dir", type=Path, default=None,
        help="nanochat base directory (default: $NANOCHAT_BASE_DIR or "
             "~/.cache/nanochat)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output path (default: the input with a .nchkpt01 suffix)")
    parser.add_argument(
        "--optimizer", type=Path, default=None,
        help="explicit optim_<step>_rank<rank>.pt shard; overrides the "
             "automatic lookup beside the input")
    parser.add_argument(
        "--rank", type=int, default=0,
        help="optimizer shard rank for the automatic lookup (default: 0)")
    parser.add_argument(
        "--no-optimizer", action="store_true",
        help="convert the parameters only, even when an optimizer shard "
             "exists")
    parser.add_argument(
        "--dtype", default="float32",
        help="container dtype: float32 (default) or bfloat16")
    parser.add_argument(
        "--skip-unknown", action="store_true",
        help="drop state-dict keys the C++ model does not define")
    parser.add_argument(
        "--skip-non-float", action="store_true",
        help="drop non-floating tensors instead of failing")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing output file")
    parser.add_argument(
        "--print-path", action="store_true",
        help="print the resolved input path and exit without loading torch")
    parser.add_argument(
        "--verify", action="store_true",
        help="read the written container back and check names/shapes")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the summary")
    return parser


def _summary(origin: str, target: str, records: Sequence[TensorRecord],
             source_dtypes: Sequence[str]) -> str:
    """One line naming what the conversion wrote."""
    elements = sum(record.numel() for record in records)
    payload = sum(len(record.data) for record in records)
    sources = ", ".join(source_dtypes) if source_dtypes else "unknown"
    return (f"{origin} -> checkpoint container: {len(records)} tensors, "
            f"{elements:,} elements, {payload / (1024 * 1024):.1f} MiB, "
            f"dtype {target} (source {sources})")


def main(argv: Sequence[str] | None = None) -> int:
    """Convert one reference checkpoint; return the process exit code."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    # Validate the target dtype before touching any file.
    target = container_dtype(args.dtype)

    resolved: ResolvedCheckpoint | None = None
    if args.input is not None:
        input_path = Path(args.input)
    else:
        resolved = resolve(args.source, args.model_tag, args.step, args.base_dir)
        input_path = resolved.model_path

    if args.print_path:
        print(input_path)
        return 0

    if not input_path.is_file():
        raise CheckpointError(f"checkpoint not found: {input_path}")

    output = (Path(args.output) if args.output is not None
              else input_path.with_name(input_path.stem + ".nchkpt01"))
    if output.exists() and not args.force:
        raise CheckpointError(
            f"refusing to overwrite {output}; pass --force to replace it")

    state_dict = _load_state_dict(input_path)
    # Normalise the source dtypes before build_records casts them away.
    source_dtypes = sorted({
        normalize_dtype(tensor.dtype)
        for tensor in state_dict.values() if hasattr(tensor, "dtype")
    })
    records = build_records(
        state_dict,
        target,
        skip_unknown=args.skip_unknown,
        skip_non_float=args.skip_non_float,
    )
    records.sort(key=lambda record: order_key(record.name))

    # The optimizer shard is optional. The parameter-only path keeps working
    # when the shard is absent or when the caller passes --no-optimizer.
    optimizer_records: list[TensorRecord] = []
    optimizer_path: Path | None = None
    if not args.no_optimizer:
        optimizer_path = _resolve_optimizer_path(args, resolved, input_path)
        if optimizer_path is not None:
            if not optimizer_path.is_file():
                raise CheckpointError(
                    f"optimizer shard not found: {optimizer_path}")
            optimizer_records = build_optimizer_records(
                _load_optimizer_state(optimizer_path))
            optimizer_records.sort(
                key=lambda record: optimizer_order_key(record.name))

    write_checkpoint(output, records, optimizer_records)

    if args.verify:
        read_params, read_state = read_checkpoint_full(output)
        if [(r.name, r.shape, r.dtype) for r in read_params] != \
                [(r.name, r.shape, r.dtype) for r in records]:
            raise CheckpointError(f"verification failed for {output}")
        if [(r.name, r.shape, r.dtype) for r in read_state] != \
                [(r.name, r.shape, r.dtype) for r in optimizer_records]:
            raise CheckpointError(
                f"optimizer-state verification failed for {output}")

    if not args.quiet:
        origin = (f"{resolved.source}/{resolved.model_tag}/step "
                  f"{resolved.step}" if resolved is not None else "explicit")
        print(f"[convert_checkpoint] "
              f"{_summary(origin, target, records, source_dtypes)}")
        if optimizer_records:
            print(f"[convert_checkpoint] optimizer state: "
                  f"{len(optimizer_records)} records from {optimizer_path}")
        print(f"[convert_checkpoint] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
