"""Reference-checkpoint conversion into the ``NCHKPT01`` container.

The C++ runtime reads only its own self-describing container
(``docs/model.md``): a magic header and a flat list of
``(name, dtype, shape, raw little-endian payload)`` records. A released
nanochat checkpoint is a PyTorch ``state_dict`` saved with ``torch.save``. This
module bridges the two:

* **Name remap.** ``_orig_mod.`` (``torch.compile``) and ``module.`` (``DDP``)
  prefixes are stripped, legacy aliases are folded onto the canonical names the
  C++ model registers (``src/model.cc``), and anything else is rejected so a
  silently dropped weight cannot slip through.
* **Dtype policy.** The container stores only IEEE fp32 or fp16 (the values of
  ``nanochat::DType``). The source tensor is upcast to fp32 and then cast to the
  requested container dtype; the record carries the matching dtype code.
* **Path resolution.** ``source`` (``base``/``sft``/``rl``), ``--model-tag`` and
  ``--step`` resolve to ``model_<step:06d>.pt`` under ``NANOCHAT_BASE_DIR``,
  mirroring ``nanochat.checkpoint_manager``.

Everything in this module is stdlib-only at import time. ``torch`` and
``numpy`` are imported lazily inside the functions that touch tensors, so the
torch-free self-test can import and exercise the name, dtype, container, and
path logic without a GPU or PyTorch. The C++ runtime itself never imports
PyTorch; this is a one-time host-side conversion.

Run it directly::

    python3 tools/convert_checkpoint.py --source base --model-tag d20 --dtype fp32
    tools/nanochat_cpp checkpoint --source sft --model-tag d20 --step 499

or import the helpers::

    from nanochat_cpp import checkpoint
    resolved = checkpoint.resolve("base", "d20", step=1000)
    records = checkpoint.build_records(state_dict, "float32")
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Container format (must match src/data.cc)
# ---------------------------------------------------------------------------

#: ``kCheckpointMagic`` in ``src/data.cc``.
MAGIC = b"NCHKPT01"

#: ``Checkpoint::Save`` writes version 1.
VERSION = 1

#: Values of ``nanochat::DType`` (``include/nanochat/tensor.h``).
DTYPE_FP32 = 0
DTYPE_FP16 = 1

DTYPE_NAMES = {
    DTYPE_FP32: "float32",
    DTYPE_FP16: "float16",
}
DTYPE_CODES = {name: code for code, name in DTYPE_NAMES.items()}

#: ``Checkpoint::Load`` rejects ranks above 8.
MAX_RANK = 8

#: The nanochat directory layout (``nanochat/checkpoint_manager.py``).
CHECKPOINT_SOURCES = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


class CheckpointError(Exception):
    """A checkpoint could not be converted or parsed."""


class UnknownParameter(CheckpointError):
    """A state-dict key does not map onto a C++ model parameter."""


class UnsupportedDtype(CheckpointError):
    """A dtype is not one the NCHKPT01 container can store."""


# ---------------------------------------------------------------------------
# Name remap
# ---------------------------------------------------------------------------
#
# The reference ``GPT`` module tree (``nanochat/gpt.py``) and the C++ parameter
# registry (``src/model.cc``) use the same names for the current model. The
# table is explicit anyway: it documents the contract, normalises the wrapper
# prefixes that ``torch.compile`` and ``DDP`` add, and gives a home for the
# handful of legacy aliases.

#: Wrappers that are stripped repeatedly from the front of a key.
_WRAPPER_PREFIXES = ("_orig_mod.", "module.")

#: Canonical names, as registered by ``TrainModel``. ``\d+`` is a layer index.
_CANONICAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"lm_head\.weight",
        r"transformer\.wte\.weight",
        r"value_embeds\.\d+\.weight",
        r"resid_lambdas",
        r"x0_lambdas",
        r"smear_gate\.weight",
        r"smear_lambda",
        r"backout_lambda",
        r"transformer\.h\.\d+\.attn\.c_q\.weight",
        r"transformer\.h\.\d+\.attn\.c_k\.weight",
        r"transformer\.h\.\d+\.attn\.c_v\.weight",
        r"transformer\.h\.\d+\.attn\.c_proj\.weight",
        r"transformer\.h\.\d+\.attn\.ve_gate\.weight",
        r"transformer\.h\.\d+\.mlp\.c_fc\.weight",
        r"transformer\.h\.\d+\.mlp\.c_proj\.weight",
    )
)

#: (pattern, replacement) applied once, in order, after the wrappers.
_ALIASES = (
    # Older checkpoints that named the projection after the module it lives in.
    (re.compile(r"^transformer\.h\.(\d+)\.attn\.proj\.weight$"),
     r"transformer.h.\1.attn.c_proj.weight"),
    (re.compile(r"^transformer\.h\.(\d+)\.mlp\.proj\.weight$"),
     r"transformer.h.\1.mlp.c_proj.weight"),
)

_VALUE_EMBEDS_RE = re.compile(r"^value_embeds\.(\d+)\.weight$")
_LAYER_RE = re.compile(r"^transformer\.h\.(\d+)\.(attn|mlp)\.([a-z_]+)\.weight$")

# C++ parameter registration order (``TrainModel::TrainModel``): used to write a
# deterministic container that matches a C++ ``Checkpointer::SaveModel``.
_GLOBAL_ORDER = {
    "lm_head.weight": (0, 0),
    "transformer.wte.weight": (0, 1),
    "resid_lambdas": (1, 0),
    "x0_lambdas": (1, 1),
    "smear_gate.weight": (1, 2),
    "smear_lambda": (1, 3),
    "backout_lambda": (1, 4),
}
_LAYER_SUBORDER = {
    "c_q": 0,
    "c_k": 1,
    "c_v": 2,
    "c_proj": 3,
    "ve_gate": 4,
    "mlp.c_fc": 5,
    "mlp.c_proj": 6,
}


def strip_wrappers(name: str) -> str:
    """Remove repeated ``_orig_mod.``/``module.`` prefixes from ``name``."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def is_canonical_name(name: str) -> bool:
    """True if ``name`` is a name the C++ parameter registry knows."""
    return any(pattern.fullmatch(name) for pattern in _CANONICAL_PATTERNS)


def remap_name(name: str) -> str | None:
    """Map a reference state-dict key to its canonical C++ name.

    Returns ``None`` when the key names a parameter the C++ model does not
    have. Wrapper prefixes are stripped and legacy aliases are folded first.
    """
    candidate = strip_wrappers(name)
    for pattern, replacement in _ALIASES:
        if pattern.match(candidate):
            candidate = pattern.sub(replacement, candidate, count=1)
            break
    return candidate if is_canonical_name(candidate) else None


def canonical_name(name: str) -> str:
    """Like :func:`remap_name`, but raise :class:`UnknownParameter` on failure."""
    canonical = remap_name(name)
    if canonical is None:
        raise UnknownParameter(name)
    return canonical


def order_key(name: str) -> tuple[int, ...]:
    """A sort key that reproduces the C++ parameter registration order.

    ``name`` must already be canonical (see :func:`canonical_name`). The tuple
    is ``(group, layer, sub)``; global parameters use layer/sub of zero, value
    embeddings use group 0 and the layer index, and per-block matrices use
    group 2.
    """
    if name in _GLOBAL_ORDER:
        group, sub = _GLOBAL_ORDER[name]
        return (group, 0, sub)
    match = _VALUE_EMBEDS_RE.match(name)
    if match is not None:
        return (0, 2, int(match.group(1)))
    match = _LAYER_RE.match(name)
    if match is not None:
        layer = int(match.group(1))
        module = match.group(2)
        operator = match.group(3)
        sub = f"{module}.{operator}" if module == "mlp" else operator
        return (2, layer, _LAYER_SUBORDER[sub])
    raise UnknownParameter(name)


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------

#: Source dtype spellings (torch/numpy/plain) mapped to a canonical name.
_SOURCE_DTYPE_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "f32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "f16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "bfloat": "bfloat16",
    "float64": "float64",
    "fp64": "float64",
    "f64": "float64",
    "double": "float64",
}

#: Dtypes the NCHKPT01 container and the C++ runtime can store.
CONTAINER_DTYPES = ("float32", "float16")


def normalize_dtype(value: Any) -> str:
    """Canonicalise a dtype spelling such as ``"torch.bfloat16"`` or ``"half"``."""
    text = str(value).strip().lower()
    if text.startswith("torch."):
        text = text[len("torch."):]
    if text.startswith("numpy."):
        text = text[len("numpy."):]
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[len("<class '"):-len("'>")].rsplit(".", 1)[-1]
    if text not in _SOURCE_DTYPE_ALIASES:
        raise UnsupportedDtype(f"unsupported dtype: {value!r}")
    return _SOURCE_DTYPE_ALIASES[text]


def container_dtype(value: Any) -> str:
    """Validate a *target* dtype and return it as a container dtype name."""
    canonical = normalize_dtype(value)
    if canonical not in CONTAINER_DTYPES:
        raise UnsupportedDtype(
            f"{value!r} cannot be stored in NCHKPT01; expected one of "
            f"{', '.join(CONTAINER_DTYPES)}")
    return canonical


def dtype_code(name: Any) -> int:
    """The ``nanochat::DType`` code for a container dtype name."""
    canonical = container_dtype(name)
    return DTYPE_CODES[canonical]


def dtype_name(code: int) -> str:
    """The container dtype name for a ``nanochat::DType`` code."""
    try:
        return DTYPE_NAMES[int(code)]
    except (KeyError, ValueError, TypeError) as error:
        raise UnsupportedDtype(f"unknown dtype code: {code!r}") from error


@dataclasses.dataclass(frozen=True)
class Conversion:
    """What the dtype policy does to one tensor.

    ``source`` is the checkpoint's dtype, ``target`` the container dtype,
    ``code`` the ``nanochat::DType`` written into the record. ``upcast`` is true
    when precision (or exponent range) is gained, ``downcast`` when it is lost;
    ``lossy`` is true for any round trip that cannot be exact.
    """

    source: str
    target: str
    code: int
    upcast: bool
    downcast: bool
    lossy: bool


def plan_conversion(source: Any, target: Any) -> Conversion:
    """Derive the dtype policy for converting ``source`` into ``target``."""
    source_name = normalize_dtype(source)
    target_name = container_dtype(target)
    # Ordering by information content: float64 > float32 > float16, with
    # bfloat16 holding fp32's exponent range but fp16's mantissa width.
    wide = {"float64": 4, "float32": 3, "bfloat16": 2, "float16": 1}
    upcast = wide[source_name] < wide[target_name]
    downcast = wide[source_name] > wide[target_name]
    # A conversion is exact only when the source bit pattern embeds losslessly
    # in the target. bfloat16 and float16 are not comparable: bfloat16 keeps
    # fp32's exponent range but only fp16's mantissa, and vice versa.
    exact = source_name == target_name or (source_name, target_name) in {
        ("float16", "float32"),
        ("float16", "float64"),
        ("bfloat16", "float32"),
        ("bfloat16", "float64"),
        ("float32", "float64"),
    }
    lossy = not exact
    return Conversion(
        source=source_name,
        target=target_name,
        code=DTYPE_CODES[target_name],
        upcast=upcast,
        downcast=downcast,
        lossy=lossy,
    )


# ---------------------------------------------------------------------------
# Container reader/writer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorRecord:
    """One tensor in the container, matching ``nanochat::TensorRecord``."""

    name: str
    dtype: int
    shape: tuple[int, ...]
    data: bytes

    @property
    def dtype_name(self) -> str:
        return dtype_name(self.dtype)

    @property
    def element_size(self) -> int:
        return 4 if self.dtype == DTYPE_FP32 else 2

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def expected_bytes(self) -> int:
        return self.numel() * self.element_size


def _pack_record(record: TensorRecord) -> bytes:
    name_bytes = record.name.encode("utf-8")
    if len(name_bytes) > 0xFFFF:
        raise CheckpointError(f"record name is too long: {record.name!r}")
    if record.dtype not in DTYPE_NAMES:
        raise UnsupportedDtype(f"unknown dtype code: {record.dtype!r}")
    if len(record.shape) > MAX_RANK:
        raise CheckpointError(f"record rank is too large: {record.name!r}")
    for dim in record.shape:
        if dim < 0:
            raise CheckpointError(
                f"negative dimension in {record.name!r}: {dim}")
    if len(record.data) != record.expected_bytes():
        raise CheckpointError(
            f"payload size mismatch for {record.name!r}: "
            f"{len(record.data)} != {record.expected_bytes()}")
    parts = [
        struct.pack("<H", len(name_bytes)),
        name_bytes,
        struct.pack("<II", record.dtype, len(record.shape)),
    ]
    for dim in record.shape:
        parts.append(struct.pack("<q", int(dim)))
    parts.append(struct.pack("<Q", len(record.data)))
    parts.append(record.data)
    return b"".join(parts)


def serialize(records: Sequence[TensorRecord]) -> bytes:
    """Serialise records into the NCHKPT01 byte layout."""
    header = MAGIC + struct.pack("<II", VERSION, len(records))
    return header + b"".join(_pack_record(record) for record in records)


def write_checkpoint(path: str | os.PathLike[str],
                     records: Sequence[TensorRecord]) -> Path:
    """Write ``records`` to ``path`` atomically and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(serialize(records))
    os.replace(temporary, target)
    return target


def _need(data: bytes, offset: int, count: int, what: str) -> None:
    if count < 0 or offset + count > len(data):
        raise CheckpointError(f"truncated checkpoint while reading {what}")


def read_checkpoint(path: str | os.PathLike[str]) -> list[TensorRecord]:
    """Parse an NCHKPT01 file; the inverse of :func:`write_checkpoint`."""
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:8] != MAGIC:
        raise CheckpointError(f"bad checkpoint magic: {path}")
    version, count = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise CheckpointError(f"unsupported checkpoint version: {version}")
    offset = 16
    records: list[TensorRecord] = []
    for _ in range(count):
        _need(data, offset, 2, "record name length")
        (name_len,) = struct.unpack_from("<H", data, offset)
        offset += 2
        _need(data, offset, name_len, "record name")
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        _need(data, offset, 8, "record header")
        dtype, rank = struct.unpack_from("<II", data, offset)
        offset += 8
        if rank > MAX_RANK:
            raise CheckpointError(f"record rank is too large: {name!r}")
        _need(data, offset, 8 * rank, "record shape")
        shape = tuple(struct.unpack_from(f"<{rank}q", data, offset))
        offset += 8 * rank
        _need(data, offset, 8, "record payload size")
        (payload_bytes,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        _need(data, offset, payload_bytes, "record payload")
        payload = data[offset:offset + payload_bytes]
        offset += payload_bytes
        records.append(TensorRecord(name, dtype, shape, payload))
    return records


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def default_base_dir() -> Path:
    """``NANOCHAT_BASE_DIR`` or ``~/.cache/nanochat``."""
    override = os.environ.get("NANOCHAT_BASE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "nanochat"


def checkpoint_root(source: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    """The directory that holds one source's model tags."""
    try:
        subdir = CHECKPOINT_SOURCES[source]
    except KeyError as error:
        raise CheckpointError(
            f"unknown checkpoint source {source!r}; expected one of "
            f"{', '.join(sorted(CHECKPOINT_SOURCES))}") from error
    root = Path(base_dir) if base_dir is not None else default_base_dir()
    return root / subdir


def find_largest_model(checkpoints_dir: str | os.PathLike[str]) -> str:
    """Guess the model tag, preferring the deepest ``d<N>`` directory.

    Mirrors ``nanochat.checkpoint_manager.find_largest_model``: ``d<number>``
    tags win by depth, otherwise the most recently modified directory wins.
    Unlike the reference it only considers tags that actually hold a
    ``model_<step>.pt``, so an empty or optimizer-only directory is skipped.
    """
    directory = Path(checkpoints_dir)
    if not directory.is_dir():
        raise CheckpointError(f"no checkpoint directory: {directory}")
    tags = sorted(
        entry.name for entry in directory.iterdir()
        if entry.is_dir() and _has_model_file(entry))
    if not tags:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    candidates: list[tuple[int, str]] = []
    for tag in tags:
        match = re.match(r"d(\d+)", tag)
        if match:
            candidates.append((int(match.group(1)), tag))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][1]
    tags.sort(key=lambda tag: (directory / tag).stat().st_mtime, reverse=True)
    return tags[0]


def _has_model_file(directory: Path) -> bool:
    """True if ``directory`` holds at least one ``model_<step>.pt``."""
    return any(re.search(r"model_(\d+)\.pt$", entry.name)
               for entry in directory.iterdir() if entry.is_file())


def find_last_step(checkpoint_dir: str | os.PathLike[str]) -> int:
    """The highest step with a ``model_<step>.pt`` file in ``checkpoint_dir``."""
    directory = Path(checkpoint_dir)
    steps = []
    for entry in directory.iterdir():
        match = re.search(r"model_(\d+)\.pt$", entry.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    return max(steps)


@dataclasses.dataclass(frozen=True)
class ResolvedCheckpoint:
    """A concrete ``model_<step>.pt`` and its metadata sidecar."""

    source: str
    model_tag: str
    step: int
    model_path: Path
    meta_path: Path


def resolve(source: str = "base", model_tag: str | None = None,
            step: int | None = None,
            base_dir: str | os.PathLike[str] | None = None) -> ResolvedCheckpoint:
    """Resolve ``source``/``model_tag``/``step`` to concrete checkpoint paths.

    Missing ``model_tag`` picks the largest model and missing ``step`` picks the
    last step, exactly as ``nanochat.checkpoint_manager.load_model_from_dir``.
    """
    root = checkpoint_root(source, base_dir)
    tag = model_tag if model_tag is not None else find_largest_model(root)
    tag_dir = root / tag
    if step is None:
        step = find_last_step(tag_dir)
    step = int(step)
    return ResolvedCheckpoint(
        source=source,
        model_tag=tag,
        step=step,
        model_path=tag_dir / f"model_{step:06d}.pt",
        meta_path=tag_dir / f"meta_{step:06d}.json",
    )


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
    import torch  # Local import: keep this module importable without torch.

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


def _tensor_payload(tensor: Any, target: str) -> bytes:
    """Cast a torch tensor to the container dtype and return little-endian bytes."""
    import torch

    cpu = tensor.detach().to(dtype=torch.float32, device="cpu")
    if target == "float16":
        cpu = cpu.to(dtype=torch.float16)
    array = cpu.contiguous().numpy()
    # The container is little-endian regardless of the host.
    return array.view(array.dtype.newbyteorder("<")).tobytes()


def build_records(state_dict: Mapping[str, Any], target: Any = "float32", *,
                  skip_unknown: bool = False,
                  skip_non_float: bool = False) -> list[TensorRecord]:
    """Convert a loaded ``state_dict`` into NCHKPT01 records.

    ``target`` is the container dtype (``fp32``/``fp16``). Unknown keys raise
    (:class:`UnknownParameter`) unless ``skip_unknown``; non-floating tensors
    raise unless ``skip_non_float``.
    """
    import torch

    target_name = container_dtype(target)
    code = DTYPE_CODES[target_name]

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
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_checkpoint",
        description="Convert a reference nanochat .pt state dict into the "
                    "NCHKPT01 container the C++ runtime loads.")
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
        "--dtype", default="float32",
        help="container dtype: float32 (default) or float16")
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
    elements = sum(record.numel() for record in records)
    payload = sum(len(record.data) for record in records)
    sources = ", ".join(source_dtypes) if source_dtypes else "unknown"
    return (f"{origin} -> NCHKPT01: {len(records)} tensors, "
            f"{elements:,} elements, {payload / (1024 * 1024):.1f} MiB, "
            f"dtype {target} (source {sources})")


def main(argv: Sequence[str] | None = None) -> int:
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
    write_checkpoint(output, records)

    if args.verify:
        read_back = read_checkpoint(output)
        if [(r.name, r.shape, r.dtype) for r in read_back] != \
                [(r.name, r.shape, r.dtype) for r in records]:
            raise CheckpointError(f"verification failed for {output}")

    if not args.quiet:
        origin = (f"{resolved.source}/{resolved.model_tag}/step "
                  f"{resolved.step}" if resolved is not None else "explicit")
        print(f"[convert_checkpoint] "
              f"{_summary(origin, target, records, source_dtypes)}")
        print(f"[convert_checkpoint] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
