"""ctypes declarations for the nanochat C application binary interface.

This module declares every C struct and function in
``include/nanochat/capi.h`` with the standard ``ctypes`` module, then turns a
nonzero ``nanochat_status`` value into a Python exception that carries the
message from ``nanochat_last_error`` (docs/python.md section 4.2).

The module imports only the Python standard library. ``declare`` sets the
argument and result types on one loaded shared library; ``load`` itself lives
in ``nanochat_cpp._lib`` so that importing this module never loads the
library.
"""

from __future__ import annotations

import ctypes
import struct

_struct = struct

__all__ = [
    "NanochatError",
    "NANOCHAT_STATUS_OK",
    "NANOCHAT_STATUS_ERROR",
    "Config",
    "Device",
    "Param",
    "Params",
    "Focus",
    "ScoreResult",
    "GenerateParams",
    "Sequences",
    "OptimizerConfig",
    "Metrics",
    "MeshTrainConfig",
    "MeshTrainDataConfig",
    "MeshTrainRunConfig",
    "MeshTrainOutputConfig",
    "MeshTrainSummary",
    "ModelHandle",
    "OptimHandle",
    "LoaderHandle",
    "TokenizerHandle",
    "MeshHandle",
    "RuntimeHandle",
    "CheckpointHandle",
    "Record",
    "CheckpointMetadata",
    "NANOCHAT_STAGE_BASE",
    "NANOCHAT_STAGE_SFT",
    "NANOCHAT_STAGE_RL",
    "NANOCHAT_PRECISION_FP32",
    "NANOCHAT_PRECISION_BF16",
    "NANOCHAT_PRECISION_FP8",
    "declare",
    "backend_name",
    "compute_type_size",
    "device_info",
    "device_count",
    "device_at",
    "last_error",
    "check_status",
    "check_handle",
    "check_code",
    "record_view",
    "set_grad_enabled",
]

#: A ``nanochat_status`` value that means success.
NANOCHAT_STATUS_OK = 0
#: A ``nanochat_status`` value that means failure.
NANOCHAT_STATUS_ERROR = 1

#: The ``nanochat_stage`` values of the metadata block, in stored order. They
#: match ``nanochat::Stage`` and never change.
NANOCHAT_STAGE_BASE = 0
NANOCHAT_STAGE_SFT = 1
NANOCHAT_STAGE_RL = 2

#: The ``nanochat_precision`` values of the metadata block, in stored order.
#: They match ``nanochat::ComputePrecision`` and never change.
NANOCHAT_PRECISION_FP32 = 0
NANOCHAT_PRECISION_BF16 = 1
NANOCHAT_PRECISION_FP8 = 2

#: One process-wide cache of the declared function table, keyed by the shared
#: library object identity. ``declare`` is idempotent, but the cache keeps a
#: second call cheap.
_declared: "set[int]" = set()


class NanochatError(RuntimeError):
    """A nonzero ``nanochat_status`` from the C library.

    ``function`` names the entry point that failed and ``message`` is the
    thread-local text from ``nanochat_last_error``. The C layer already
    prefixes that text with the function name, so the constructor adds the
    name only for a message that does not carry it.
    """

    def __init__(self, function: str, message: str) -> None:
        self.function = function
        self.message = message
        if message and function and not message.startswith(function + ":"):
            text = f"{function}: {message}"
        else:
            text = message or function
        super().__init__(text)


# ---------------------------------------------------------------------------
# Plain C structs, a field-for-field mirror of include/nanochat/capi.h
# ---------------------------------------------------------------------------


class Config(ctypes.Structure):
    """A mirror of ``nanochat::Config`` (``nanochat_config``)."""

    _fields_ = [
        ("num_layers", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("seq_len", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("padded_vocab_size", ctypes.c_int),
        ("rope_base", ctypes.c_float),
        ("window_pattern", ctypes.c_char_p),
        ("value_embedding", ctypes.c_int),
    ]


class Params(ctypes.Structure):
    """A mirror of ``nanochat::ParamBreakdown`` (``nanochat_params``).

    ``total`` counts every allocated parameter. ``transformer_matrices`` and
    ``lm_head`` mirror the reference ``GPT.num_scaling_params``; their sum is
    the scaling-parameter count that sets the training horizon.
    """

    _fields_ = [
        ("total", ctypes.c_int64),
        ("transformer_matrices", ctypes.c_int64),
        ("lm_head", ctypes.c_int64),
        ("embeddings", ctypes.c_int64),
        ("scalars", ctypes.c_int64),
        ("flops_per_token", ctypes.c_double),
    ]


class Param(ctypes.Structure):
    """A mirror of ``nanochat::ParamView`` (``nanochat_param``)."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_void_p),
        ("grad", ctypes.c_void_p),
        ("count", ctypes.c_int64),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
    ]


class Device(ctypes.Structure):
    """A partial mirror of ``nanochat::Caps`` (``nanochat_device``).

    The backend itself comes from ``nanochat_backend``; this struct carries
    the device index and diagnostics.
    """

    _fields_ = [
        ("device_index", ctypes.c_int),
        ("compute_major", ctypes.c_int),
        ("compute_minor", ctypes.c_int),
        ("total_memory_bytes", ctypes.c_int64),
        ("device_name", ctypes.c_char_p),
    ]


class Focus(ctypes.Structure):
    """A mirror of ``nanochat::ScoreFocus`` (``nanochat_focus``)."""

    _fields_ = [
        ("position", ctypes.c_int),
        ("ids", ctypes.POINTER(ctypes.c_int)),
        ("count", ctypes.c_int),
    ]


class ScoreResult(ctypes.Structure):
    """A mirror of ``nanochat::ScoreResult`` (``nanochat_score_result``)."""

    _fields_ = [
        ("nll", ctypes.POINTER(ctypes.c_float)),
        ("argmax", ctypes.POINTER(ctypes.c_int)),
        ("focus_logits", ctypes.POINTER(ctypes.c_float)),
    ]


class GenerateParams(ctypes.Structure):
    """A mirror of ``nanochat::GenerateParams`` (``nanochat_generate_params``)."""

    _fields_ = [
        ("num_samples", ctypes.c_int),
        ("max_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("stop_id", ctypes.c_int),
        ("bos_id", ctypes.c_int),
        ("stop_ids", ctypes.POINTER(ctypes.c_int)),
    ]


class Sequences(ctypes.Structure):
    """A mirror of ``nanochat_sequences``, the generated rows."""

    _fields_ = [
        ("count", ctypes.c_int),
        ("tokens", ctypes.POINTER(ctypes.c_int)),
        ("mask", ctypes.POINTER(ctypes.c_uint8)),
        ("lengths", ctypes.POINTER(ctypes.c_int)),
        ("offsets", ctypes.POINTER(ctypes.c_int)),
    ]


class OptimizerConfig(ctypes.Structure):
    """The combined optimizer and scheduler hyperparameters.

    The field order matches ``nanochat_optim_config`` exactly: the
    ``nanochat::OptimizerConfig`` group first, then the
    ``nanochat::SchedulerConfig`` group.
    """

    _fields_ = [
        ("unembedding_lr", ctypes.c_float),
        ("embedding_lr", ctypes.c_float),
        ("matrix_lr", ctypes.c_float),
        ("scalar_lr", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("muon_ns_steps", ctypes.c_int),
        ("muon_beta2", ctypes.c_float),
        ("adam_eps", ctypes.c_float),
        ("clip", ctypes.c_float),
        ("num_iterations", ctypes.c_int),
        ("warmup_steps", ctypes.c_int),
        ("warmdown_ratio", ctypes.c_float),
        ("final_lr_frac", ctypes.c_float),
        ("weight_decay_base", ctypes.c_float),
        ("muon_momentum_warmup_steps", ctypes.c_float),
        ("muon_momentum_start", ctypes.c_float),
        ("muon_momentum_peak", ctypes.c_float),
        ("muon_momentum_final", ctypes.c_float),
    ]


class Metrics(ctypes.Structure):
    """A mirror of ``nanochat::MeshMetrics`` (``nanochat_metrics``).

    ``loss`` points to ``device_count`` per-replica losses. The buffer belongs
    to the runtime and stays valid until the next step or the next call to
    ``nanochat_runtime_metrics``.
    """

    _fields_ = [
        ("device_count", ctypes.c_int),
        ("loss", ctypes.POINTER(ctypes.c_float)),
        ("grad_norm", ctypes.c_float),
        ("step_seconds", ctypes.c_float),
    ]


class MeshTrainDataConfig(ctypes.Structure):
    """A mirror of ``nanochat_mesh_data_config``: the dataset and the
    tokenizer (docs/client-runtime-design.md section 10.1).
    """

    _fields_ = [
        ("parquet", ctypes.POINTER(ctypes.c_char_p)),
        ("parquet_count", ctypes.c_int),
        ("tokenizer_path", ctypes.c_char_p),
        ("packed_rows_path", ctypes.c_char_p),
        ("data_seed", ctypes.c_uint64),
    ]


class MeshTrainRunConfig(ctypes.Structure):
    """A mirror of ``nanochat_mesh_run_config``: the step loop
    (docs/client-runtime-design.md section 10.1).
    """

    _fields_ = [
        ("batch", ctypes.c_int),
        ("grad_accum", ctypes.c_int),
        ("seq", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("num_iterations", ctypes.c_int),
        ("log_every", ctypes.c_int),
        ("save_every", ctypes.c_int),
        ("resume", ctypes.c_int),
    ]


class MeshTrainOutputConfig(ctypes.Structure):
    """A mirror of ``nanochat_mesh_output_config``: the written artifacts
    (docs/client-runtime-design.md section 10.1).
    """

    _fields_ = [
        ("out_dir", ctypes.c_char_p),
        ("checkpoint_path", ctypes.c_char_p),
    ]


class MeshTrainConfig(ctypes.Structure):
    """A mirror of ``nanochat_mesh_train_config``
    (docs/client-runtime-design.md section 10.1). The caller owns every
    pointer, the mesh handle, and the struct, so the C call reads it only.
    """

    _fields_ = [
        ("model", Config),
        ("mesh", ctypes.c_void_p),
        ("preset", ctypes.c_char_p),
        ("optimizer", ctypes.POINTER(OptimizerConfig)),
        ("warm_start_path", ctypes.c_char_p),
        ("flops_per_token", ctypes.c_double),
        ("peak_flops", ctypes.c_double),
        ("data", MeshTrainDataConfig),
        ("run", MeshTrainRunConfig),
        ("output", MeshTrainOutputConfig),
    ]


class MeshTrainSummary(ctypes.Structure):
    """A mirror of ``nanochat_mesh_train_summary``
    (docs/client-runtime-design.md section 10.1). The caller owns the struct,
    so ``checkpoint`` is a fixed buffer and not an owned pointer.
    """

    _fields_ = [
        ("steps", ctypes.c_int),
        ("interrupted", ctypes.c_int),
        ("mean_loss", ctypes.c_double),
        ("steady_mfu", ctypes.c_double),
        ("checkpoint", ctypes.c_char * 4096),
    ]


class Record(ctypes.Structure):
    """A mirror of ``nanochat_record``.

    ``dims`` holds ``ndim`` extents, ``data`` holds ``bytes`` payload bytes,
    and ``attributes`` points to ``attribute_count`` records. Every pointer
    belongs to the checkpoint owner, or to the caller of the write call.
    """

    pass


Record._fields_ = [
    ("name", ctypes.c_char_p),
    ("dtype", ctypes.c_int32),
    ("ndim", ctypes.c_int32),
    ("dims", ctypes.POINTER(ctypes.c_int64)),
    ("data", ctypes.c_void_p),
    ("bytes", ctypes.c_int64),
    ("attribute_count", ctypes.c_int64),
    ("attributes", ctypes.POINTER(Record)),
]


class CheckpointMetadata(ctypes.Structure):
    """A mirror of ``nanochat_checkpoint_metadata``.

    ``schema`` is the metadata schema, not the container version, and ``stage``
    and ``precision`` hold a ``nanochat_stage`` and a ``nanochat_precision``
    code. Every string pointer belongs to the checkpoint handle that the read
    returned, ``notes`` points to ``note_count`` entries, and the caller must
    free neither.
    """

    _fields_ = [
        ("schema", ctypes.c_int32),
        ("stage", ctypes.c_int32),
        ("step", ctypes.c_int64),
        ("world_size", ctypes.c_int64),
        ("precision", ctypes.c_int32),
        ("config_json", ctypes.c_char_p),
        ("tokenizer_id", ctypes.c_char_p),
        ("git_revision", ctypes.c_char_p),
        ("created_utc", ctypes.c_char_p),
        ("notes", ctypes.POINTER(ctypes.c_char_p)),
        ("note_count", ctypes.c_int32),
    ]


# Opaque handles. The C declarations use ``struct nanochat_model *`` and
# friends; a ``c_void_p`` carries the same pointer width.
ModelHandle = ctypes.c_void_p
OptimHandle = ctypes.c_void_p
LoaderHandle = ctypes.c_void_p
TokenizerHandle = ctypes.c_void_p
MeshHandle = ctypes.c_void_p
RuntimeHandle = ctypes.c_void_p
CheckpointHandle = ctypes.c_void_p


#: The kind tag of a tokenizer handle. The mint site and the check read
#: this one name, so a typo cannot mint a handle no consumer accepts.
KIND_TOKENIZER = "tokenizer"
KIND_MODEL = "model"
KIND_OPTIMIZER = "optimizer"
KIND_LOADER = "loader"
KIND_MESH = "mesh"
KIND_RUNTIME = "runtime"


class Handle(int):
    """A C handle that `_core` mints, tagged with its kind.

    An opaque pointer is an ``int`` in Python, so every int looked like a
    handle: a path string reached C as a character pointer and the encoder
    dereferenced it, which killed the process with SIGSEGV. A plain int for a
    handle would also let two objects own one pointer, and then free it twice.

    Subclassing ``int`` keeps every ctypes call working unchanged, because a
    ``c_void_p`` argument already accepts an int. ``kind`` is checked by the
    constructor that takes a handle from a caller, and ``owner`` makes a second
    claim a failure instead of a double free.

    The tag is a guard against pointer-shaped mistakes, not a capability
    boundary: ``_core`` is importable, so a forged ``Handle(42, "tokenizer")``
    still passes the tag check. It cannot prevent a caller who reaches for the
    private module on purpose; it prevents the accident.
    """

    def __new__(cls, raw, kind: str):
        handle = int.__new__(cls, int(raw))
        handle.kind = str(kind)
        handle.owner = None
        handle.freed = False
        return handle

    def __repr__(self) -> str:
        # The raw address stays out of logs on purpose: it is an ASLR leak, and
        # the tag and the owner are what a reader needs.
        state = "freed" if self.freed else (self.owner or "unowned")
        return f"Handle(kind={self.kind!r}, {state})"

    def __copy__(self) -> "Handle":
        # A copy must share one `freed` flag, or two Python objects would free
        # the same address.
        return self

    def __deepcopy__(self, memo) -> "Handle":
        return self

    def claim(self, owner: str) -> "Handle":
        """Take ownership. A second claim, or a freed handle, is a failure.

        Strict, because a second owner would free the pointer a second time,
        and a freed pointer may already name someone else's memory.
        """
        if self.freed:
            raise TypeError(
                f"this {self.kind} handle was freed; load or create a new one")
        if self.owner is not None:
            raise TypeError(
                f"this {self.kind} handle already belongs to {self.owner}; "
                "load or create a second one instead of sharing this one")
        self.owner = owner
        return self

    def detach(self) -> int:
        """Give up the raw pointer, so a second free cannot reach it."""
        if self.freed:
            raise TypeError(f"this {self.kind} handle was already freed")
        raw = int(self)
        self.owner = None
        self.freed = True
        return raw

# The function table: name, argument types, result type. Every entry in
# include/nanochat/capi.h is here.
_SIGNATURES = (
    ("nanochat_init", (), ctypes.c_int),
    ("nanochat_last_error", (), ctypes.c_char_p),
    ("nanochat_version", (), ctypes.c_char_p),
    ("nanochat_backend", (), ctypes.c_char_p),
    ("nanochat_compute_type_size", (), ctypes.c_int),
    ("nanochat_device_info", (ctypes.POINTER(Device),), None),
    ("nanochat_device_count", (), ctypes.c_int),
    ("nanochat_device_at", (ctypes.c_int, ctypes.POINTER(Device)),
     ctypes.c_int),
    ("nanochat_mesh_create",
     (ctypes.POINTER(ctypes.c_int), ctypes.c_int), MeshHandle),
    ("nanochat_mesh_free", (MeshHandle,), None),
    ("nanochat_mesh_size", (MeshHandle,), ctypes.c_int),
    ("nanochat_runtime_create",
     (MeshHandle, ctypes.POINTER(Config),
      ctypes.POINTER(OptimizerConfig), ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_uint64), RuntimeHandle),
    ("nanochat_runtime_free", (RuntimeHandle,), None),
    ("nanochat_runtime_world_size", (RuntimeHandle,), ctypes.c_int),
    ("nanochat_runtime_step", (RuntimeHandle,), ctypes.c_int),
    ("nanochat_runtime_train_step",
     (RuntimeHandle, ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int)), ctypes.c_float),
    ("nanochat_runtime_metrics",
     (RuntimeHandle, ctypes.POINTER(Metrics)), ctypes.c_int),
    ("nanochat_runtime_param_count",
     (RuntimeHandle, ctypes.c_int), ctypes.c_int),
    ("nanochat_runtime_param_info",
     (RuntimeHandle, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Param)),
     ctypes.c_int),
    ("nanochat_runtime_param_read",
     (RuntimeHandle, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p), ctypes.c_int64),
    ("nanochat_runtime_param_digest",
     (RuntimeHandle, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)),
     ctypes.c_int),
    ("nanochat_runtime_save", (RuntimeHandle, ctypes.c_char_p), None),
    ("nanochat_runtime_load", (RuntimeHandle, ctypes.c_char_p), None),
    ("nanochat_runtime_load_weights", (RuntimeHandle, ctypes.c_char_p),
     None),
    ("nanochat_mesh_train_run",
     (ctypes.POINTER(MeshTrainConfig), ctypes.POINTER(MeshTrainSummary)),
     ctypes.c_float),
    ("nanochat_params_get",
     (ctypes.POINTER(Config), ctypes.POINTER(Params)), ctypes.c_int),
    ("nanochat_model_create",
     (ctypes.POINTER(Config), ctypes.c_uint64), ModelHandle),
    ("nanochat_model_free", (ModelHandle,), None),
    ("nanochat_forward_loss",
     (ModelHandle, ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int),
     ctypes.c_float),
    ("nanochat_backward", (ModelHandle,), None),
    ("nanochat_backward_weighted",
     (ModelHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_float), None),
    ("nanochat_backward_accumulate", (ModelHandle, ctypes.c_float), None),
    ("nanochat_train_step",
     (ModelHandle, OptimHandle, ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int),
     ctypes.c_float),
    ("nanochat_zero_grad", (ModelHandle,), None),
    ("nanochat_param_count", (ModelHandle,), ctypes.c_int),
    ("nanochat_param_info",
     (ModelHandle, ctypes.c_int, ctypes.POINTER(Param)), ctypes.c_int),
    ("nanochat_param_read",
     (ModelHandle, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
      ctypes.c_int64, ctypes.c_void_p), ctypes.c_int64),
    ("nanochat_param_write",
     (ModelHandle, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
      ctypes.c_int64, ctypes.c_void_p), ctypes.c_int64),
    ("nanochat_save", (ModelHandle, ctypes.c_char_p), None),
    ("nanochat_load", (ModelHandle, ctypes.c_char_p), None),
    ("nanochat_optim_create",
     (ModelHandle, ctypes.POINTER(OptimizerConfig)), OptimHandle),
    ("nanochat_optim_config_preset",
     (ctypes.c_char_p, ctypes.POINTER(OptimizerConfig)), ctypes.c_int),
    ("nanochat_optim_step", (OptimHandle, ctypes.c_int), None),
    ("nanochat_optim_grad_norm", (OptimHandle,), ctypes.c_float),
    ("nanochat_optim_free", (OptimHandle,), None),
    ("nanochat_loader_create",
     (ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_char_p,
      TokenizerHandle, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
      ctypes.c_int, ctypes.c_size_t),
     LoaderHandle),
    ("nanochat_loader_next",
     (LoaderHandle, ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int)), ctypes.c_int),
    ("nanochat_loader_token_bytes",
     (LoaderHandle, ctypes.POINTER(ctypes.c_int)),
     ctypes.POINTER(ctypes.c_uint8)),
    ("nanochat_loader_free", (LoaderHandle,), None),
    ("nanochat_score_batch",
     (ModelHandle, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Focus),
      ctypes.POINTER(ScoreResult)),
     None),
    ("nanochat_generate",
     (ModelHandle, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
      ctypes.POINTER(GenerateParams), ctypes.POINTER(Sequences)),
     None),
    ("nanochat_eval_bpb",
     (ModelHandle, LoaderHandle, ctypes.c_int), ctypes.c_float),
    ("nanochat_tokenizer_load", (ctypes.c_char_p,), TokenizerHandle),
    ("nanochat_encode",
     (TokenizerHandle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
      ctypes.c_int),
     ctypes.c_int),
    ("nanochat_decode",
     (TokenizerHandle, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
      ctypes.c_void_p, ctypes.c_int),
     ctypes.c_int),
    ("nanochat_tokenizer_free", (TokenizerHandle,), None),
    ("nanochat_checkpoint_dtype",
     (ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p),
      ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)),
     ctypes.c_int),
    ("nanochat_checkpoint_write",
     (ctypes.c_char_p, ctypes.POINTER(Record), ctypes.c_int64,
      ctypes.POINTER(Record), ctypes.c_int64),
     ctypes.c_int),
    ("nanochat_checkpoint_read",
     (ctypes.c_char_p, ctypes.POINTER(CheckpointHandle),
      ctypes.POINTER(ctypes.POINTER(Record)), ctypes.POINTER(ctypes.c_int64),
      ctypes.POINTER(ctypes.POINTER(Record)), ctypes.POINTER(ctypes.c_int64)),
     ctypes.c_int),
    ("nanochat_checkpoint_free", (CheckpointHandle,), None),
    ("nanochat_checkpoint_metadata_read",
     (ctypes.c_char_p, ctypes.POINTER(CheckpointHandle),
      ctypes.POINTER(CheckpointMetadata)),
     ctypes.c_int),
    ("nanochat_checkpoint_metadata_stage",
     (ctypes.POINTER(CheckpointMetadata),), ctypes.c_int32),
    ("nanochat_checkpoint_metadata_step",
     (ctypes.POINTER(CheckpointMetadata),), ctypes.c_int64),
    ("nanochat_checkpoint_metadata_world_size",
     (ctypes.POINTER(CheckpointMetadata),), ctypes.c_int64),
    ("nanochat_checkpoint_metadata_config_json",
     (ctypes.POINTER(CheckpointMetadata),), ctypes.c_char_p),
)


def declare(library) -> None:
    """Set the argument and result types for every C entry point.

    ``declare`` is idempotent. It skips a symbol that the library does not
    export, so an older or a narrower build still loads. A missing required
    symbol then fails at the call site, not at import time.
    """
    if id(library) in _declared:
        return
    for name, argtypes, restype in _SIGNATURES:
        function = getattr(library, name, None)
        if function is None:
            continue
        function.argtypes = list(argtypes)
        function.restype = restype
    _declared.add(id(library))


def last_error(library) -> str:
    """The thread-local error text, or an empty string when none is set."""
    value = library.nanochat_last_error()
    if not value:
        return ""
    return value.decode("utf-8", "replace")


def backend_name(library) -> str:
    """The active backend: ``"cpu"`` or ``"cuda"``."""
    value = library.nanochat_backend()
    return value.decode("utf-8", "replace") if value else ""


def compute_type_size(library) -> int:
    """The parameter element size in bytes: four for fp32, two for bf16."""
    return int(library.nanochat_compute_type_size())


#: The compute element sizes the Python surface can decode.
SUPPORTED_COMPUTE_SIZES = (2, 4)


def compute_type_name(element_size: int) -> str:
    """The compute type name for an element size: ``"bf16"`` or ``"fp32"``."""
    return "bf16" if int(element_size) == 2 else "fp32"


def compute_buffer(element_size: int, count: int):
    """A ctypes buffer of ``count`` compute elements.

    The buffer holds the build compute type, so the byte layout is the one the C
    surface copies (docs/precision.md section 5).
    """
    element_size = int(element_size)
    if element_size == 4:
        return (ctypes.c_float * int(count))()
    if element_size == 2:
        return (ctypes.c_uint16 * int(count))()
    raise NanochatError(
        "compute_buffer",
        f"{element_size} bytes per element is not a supported compute type",
    )


def compute_values(buffer, element_size: int) -> list:
    """The float values of a compute buffer.

    A four-byte buffer already holds fp32. A two-byte buffer holds bf16, the
    upper half of the fp32 bit pattern, so the decode is a shift and a bit
    reinterpretation.
    """
    if int(element_size) == 4:
        return [float(value) for value in buffer]
    out = []
    for raw in buffer:
        out.append(_bf16_to_float(int(raw)))
    return out


def _bf16_to_float(bits: int) -> float:
    """The fp32 value of one bfloat16 encoding."""
    word = (int(bits) & 0xFFFF) << 16
    return _struct.unpack("<f", _struct.pack("<I", word))[0]


def _float_to_bf16(value: float) -> int:
    """The round-to-nearest-even bfloat16 encoding of one fp32 value."""
    word = _struct.unpack("<I", _struct.pack("<f", float(value)))[0]
    if word & 0x7F800000 == 0x7F800000:
        return (word >> 16) & 0xFFFF
    word += 0x7FFF + ((word >> 16) & 1)
    return (word >> 16) & 0xFFFF


def store_compute_value(buffer, index: int, value: float,
                        element_size: int) -> None:
    """Writes one float into a compute buffer at ``index``."""
    if int(element_size) == 4:
        buffer[int(index)] = float(value)
        return
    buffer[int(index)] = _float_to_bf16(value)


def device_info(library) -> "Device":
    """The active device description."""
    info = Device()
    library.nanochat_device_info(ctypes.byref(info))
    return info


def device_count(library) -> int:
    """The number of local accelerators."""
    return int(library.nanochat_device_count())


def device_at(library, ordinal: int) -> "Device":
    """The local accelerator at ``ordinal``.

    Raises ``NanochatError`` when the ordinal is out of range.
    """
    info = Device()
    status = library.nanochat_device_at(int(ordinal), ctypes.byref(info))
    if status != NANOCHAT_STATUS_OK:
        raise NanochatError("nanochat_device_at", last_error(library))
    return info


def check_status(library, status: int, function: str) -> int:
    """Raise ``NanochatError`` when ``status`` is not zero."""
    if status != NANOCHAT_STATUS_OK:
        raise NanochatError(function, last_error(library))
    return status


def check_handle(library, handle, function: str, kind: str = "handle"):
    """Raise ``NanochatError`` when a ``create`` function returned null.

    Otherwise mint the :class:`Handle` that the callers pass around, so a
    handle is always a value this module produced and never a bare int, a path,
    or a pointer borrowed from another object.
    """
    if not handle:
        raise NanochatError(function, last_error(library))
    return Handle(handle, kind)


def check_code(library, code: int, function: str) -> int:
    """Raise ``NanochatError`` when a length or index is negative."""
    if code < 0:
        raise NanochatError(function, last_error(library))
    return code


def record_view(view: "Record") -> tuple[str, int, tuple[int, ...], bytes]:
    """Decode one ``Record`` into a plain Python tuple.

    The result is ``(name, dtype, shape, data)``. ``data`` is a copy, so it
    stays valid after the checkpoint handle is freed.
    """
    name = view.name.decode("utf-8", "replace") if view.name else ""
    shape = tuple(int(view.dims[axis]) for axis in range(int(view.ndim)))
    if view.bytes > 0:
        data = ctypes.string_at(view.data, int(view.bytes))
    else:
        data = b""
    return name, int(view.dtype), shape, data


def set_grad_enabled(library, enabled: bool) -> bool:
    """Toggle the model grad mode when the surface exposes the setter.

    ``nanochat_set_grad_enabled`` is not part of the current C application
    binary interface, so this is a no-op that reports ``False``. The call
    keeps ``no_grad`` ready for a later additive surface.
    """
    function = getattr(library, "nanochat_set_grad_enabled", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_int]
    function.restype = None
    function(1 if enabled else 0)
    return True
