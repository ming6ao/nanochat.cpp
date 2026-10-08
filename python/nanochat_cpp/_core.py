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
    "ModelHandle",
    "OptimHandle",
    "LoaderHandle",
    "TokenizerHandle",
    "declare",
    "backend_name",
    "compute_type_size",
    "device_info",
    "last_error",
    "check_status",
    "check_handle",
    "check_code",
    "set_grad_enabled",
]

#: A ``nanochat_status`` value that means success.
NANOCHAT_STATUS_OK = 0
#: A ``nanochat_status`` value that means failure.
NANOCHAT_STATUS_ERROR = 1

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
        ("matrix_optimizer", ctypes.c_int),
        ("adam_step_period", ctypes.c_int),
        ("anvil_lr", ctypes.c_float),
        ("anvil_weight_decay", ctypes.c_float),
        ("anvil_momentum", ctypes.c_float),
        ("anvil_beta2", ctypes.c_float),
        ("anvil_fast_beta", ctypes.c_float),
        ("anvil_slow_beta", ctypes.c_float),
        ("anvil_fast_weight", ctypes.c_float),
        ("anvil_engage_step", ctypes.c_int),
        ("anvil_num_maps", ctypes.c_int),
        ("rail_beta_warmup_steps", ctypes.c_float),
        ("rail_beta_cooldown_steps", ctypes.c_float),
        ("rail_beta_min", ctypes.c_float),
        ("rail_beta_max", ctypes.c_float),
    ]


# Opaque handles. The C declarations use ``struct nanochat_model *`` and
# friends; a ``c_void_p`` carries the same pointer width.
ModelHandle = ctypes.c_void_p
OptimHandle = ctypes.c_void_p
LoaderHandle = ctypes.c_void_p
TokenizerHandle = ctypes.c_void_p

# The function table: name, argument types, result type. Every entry in
# include/nanochat/capi.h is here.
_SIGNATURES = (
    ("nanochat_init", (), ctypes.c_int),
    ("nanochat_last_error", (), ctypes.c_char_p),
    ("nanochat_version", (), ctypes.c_char_p),
    ("nanochat_backend", (), ctypes.c_char_p),
    ("nanochat_compute_type_size", (), ctypes.c_int),
    ("nanochat_device_info", (ctypes.POINTER(Device),), None),
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
    """The parameter element size in bytes: four for fp32, two for fp16."""
    return int(library.nanochat_compute_type_size())


def device_info(library) -> "Device":
    """The active device description."""
    info = Device()
    library.nanochat_device_info(ctypes.byref(info))
    return info


def check_status(library, status: int, function: str) -> int:
    """Raise ``NanochatError`` when ``status`` is not zero."""
    if status != NANOCHAT_STATUS_OK:
        raise NanochatError(function, last_error(library))
    return status


def check_handle(library, handle, function: str):
    """Raise ``NanochatError`` when a ``create`` function returned null."""
    if not handle:
        raise NanochatError(function, last_error(library))
    return handle


def check_code(library, code: int, function: str) -> int:
    """Raise ``NanochatError`` when a length or index is negative."""
    if code < 0:
        raise NanochatError(function, last_error(library))
    return code


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
