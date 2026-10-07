"""The in-process Python API for nanochat.cpp.

The module follows docs/python.md section 6. It builds a thin, Pythonic
layer over the ctypes surface in ``nanochat_cpp._core``; every owned C handle
is released in ``__del__``.

The module imports only the Python standard library. It never imports
``torch``.
"""

from __future__ import annotations

import array
import ctypes
import dataclasses
import glob as _glob
import os
from pathlib import Path

from . import _build, _core, _lib
from ._core import NanochatError

__all__ = [
    "Config",
    "EvalReport",
    "Evaluator",
    "GeneratedRow",
    "Model",
    "NanochatError",
    "Optimizer",
    "ParamView",
    "ScoreReport",
    "TensorView",
    "TokenData",
    "Tokenizer",
    "Trainer",
    "build",
    "evaluate",
    "no_grad",
]

#: The padding multiple for the classifier width, from scripts/base_train.py.
PAD_VOCAB_TO = 64
#: The model aspect ratio, from scripts/base_train.py.
ASPECT_RATIO = 64
#: The reference head width, from scripts/base_train.py.
HEAD_DIM = 128


def _flatten_ints(values) -> list[int]:
    """A flat list of ints from a nested sequence or a ctypes array."""
    flat: list[int] = []

    def visit(value) -> None:
        if isinstance(value, (list, tuple, array.array, ctypes.Array)):
            for item in value:
                visit(item)
        else:
            flat.append(int(value))

    visit(values)
    return flat


def _flatten_floats(values) -> list[float]:
    """A flat list of floats from a nested sequence or a ctypes array."""
    flat: list[float] = []

    def visit(value) -> None:
        if isinstance(value, (list, tuple, array.array, ctypes.Array)):
            for item in value:
                visit(item)
        else:
            flat.append(float(value))

    visit(values)
    return flat


def _int_buffer(values):
    """A flat ``ctypes`` int array from any int sequence."""
    flat = _flatten_ints(values)
    return (ctypes.c_int * len(flat))(*flat)


def _expand_paths(patterns) -> list[str]:
    """Expand globs and keep explicit files in order."""
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    files: list[str] = []
    for pattern in patterns:
        text = str(pattern)
        matches = sorted(_glob.glob(text))
        if matches:
            files.extend(matches)
        elif Path(text).is_file():
            files.append(text)
    return files


def _void(library, name: str, *arguments) -> None:
    """Call a ``void`` entry point and raise when it reports a new error.

    The C surface returns no status from ``void`` functions, so the wrapper
    compares the thread-local error text before and after the call. A stale
    message from an earlier call does not raise.
    """
    function = getattr(library, name, None)
    if function is None:
        raise NanochatError(name, "the loaded library does not export " + name)
    before = _core.last_error(library)
    function(*arguments)
    after = _core.last_error(library)
    if after and after != before:
        raise NanochatError(name, after)


class TensorView:
    """A host-safe view over one model parameter buffer.

    The view copies each access through the C surface, so it is safe on the CPU
    backend and on the CUDA backend where the buffer lives in device memory.
    The view keeps the model alive so the buffer stays valid. The dtype is the
    build compute type; the Python view supports the four-byte float32 build.
    """

    def __init__(self, model,
                 index: int, grad: bool, count: int) -> None:
        self.count = int(count)
        self._model = model
        self._index = int(index)
        self._grad = 1 if grad else 0
        self._library = model._lib
        self._element_size = _core.compute_type_size(self._library)
        if self._element_size != ctypes.sizeof(ctypes.c_float):
            raise NanochatError(
                "TensorView",
                "the Python view supports a four-byte float32 build; this "
                f"build uses {self._element_size} bytes per element")

    def _buffer(self, count: int):
        return (ctypes.c_float * int(count))()

    def _read(self, offset: int, count: int) -> list[float]:
        count = int(count)
        if count <= 0:
            return []
        buffer = self._buffer(count)
        copied = self._library.nanochat_param_read(
            self._model._handle, self._index, self._grad, int(offset), count,
            ctypes.cast(buffer, ctypes.c_void_p))
        if copied != count:
            raise NanochatError("nanochat_param_read",
                                _core.last_error(self._library))
        return [float(value) for value in buffer]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> float:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return self._read(index, 1)[0]

    def __setitem__(self, index: int, value: float) -> None:
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        buffer = self._buffer(1)
        buffer[0] = float(value)
        copied = self._library.nanochat_param_write(
            self._model._handle, self._index, self._grad, index, 1,
            ctypes.cast(buffer, ctypes.c_void_p))
        if copied != 1:
            raise NanochatError("nanochat_param_write",
                                _core.last_error(self._library))

    def tolist(self) -> list[float]:
        return self._read(0, self.count)


def _tensor_view(model, index: int, grad: bool, count: int):
    if count <= 0:
        return None
    return TensorView(model, index, grad, count)


def _parse_device_index(device):
    """The device index in a Python device request, or None."""
    if device is None:
        return None
    if isinstance(device, int):
        return int(device)
    text = str(device).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:") and text[5:].isdigit():
        return int(text[5:])
    return None


@dataclasses.dataclass
class ParamView:
    """One model parameter, a Python mirror of ``nanochat::ParamView``."""

    name: str
    count: int
    rows: int
    cols: int
    value: TensorView | None
    grad: TensorView | None

    @property
    def is_matrix(self) -> bool:
        return self.rows > 0 and self.cols > 0


@dataclasses.dataclass
class GeneratedRow:
    """One generated sequence. ``mask`` is 1 for a sampled id, 0 for a prompt id."""

    tokens: list[int]
    mask: list[int]


@dataclasses.dataclass
class ScoreReport:
    """Per-row scoring results from ``nanochat_score_batch``."""

    nll: list[list[float]]
    argmax: list[list[int]]
    focus_logits: list[list[float]]


@dataclasses.dataclass
class EvalReport:
    """The result of ``evaluate``. ``core`` is None when no core set is given."""

    bpb: float
    core: float | None = None


class Config:
    """Model hyperparameters, a Python mirror of ``nanochat::Config``.

    ``depth`` sets the layer count and, together with ``aspect_ratio`` and
    ``head_dim``, derives ``hidden_dim`` and ``num_heads``. Pass an explicit
    ``hidden_dim``, ``num_heads``, or ``num_kv_heads`` to override the derived
    value.
    """

    def __init__(self, depth: int | None = None, *, num_layers: int | None = None,
                 num_heads: int | None = None, num_kv_heads: int | None = None,
                 hidden_dim: int | None = None, seq_len: int = 2048,
                 vocab_size: int = 32768, padded_vocab_size: int | None = None,
                 rope_base: float = 100000.0, window_pattern: str = "SSSL",
                 aspect_ratio: int = ASPECT_RATIO,
                 head_dim: int = HEAD_DIM) -> None:
        if num_layers is None:
            num_layers = int(depth) if depth is not None else 12
        if hidden_dim is None:
            base = int(num_layers) * int(aspect_ratio)
            hidden_dim = ((base + int(head_dim) - 1) // int(head_dim)) * int(head_dim)
        if num_heads is None:
            num_heads = max(1, int(hidden_dim) // int(head_dim))
        if num_kv_heads is None:
            num_kv_heads = int(num_heads)
        if padded_vocab_size is None:
            padded_vocab_size = (
                (int(vocab_size) + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO
            ) * PAD_VOCAB_TO

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.hidden_dim = int(hidden_dim)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)
        self.rope_base = float(rope_base)
        self.window_pattern = str(window_pattern)

    def _to_c(self) -> _core.Config:
        value = _core.Config()
        value.num_layers = self.num_layers
        value.num_heads = self.num_heads
        value.num_kv_heads = self.num_kv_heads
        value.hidden_dim = self.hidden_dim
        value.seq_len = self.seq_len
        value.vocab_size = self.vocab_size
        value.padded_vocab_size = self.padded_vocab_size
        value.rope_base = self.rope_base
        value.window_pattern = self.window_pattern.encode("utf-8")
        return value

    def __repr__(self) -> str:
        return (f"Config(num_layers={self.num_layers}, "
                f"num_heads={self.num_heads}, "
                f"num_kv_heads={self.num_kv_heads}, "
                f"hidden_dim={self.hidden_dim}, seq_len={self.seq_len}, "
                f"vocab_size={self.vocab_size}, "
                f"padded_vocab_size={self.padded_vocab_size}, "
                f"window_pattern={self.window_pattern!r})")


class Tokenizer:
    """The native ``NCTOKEN1`` tokenizer."""

    def __init__(self, handle, vocab_size: int | None = None,
                 special_tokens: dict | None = None) -> None:
        self._lib = _lib.load()
        self._handle = handle
        self.vocab_size = vocab_size
        self.special_tokens = dict(special_tokens) if special_tokens else {}

    @classmethod
    def load(cls, path) -> "Tokenizer":
        library = _lib.load()
        encoded = str(path).encode("utf-8")
        handle = _core.check_handle(
            library, library.nanochat_tokenizer_load(encoded),
            "nanochat_tokenizer_load")
        vocab_size = None
        special_tokens: dict[str, int] = {}
        try:
            from .data import BASE_VOCAB_SIZE, read_nctoken1
            artifact = read_nctoken1(path)
            vocab_size = (BASE_VOCAB_SIZE + len(artifact.merge_pairs)
                          + len(artifact.special_tokens))
            special_tokens = dict(artifact.special_tokens)
        except Exception:  # noqa: BLE001 - the metadata is optional
            vocab_size = None
        return cls(handle, vocab_size, special_tokens)

    def encode_special(self, name: str) -> int:
        """The id of a special token, or ``-1`` when the tokenizer lacks it."""
        return int(self.special_tokens.get(name, -1))

    def get_bos_token_id(self) -> int:
        """The id of ``<|bos|>``, or ``-1`` when the tokenizer lacks it."""
        return self.encode_special("<|bos|>")

    def encode(self, text):
        """Encode ``text``. A string gives a list of ids; a list gives a list of lists."""
        if isinstance(text, str):
            return self._encode_one(text)
        if isinstance(text, (list, tuple)):
            return [self._encode_one(item) for item in text]
        raise TypeError("encode expects a string or a list of strings")

    def _encode_one(self, text: str) -> list[int]:
        library = self._lib
        raw = text.encode("utf-8")
        needed = _core.check_code(
            library, library.nanochat_encode(self._handle, raw, None, 0),
            "nanochat_encode")
        if needed == 0:
            return []
        buffer = (ctypes.c_int * needed)()
        _core.check_code(
            library, library.nanochat_encode(self._handle, raw, buffer, needed),
            "nanochat_encode")
        return [int(value) for value in buffer]

    def decode(self, ids) -> str:
        """Decode a sequence of token ids into text."""
        flat = _flatten_ints(ids)
        library = self._lib
        if flat:
            buffer_ids = (ctypes.c_int * len(flat))(*flat)
        else:
            buffer_ids = None
        needed = _core.check_code(
            library,
            library.nanochat_decode(self._handle, buffer_ids, len(flat), None,
                                    0),
            "nanochat_decode")
        output = ctypes.create_string_buffer(needed + 1)
        _core.check_code(
            library,
            library.nanochat_decode(self._handle, buffer_ids, len(flat),
                                    ctypes.cast(output, ctypes.c_void_p),
                                    needed + 1),
            "nanochat_decode")
        return output.value.decode("utf-8", "replace")

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_lib", None)
        if handle and library is not None:
            try:
                library.nanochat_tokenizer_free(handle)
            except Exception:  # noqa: BLE001 - never raise in a destructor
                pass
            self._handle = None


class Model:
    """A nanochat model on the reference or the CUDA backend."""

    def __init__(self, config: Config, device=None,
                 seed: int | None = None) -> None:
        if not isinstance(config, Config):
            raise TypeError("config must be a nanochat_cpp.Config")
        self._lib = _lib.load(device=device)
        self.config = config
        if _core.backend_name(self._lib) == "cuda":
            info = _core.device_info(self._lib)
            requested = _parse_device_index(device)
            if requested is not None and requested != int(info.device_index):
                raise NanochatError(
                    "device",
                    f"device index {requested} is not selected; the CUDA "
                    f"library runs on device {info.device_index}, and device "
                    "selection is not yet in the C surface")
            self.device = f"cuda:{int(info.device_index)}"
        else:
            self.device = "cpu"
        self.seed = int(seed) if seed is not None else 0
        self._last_rows = 0
        self._c_config = config._to_c()
        handle = self._lib.nanochat_model_create(
            ctypes.byref(self._c_config), ctypes.c_uint64(self.seed))
        self._handle = _core.check_handle(self._lib, handle,
                                          "nanochat_model_create")

    def forward_loss(self, tokens, targets, batch: int | None = None,
                     seq: int | None = None) -> float:
        """The batch-mean cross-entropy loss in nats."""
        buffer_tokens = _int_buffer(tokens)
        buffer_targets = _int_buffer(targets)
        count = len(buffer_tokens)
        if seq is None:
            seq = self.config.seq_len
        seq = int(seq)
        if batch is None:
            batch = count // seq if seq else 0
        batch = int(batch)
        if batch <= 0 or seq <= 0 or count != batch * seq:
            raise ValueError(
                f"{count} tokens do not fit batch {batch} x seq {seq}")
        self._last_rows = batch * seq
        return float(self._lib.nanochat_forward_loss(
            self._handle, buffer_tokens, buffer_targets, batch, seq))

    def backward(self) -> None:
        _void(self._lib, "nanochat_backward", self._handle)

    def backward_weighted(self, row_weights, scale: float = 1.0) -> None:
        """Accumulate a weighted backward pass over the most recent forward.

        ``row_weights`` holds one weight per row, that is ``batch * seq``
        entries for the most recent ``forward_loss``. The call accumulates
        into the parameter gradients, so call ``zero_grad`` first
        (docs/post-training.md section 2.2).
        """
        weights = _flatten_floats(row_weights)
        if self._last_rows <= 0:
            raise RuntimeError(
                "forward_loss must run before backward_weighted")
        if len(weights) != self._last_rows:
            raise ValueError(
                f"row_weights has {len(weights)} entries; "
                f"expected {self._last_rows}")
        buffer = (ctypes.c_float * len(weights))(*weights)
        _void(self._lib, "nanochat_backward_weighted", self._handle, buffer,
              ctypes.c_float(float(scale)))

    def zero_grad(self) -> None:
        _void(self._lib, "nanochat_zero_grad", self._handle)

    def param_count(self) -> int:
        return int(self._lib.nanochat_param_count(self._handle))

    def params(self) -> list[ParamView]:
        """The parameter views in the optimizer's grouping order."""
        library = self._lib
        count = self.param_count()
        views: list[ParamView] = []
        for index in range(count):
            param = _core.Param()
            _core.check_code(
                library,
                library.nanochat_param_info(self._handle, index,
                                            ctypes.byref(param)),
                "nanochat_param_info")
            name = param.name.decode("utf-8", "replace") if param.name else ""
            views.append(ParamView(
                name=name,
                count=int(param.count),
                rows=int(param.rows),
                cols=int(param.cols),
                value=_tensor_view(self, index, False, param.count)
                if param.value else None,
                grad=_tensor_view(self, index, True, param.count)
                if param.grad else None,
            ))
        return views

    def save(self, path) -> None:
        _void(self._lib, "nanochat_save", self._handle,
              str(path).encode("utf-8"))

    def load(self, path) -> None:
        _void(self._lib, "nanochat_load", self._handle,
              str(path).encode("utf-8"))

    def generate(self, prompt, max_tokens: int = 256, temperature: float = 1.0,
                 top_k: int = 0, num_samples: int = 1,
                 seed: int = 42, stop_id: int = -1, bos_id: int = -1,
                 stop_ids=None) -> list[GeneratedRow]:
        """Generate ``num_samples`` rows from ``prompt``."""
        library = self._lib
        flat = _flatten_ints(prompt)
        prompt_buffer = _int_buffer(flat)
        params = _core.GenerateParams()
        params.num_samples = int(num_samples)
        params.max_tokens = int(max_tokens)
        params.temperature = float(temperature)
        params.top_k = int(top_k)
        params.seed = int(seed)
        params.stop_id = int(stop_id)
        params.bos_id = int(bos_id)
        keep_alive = None
        if stop_ids is not None:
            keep_alive = (ctypes.c_int * len(stop_ids))(*stop_ids)
            params.stop_ids = ctypes.cast(
                keep_alive, ctypes.POINTER(ctypes.c_int))
        out = _core.Sequences()
        _void(library, "nanochat_generate", self._handle, prompt_buffer,
              len(flat), ctypes.byref(params), ctypes.byref(out))
        rows: list[GeneratedRow] = []
        for index in range(int(out.count)):
            length = int(out.lengths[index])
            offset = int(out.offsets[index])
            tokens = [int(out.tokens[offset + i]) for i in range(length)]
            mask = [int(out.mask[offset + i]) for i in range(length)]
            rows.append(GeneratedRow(tokens=tokens, mask=mask))
        return rows

    def no_grad(self):
        """A context manager that suspends gradient tracking."""
        return no_grad(self)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_lib", None)
        if handle and library is not None:
            try:
                library.nanochat_model_free(handle)
            except Exception:  # noqa: BLE001 - never raise in a destructor
                pass
            self._handle = None


class Optimizer:
    """The combined AdamW and Muon optimizer over one model."""

    def __init__(self, model: Model, *, unembedding_lr: float = 0.004,
                 embedding_lr: float = 0.2, matrix_lr: float = 0.02,
                 scalar_lr: float = 0.5, weight_decay: float = 0.0,
                 muon_ns_steps: int = 5, muon_beta2: float = 0.9,
                 adam_eps: float = 1e-10, clip: float = 1.0,
                 num_iterations: int = 0, warmup_steps: int = 0,
                 warmdown_ratio: float = 0.65, final_lr_frac: float = 0.0,
                 weight_decay_base: float = 0.0,
                 muon_momentum_warmup_steps: float = 400.0,
                 muon_momentum_start: float = 0.85,
                 muon_momentum_peak: float = 0.97,
                 muon_momentum_final: float = 0.90,
                 matrix_optimizer: int = 0, adam_step_period: int = 1,
                 anvil_lr: float = 0.023, anvil_weight_decay: float = 2.25,
                 anvil_momentum: float = 0.95, anvil_beta2: float = 0.9,
                 anvil_fast_beta: float = 0.85,
                 anvil_slow_beta: float = 0.98,
                 anvil_fast_weight: float = 0.4385,
                 anvil_engage_step: int = 514, anvil_num_maps: int = 6,
                 rail_beta_warmup_steps: float = 240.0,
                 rail_beta_cooldown_steps: float = 50.0,
                 rail_beta_min: float = 0.85,
                 rail_beta_max: float = 0.93) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a nanochat_cpp.Model")
        self._lib = _lib.load()
        self.model = model
        self.num_iterations = int(num_iterations)
        self.warmup_steps = int(warmup_steps)
        config = _core.OptimizerConfig()
        config.unembedding_lr = float(unembedding_lr)
        config.embedding_lr = float(embedding_lr)
        config.matrix_lr = float(matrix_lr)
        config.scalar_lr = float(scalar_lr)
        config.weight_decay = float(weight_decay)
        config.muon_ns_steps = int(muon_ns_steps)
        config.muon_beta2 = float(muon_beta2)
        config.adam_eps = float(adam_eps)
        config.clip = float(clip)
        config.num_iterations = int(num_iterations)
        config.warmup_steps = int(warmup_steps)
        config.warmdown_ratio = float(warmdown_ratio)
        config.final_lr_frac = float(final_lr_frac)
        config.weight_decay_base = float(weight_decay_base)
        config.muon_momentum_warmup_steps = float(muon_momentum_warmup_steps)
        config.muon_momentum_start = float(muon_momentum_start)
        config.muon_momentum_peak = float(muon_momentum_peak)
        config.muon_momentum_final = float(muon_momentum_final)
        config.matrix_optimizer = int(matrix_optimizer)
        config.adam_step_period = int(adam_step_period)
        config.anvil_lr = float(anvil_lr)
        config.anvil_weight_decay = float(anvil_weight_decay)
        config.anvil_momentum = float(anvil_momentum)
        config.anvil_beta2 = float(anvil_beta2)
        config.anvil_fast_beta = float(anvil_fast_beta)
        config.anvil_slow_beta = float(anvil_slow_beta)
        config.anvil_fast_weight = float(anvil_fast_weight)
        config.anvil_engage_step = int(anvil_engage_step)
        config.anvil_num_maps = int(anvil_num_maps)
        config.rail_beta_warmup_steps = float(rail_beta_warmup_steps)
        config.rail_beta_cooldown_steps = float(rail_beta_cooldown_steps)
        config.rail_beta_min = float(rail_beta_min)
        config.rail_beta_max = float(rail_beta_max)
        self.config = config
        handle = self._lib.nanochat_optim_create(
            model._handle, ctypes.byref(config))
        self._handle = _core.check_handle(self._lib, handle,
                                          "nanochat_optim_create")

    def step(self, step: int) -> None:
        """Apply one 1-based optimizer step."""
        _void(self._lib, "nanochat_optim_step", self._handle, int(step))

    def grad_norm(self) -> float:
        """The global gradient norm from the most recent step."""
        return float(self._lib.nanochat_optim_grad_norm(self._handle))

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_lib", None)
        if handle and library is not None:
            try:
                library.nanochat_optim_free(handle)
            except Exception:  # noqa: BLE001 - never raise in a destructor
                pass
            self._handle = None


class TokenData:
    """A parquet document loader that packs token rows on demand."""

    def __init__(self, parquet, tokenizer: Tokenizer, seq_len: int,
                 batch: int = 1, seed: int = 42, text_column: str = "text",
                 threads: int = 1, document_buffer: int = 1000) -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must be a nanochat_cpp.Tokenizer")
        self._lib = _lib.load()
        self.tokenizer = tokenizer
        self._parquet = parquet
        self._text_column = text_column
        self._seed = int(seed)
        self._threads = int(threads)
        self._document_buffer = int(document_buffer)
        self._seq = int(seq_len)
        self._batch = int(batch)
        files = _expand_paths(parquet)
        if not files:
            raise ValueError(f"no parquet files match {parquet!r}")
        self._files = files
        self._c_files = (ctypes.c_char_p * len(files))(
            *[name.encode("utf-8") for name in files])
        self._handle = self._open()

    def _open(self):
        """Open one owned C loader handle from the stored inputs."""
        handle = self._lib.nanochat_loader_create(
            self._c_files, len(self._files),
            self._text_column.encode("utf-8"), self.tokenizer._handle,
            self._batch, self._seq, ctypes.c_uint64(self._seed),
            self._threads, self._document_buffer)
        return _core.check_handle(self._lib, handle,
                                  "nanochat_loader_create")

    def reset(self) -> None:
        """Restart the document stream from the first batch.

        The C application binary interface has no reset entry point. The
        wrapper releases the loader and opens a fresh one instead. The loader
        is deterministic, so a fresh loader yields the same batches as the C++
        ``DataLoader::Reset``.
        """
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.nanochat_loader_free(handle)
            self._handle = None
        self._handle = self._open()

    def rebatch(self, batch: int) -> "TokenData":
        """A fresh loader with the same inputs and a different batch size."""
        return TokenData(parquet=self._parquet, tokenizer=self.tokenizer,
                         seq_len=self._seq, batch=batch, seed=self._seed,
                         text_column=self._text_column, threads=self._threads,
                         document_buffer=self._document_buffer)

    def next(self):
        """The next ``(tokens, targets)`` batch, or None at the end."""
        size = self._batch * self._seq
        tokens = (ctypes.c_int * size)()
        targets = (ctypes.c_int * size)()
        code = _core.check_code(
            self._lib,
            self._lib.nanochat_loader_next(self._handle, tokens, targets),
            "nanochat_loader_next")
        if code == 0:
            return None
        return array.array("i", tokens), array.array("i", targets)

    def token_bytes(self) -> list[int]:
        """The per-token byte lengths, indexed by token id."""
        vocab = ctypes.c_int()
        pointer = self._lib.nanochat_loader_token_bytes(self._handle,
                                                        ctypes.byref(vocab))
        if not pointer:
            return []
        return [int(pointer[index]) for index in range(vocab.value)]

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_lib", None)
        if handle and library is not None:
            try:
                library.nanochat_loader_free(handle)
            except Exception:  # noqa: BLE001 - never raise in a destructor
                pass
            self._handle = None


class Trainer:
    """An iterable training loop. Each iteration yields ``(step, loss)``."""

    def __init__(self, model: Model, data: TokenData, batch: int | None = None,
                 total_batch: int | None = None, warmup_steps: int | None = None,
                 num_iterations: int | None = None, optimizer: Optimizer | None = None,
                 grad_accum: int | None = None, seed: int | None = None,
                 optimizer_config: dict | None = None,
                 scheduler_config: dict | None = None) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a nanochat_cpp.Model")
        if not isinstance(data, TokenData):
            raise TypeError("data must be a nanochat_cpp.TokenData")
        self._lib = _lib.load()
        self.model = model
        if batch is not None and int(batch) != data._batch:
            data = data.rebatch(int(batch))
        self.data = data
        self.batch = data._batch
        self.seq = data._seq
        if grad_accum is None:
            if total_batch is not None and self.batch * self.seq > 0:
                grad_accum = max(1, int(total_batch) // (self.batch * self.seq))
            else:
                grad_accum = 1
        self.grad_accum = max(1, int(grad_accum))
        if num_iterations is None:
            num_iterations = getattr(optimizer, "num_iterations", 0) or 1
        self.num_iterations = int(num_iterations)
        self._step = 0
        if optimizer is None:
            options = dict(optimizer_config or {})
            schedule = dict(scheduler_config or {})
            if warmup_steps is not None:
                schedule["warmup_steps"] = int(warmup_steps)
            schedule.setdefault("num_iterations", self.num_iterations)
            options.update(schedule)
            optimizer = Optimizer(model, **options)
        self.optimizer = optimizer

    def __iter__(self) -> "Trainer":
        return self

    def backward_weighted(self, row_weights, scale: float = 1.0) -> None:
        """Accumulate a weighted backward pass over one micro-batch.

        ``row_weights`` holds one weight per row, that is ``batch * seq``
        entries. The call forwards to ``nanochat_backward_weighted`` and
        accumulates into the parameter gradients (docs/post-training.md
        section 2.2).
        """
        weights = _flatten_floats(row_weights)
        expected = self.batch * self.seq
        if len(weights) != expected:
            raise ValueError(
                f"row_weights has {len(weights)} entries; expected {expected}")
        self.model.backward_weighted(weights, scale)

    def __next__(self):
        if self._step >= self.num_iterations:
            raise StopIteration
        step = self._step + 1
        # Gradient accumulation: sum `grad_accum` micro-batch gradients (each
        # scaled by 1/grad_accum) before one optimizer step, exactly as the
        # C++ TrainLoop does. The C surface has no uniform-scale backward, so a
        # micro-batch uses the all-ones row weight that has the same effect.
        accum = self.grad_accum if self.grad_accum > 0 else 1
        inv_accum = 1.0 / accum
        self.model.zero_grad()
        loss_sum = 0.0
        for _ in range(accum):
            batch = self.data.next()
            if batch is None:
                # Cycle the epoch the same way the C++ TrainLoop does: reset
                # the document stream once and read the first batch again. The
                # loader provides fewer batches than the fixture has steps.
                self.data.reset()
                batch = self.data.next()
            if batch is None:
                raise RuntimeError(
                    "the document loader is exhausted after a reset")
            tokens, targets = batch
            loss_sum += self.model.forward_loss(tokens, targets,
                                                batch=self.batch, seq=self.seq)
            if accum == 1:
                self.model.backward()
            else:
                self.backward_weighted([1.0] * (self.batch * self.seq),
                                       scale=inv_accum)
        self.optimizer.step(step)
        self._step = step
        return step, loss_sum * inv_accum

    def save(self, path) -> None:
        self.model.save(path)

    def __len__(self) -> int:
        return self.num_iterations


class Evaluator:
    """Forward-only scoring, bits-per-byte, and generation over one model."""

    def __init__(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a nanochat_cpp.Model")
        self._lib = _lib.load()
        self.model = model

    def bpb(self, data: TokenData, steps: int) -> float:
        """The bits-per-byte over ``steps`` batches from ``data``."""
        if not isinstance(data, TokenData):
            raise TypeError("data must be a nanochat_cpp.TokenData")
        return float(self._lib.nanochat_eval_bpb(
            self.model._handle, data._handle, int(steps)))

    def score(self, tokens, lengths=None, focus=None) -> ScoreReport:
        """Score a batch of padded sequences.

        ``focus`` is an optional per-row ``(position, ids)`` pair or None.
        """
        flat = _flatten_ints(tokens)
        seq = self.model.config.seq_len
        batch = len(flat) // seq if seq else 0
        if batch <= 0 or len(flat) != batch * seq:
            raise ValueError("tokens do not fill whole rows")
        token_buffer = _int_buffer(flat)
        if lengths is not None:
            length_values = [int(value) for value in lengths]
            if len(length_values) != batch:
                raise ValueError("lengths must have one entry per row")
            length_buffer = (ctypes.c_int * batch)(*length_values)
        else:
            length_buffer = None

        focus_buffer = None
        keep_alive: list = []
        if focus is not None:
            requests = list(focus)
            if len(requests) != batch:
                raise ValueError("focus must have one entry per row")
            focus_buffer = (_core.Focus * batch)()
            for index, request in enumerate(requests):
                if request is None:
                    continue
                position, ids = request
                ids = [int(value) for value in ids]
                if not ids:
                    continue
                ids_buffer = (ctypes.c_int * len(ids))(*ids)
                keep_alive.append(ids_buffer)
                focus_buffer[index].position = int(position)
                focus_buffer[index].ids = ctypes.cast(
                    ids_buffer, ctypes.POINTER(ctypes.c_int))
                focus_buffer[index].count = len(ids)

        results = (_core.ScoreResult * batch)()
        _void(self._lib, "nanochat_score_batch", self.model._handle,
              token_buffer, batch, seq, length_buffer, focus_buffer, results)

        nll: list[list[float]] = []
        argmax: list[list[int]] = []
        focus_logits: list[list[float]] = []
        for index in range(batch):
            nll.append([float(results[index].nll[p]) for p in range(seq)])
            argmax.append([int(results[index].argmax[p]) for p in range(seq)])
            if focus_buffer is None or focus_buffer[index].count <= 0:
                focus_logits.append([])
                continue
            count = int(focus_buffer[index].count)
            focus_logits.append(
                [float(results[index].focus_logits[k]) for k in range(count)])
        return ScoreReport(nll=nll, argmax=argmax, focus_logits=focus_logits)

    def generate(self, prompt, **kwargs) -> list[GeneratedRow]:
        return self.model.generate(prompt, **kwargs)


class _NoGrad:
    """The context manager returned by ``no_grad``."""

    def __init__(self, model: Model | None = None) -> None:
        self._lib = model._lib if isinstance(model, Model) else None

    def __enter__(self) -> "_NoGrad":
        if self._lib is not None:
            _core.set_grad_enabled(self._lib, False)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._lib is not None:
            _core.set_grad_enabled(self._lib, True)
        return False


def no_grad(model: Model | None = None) -> _NoGrad:
    """A context manager that suspends gradient tracking (docs/python.md).

    The current C surface exposes no grad-mode setter, so the manager is a
    no-op until ``nanochat_set_grad_enabled`` is added. The entry point keeps
    the notebook code unchanged.
    """
    return _NoGrad(model)


def evaluate(model: Model, parquet, tokenizer: Tokenizer, tokens: int | None = None,
             steps: int | None = None, batch: int = 8, seed: int = 42,
             threads: int = 4, document_buffer: int = 1000,
             text_column: str = "text") -> EvalReport:
    """Evaluate ``model`` on a parquet split and return the bits-per-byte.

    ``tokens`` gives the split size in tokens; the wrapper converts it to a
    batch count, as ``scripts/base_eval.py`` does. ``steps`` overrides it.
    """
    if steps is None:
        if tokens is not None and batch * model.config.seq_len > 0:
            steps = max(1, int(tokens) // (batch * model.config.seq_len))
        else:
            steps = 1
    data = TokenData(parquet=parquet, tokenizer=tokenizer,
                     seq_len=model.config.seq_len, batch=batch, seed=seed + 1,
                     text_column=text_column, threads=threads,
                     document_buffer=document_buffer)
    bpb = Evaluator(model).bpb(data, steps)
    return EvalReport(bpb=bpb, core=None)


def build() -> Path:
    """Build the shared library on demand and return its path."""
    return _build.ensure_library()
