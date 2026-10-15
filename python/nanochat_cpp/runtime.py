"""The single-client multi-device runtime client.

One :class:`MeshRuntime` owns one model replica per device and runs one training
step over the whole mesh. The client supplies one global batch; the runtime
splits it into contiguous slices and reduces the gradients in process, so every
replica applies the same update. See ``docs/client-runtime-design.md``.

This is the in-process runtime A (one process, one device thread per device).
The compatibility process runtime is a later phase. The module imports the
standard library and the shared-library loader only.
"""

from __future__ import annotations

import array
import copy
import ctypes
import dataclasses
import json
import os
from pathlib import Path

from . import _core, _lib
from .mesh import Mesh

__all__ = ["MeshRuntime", "MeshTrainConfig", "MeshTrainResult",
           "StepMetrics", "global_batch"]


def _as_core_config(value, expected, label):
    """The low-level ctypes value for `value`, through `_to_c()` if needed."""
    if isinstance(value, expected):
        return value
    converter = getattr(value, "_to_c", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, expected):
            return converted
    raise TypeError(
        f"{label} must be a nanochat_cpp._core.{expected.__name__} or an "
        "object with _to_c()"
    )


def _core_model_config(config):
    """The ``_core.Config`` for ``config`` and the live window-pattern bytes.

    ``Config._to_c`` sets ``window_pattern`` from temporary bytes, so the
    caller keeps a copy alive across the later C call.
    """
    core = _as_core_config(config, _core.Config, "config")
    pattern = bytes(core.window_pattern) if core.window_pattern else b""
    core.window_pattern = pattern
    return core, pattern


def _model_to_dict(core) -> dict:
    """The serializable model shape from a ``_core.Config``."""
    return {
        "num_layers": int(core.num_layers),
        "num_heads": int(core.num_heads),
        "num_kv_heads": int(core.num_kv_heads),
        "hidden_dim": int(core.hidden_dim),
        "seq_len": int(core.seq_len),
        "vocab_size": int(core.vocab_size),
        "padded_vocab_size": int(core.padded_vocab_size),
        "rope_base": float(core.rope_base),
        "window_pattern": (core.window_pattern or b"").decode("utf-8"),
        "value_embedding": bool(core.value_embedding),
    }


def _model_from_dict(model: dict):
    """The ``Config`` for a job dictionary."""
    from .api import Config

    return Config(**model)


def global_batch(per_rank_batch: int, grad_accum: int,
                 world_size: int) -> int:
    """The global sequence count of one step.

    The driver takes the global batch, so this function rebuilds the total:
    ``per_rank_batch * grad_accum * world_size``.
    """
    per_rank = int(per_rank_batch)
    accum = int(grad_accum)
    world = int(world_size)
    if per_rank < 1 or accum < 1 or world < 1:
        raise ValueError("per_rank_batch, grad_accum, and world_size must be "
                         "positive")
    return per_rank * accum * world


def _int_buffer(values):
    """A flat ``ctypes`` int array from a batch of int32 ids.

    A batch may arrive as a list, a tuple, an ``array.array``, a ``ctypes``
    array, or a raw little-endian ``bytes`` buffer. The token loader returns
    ``array.array``. A raw buffer is reinterpreted as int32, not iterated,
    because iterating a ``bytes`` object yields single bytes. Without the
    buffer arm, ``int()`` receives the whole buffer and parses its raw bytes,
    which raises ``ValueError``.
    """
    if isinstance(values, (bytes, bytearray)):
        values = array.array("i", values)
    elif isinstance(values, memoryview) and values.itemsize == 1:
        values = values.cast("i")
    flat: list[int] = []

    def visit(value) -> None:
        if isinstance(value, (list, tuple, array.array, ctypes.Array,
                              memoryview)):
            for item in value:
                visit(item)
        else:
            flat.append(int(value))

    visit(values)
    return (ctypes.c_int * len(flat))(*flat)


@dataclasses.dataclass(frozen=True)
class StepMetrics:
    """The report of the last training step.

    ``loss`` holds one entry per replica. ``grad_norm`` is the reduced gradient
    norm, so every replica reports the same value. ``step_seconds`` is the
    wall-clock time of the last step.
    """

    loss: tuple[float, ...]
    grad_norm: float
    step_seconds: float

    @property
    def mean_loss(self) -> float:
        """The mean loss over the replicas, or zero when the run has none."""
        return sum(self.loss) / len(self.loss) if self.loss else 0.0


class MeshRuntime:
    """A training runtime over a device mesh.

    The runtime wraps the C++ ``MeshRuntime`` behind the C application binary
    interface. It is CPU-tested on the reference backend; the CUDA
    thread-local path is a later phase.
    """

    def __init__(self, mesh: Mesh, config, optimizer_config=None, *,
                 batch: int, grad_accum: int = 1, seq: int = 0,
                 seed: int = 42) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a Mesh")
        core_config = _as_core_config(config, _core.Config, "config")
        core_optimizer = (
            None
            if optimizer_config is None
            else _as_core_config(optimizer_config, _core.OptimizerConfig,
                                 "optimizer_config")
        )
        library = _lib.load()
        self._lib = library
        self._mesh = mesh
        indices = [int(device.index) for device in mesh.devices]
        indices_buffer = (ctypes.c_int * len(indices))(*indices)
        self._mesh_handle = library.nanochat_mesh_create(indices_buffer,
                                                         len(indices))
        _core.check_handle(library, self._mesh_handle,
                           "nanochat_mesh_create")
        optim_pointer = (
            ctypes.byref(core_optimizer)
            if core_optimizer is not None
            else None
        )
        self._handle = library.nanochat_runtime_create(
            self._mesh_handle,
            ctypes.byref(core_config),
            optim_pointer,
            int(batch),
            int(grad_accum),
            int(seq),
            int(seed),
        )
        _core.check_handle(library, self._handle, "nanochat_runtime_create")
        self._world_size = int(
            library.nanochat_runtime_world_size(self._handle))

    @property
    def mesh(self) -> Mesh:
        """The device mesh."""
        return self._mesh

    @property
    def world_size(self) -> int:
        """The number of replicas, one per device."""
        return self._world_size

    @property
    def step(self) -> int:
        """The 1-based step of the last step, or the loaded step."""
        return int(self._lib.nanochat_runtime_step(self._handle))

    def train_step(self, tokens, targets) -> float:
        """Runs one step over one global batch. Returns the global mean loss.

        ``tokens`` and ``targets`` hold ``batch * seq`` row-major ids. The
        runtime splits the batch across the mesh, so the caller never names a
        rank.
        """
        token_buffer = _int_buffer(tokens)
        target_buffer = _int_buffer(targets)
        if len(token_buffer) != len(target_buffer):
            raise ValueError("tokens and targets must have the same length")
        before = _core.last_error(self._lib)
        loss = self._lib.nanochat_runtime_train_step(
            self._handle, token_buffer, target_buffer)
        after = _core.last_error(self._lib)
        if after and after != before:
            raise _core.NanochatError("nanochat_runtime_train_step", after)
        return float(loss)

    def metrics(self) -> StepMetrics:
        """The metrics of the last step, one entry per replica.

        Call after :meth:`train_step`. The per-replica loss buffer belongs to
        the runtime, so the wrapper copies it into the result.
        """
        info = _core.Metrics()
        status = self._lib.nanochat_runtime_metrics(self._handle,
                                                    ctypes.byref(info))
        _core.check_status(self._lib, status, "nanochat_runtime_metrics")
        count = int(info.device_count)
        loss = (tuple(float(info.loss[i]) for i in range(count))
                if info.loss and count > 0 else ())
        return StepMetrics(loss=loss, grad_norm=float(info.grad_norm),
                           step_seconds=float(info.step_seconds))

    def replica_digest(self, replica: int = 0) -> int:
        """A 64-bit digest of every parameter of ``replica``.

        The replicas of one runtime return the same digest after a step, so a
        client can check the gradient reduce without copying the whole model
        to the host.
        """
        out = ctypes.c_uint64()
        status = self._lib.nanochat_runtime_param_digest(
            self._handle, int(replica), ctypes.byref(out))
        _core.check_status(self._lib, status,
                           "nanochat_runtime_param_digest")
        return int(out.value)

    def param_count(self, replica: int = 0) -> int:
        """The parameter count of ``replica``."""
        return int(
            self._lib.nanochat_runtime_param_count(self._handle,
                                                   int(replica)))

    def param_name(self, index: int, replica: int = 0) -> str:
        """The name of parameter ``index`` on ``replica``."""
        info = self._param_info(index, replica)
        name = info.name
        return name.decode("utf-8", "replace") if name else ""

    def param_read(self, index: int, replica: int = 0, grad: bool = False):
        """The elements of parameter ``index`` on ``replica``.

        The read copies through the kernel seam, so it is safe on a device
        backend. The Python view supports a four-byte float32 build.
        """
        info = self._param_info(index, replica)
        count = int(info.count)
        if count <= 0:
            return []
        if _core.compute_type_size(self._lib) != 4:
            raise _core.NanochatError(
                "param_read",
                "the Python view supports a four-byte float32 build",
            )
        buffer = (ctypes.c_float * count)()
        copied = self._lib.nanochat_runtime_param_read(
            self._handle,
            int(replica),
            int(index),
            1 if grad else 0,
            0,
            count,
            ctypes.cast(buffer, ctypes.c_void_p),
        )
        if copied != count:
            raise _core.NanochatError("nanochat_runtime_param_read",
                                      _core.last_error(self._lib))
        return list(buffer)

    def save(self, path) -> None:
        """Write the canonical checkpoint from replica 0.

        The file carries the optimizer state and the step beside the
        parameters, so :meth:`load` resumes the schedule.
        """
        self._void("nanochat_runtime_save", str(path).encode("utf-8"))

    def load(self, path, optimizer: bool = True) -> None:
        """Load one checkpoint into every replica.

        With ``optimizer=True`` (the default) the load reads the optimizer
        state and the step, so :attr:`step` reports where to resume. With
        ``optimizer=False`` the load reads the parameter records only, and the
        optimizer state and the step stay unchanged. A fine-tuning warm start
        uses ``optimizer=False``, so it does not inherit the pretraining
        moments.
        """
        if optimizer:
            self._void("nanochat_runtime_load", str(path).encode("utf-8"))
        else:
            self._void("nanochat_runtime_load_weights",
                       str(path).encode("utf-8"))

    def _void(self, name, *arguments) -> None:
        """Calls a ``void`` entry point and raises on a new error."""
        before = _core.last_error(self._lib)
        getattr(self._lib, name)(self._handle, *arguments)
        after = _core.last_error(self._lib)
        if after and after != before:
            raise _core.NanochatError(name, after)

    def close(self) -> None:
        """Releases the runtime and the mesh handles. Safe to call twice."""
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.nanochat_runtime_free(handle)
            self._handle = None
        mesh_handle = getattr(self, "_mesh_handle", None)
        if mesh_handle:
            self._lib.nanochat_mesh_free(mesh_handle)
            self._mesh_handle = None

    def __enter__(self) -> "MeshRuntime":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _param_info(self, index: int, replica: int) -> "_core.Param":
        count = self.param_count(replica)
        if index < 0 or index >= count:
            raise IndexError(index)
        info = _core.Param()
        status = self._lib.nanochat_runtime_param_info(
            self._handle, int(replica), int(index), ctypes.byref(info))
        if status != _core.NANOCHAT_STATUS_OK:
            raise _core.NanochatError("nanochat_runtime_param_info",
                                      _core.last_error(self._lib))
        return info


@dataclasses.dataclass(frozen=True)
class MeshTrainResult:
    """The outcome of one mesh driver run
    (docs/client-runtime-design.md section 10.1).

    ``interrupted`` is true when the run stopped before the last step, for
    example after Control-C. ``checkpoint`` is the final checkpoint path, or
    the empty string when the run wrote none.
    """

    loss: float
    steps: int
    interrupted: bool
    mean_loss: float
    steady_mfu: float
    checkpoint: str


class MeshTrainConfig:
    """The C++ mesh training driver config, the coarse client
    (docs/client-runtime-design.md section 10.1 and docs/training-seam.md
    section 6.6).

    The driver owns the step loop, the metric record, the summary, the
    checkpoint interval, and the resume. The client holds the job as a plain
    dictionary, builds one C config from it, and calls the C run. Python owns
    the job file for the child process, so the C interface needs no serializer.
    """

    def __init__(self, config, preset="track3", *, world_size=1,
                 per_rank_batch, grad_accum, steps, tokenizer, train_parquet,
                 seq=0, seed=42, data_seed=42, log_every=1, save_every=0,
                 resume=False, out_dir="", checkpoint_path="",
                 flops_per_token=0.0, peak_flops=0.0) -> None:
        self._lib = _lib.load()
        self._keepalive: list = []
        self._mesh = None
        self._config = None
        core_model = _as_core_config(config, _core.Config, "config")
        paths = ([train_parquet]
                 if isinstance(train_parquet, (str, bytes, os.PathLike))
                 else list(train_parquet))
        count = max(1, int(world_size))
        batch = global_batch(per_rank_batch, grad_accum, count)
        self._job = {
            "model": _model_to_dict(core_model),
            "preset": str(preset),
            "flops_per_token": float(flops_per_token),
            "peak_flops": float(peak_flops),
            "mesh": {"count": count},
            "data": {
                "parquet": [str(path) for path in paths],
                "tokenizer_path": str(tokenizer),
                "data_seed": int(data_seed),
            },
            "run": {
                "batch": batch,
                "grad_accum": int(grad_accum),
                "seq": int(seq),
                "seed": int(seed),
                "num_iterations": int(steps),
                "log_every": int(log_every),
                "save_every": int(save_every),
                "resume": 1 if resume else 0,
            },
            "output": {
                "out_dir": str(out_dir),
                "checkpoint_path": str(checkpoint_path),
            },
        }

    @classmethod
    def from_dict(cls, data) -> "MeshTrainConfig":
        """Rebuild a client from a job dictionary. The child uses this."""
        job = cls.__new__(cls)
        job._lib = _lib.load()
        job._keepalive = []
        job._mesh = None
        job._config = None
        job._job = copy.deepcopy(dict(data))
        return job

    @property
    def out_dir(self) -> str:
        """The run directory."""
        return str(self._job["output"]["out_dir"])

    @property
    def global_sequences(self) -> int:
        """The global sequence count of one step."""
        return int(self._job["run"]["batch"])

    @property
    def sequence_length(self) -> int:
        """The sequence length of one step. A zero `seq` selects the model."""
        seq = int(self._job["run"]["seq"])
        return seq if seq > 0 else int(self._job["model"]["seq_len"])

    @property
    def tokens_per_step(self) -> int:
        """The global token count of one step."""
        return self.global_sequences * self.sequence_length

    def checkpoint_path(self) -> Path:
        """The canonical checkpoint path."""
        path = str(self._job["output"]["checkpoint_path"])
        return Path(path) if path else Path(self.out_dir) / "checkpoint.ckpt"

    def metrics_path(self) -> Path:
        """The per-step metrics path."""
        return Path(self.out_dir) / "metrics.jsonl"

    def summary_path(self) -> Path:
        """The run summary path."""
        return Path(self.out_dir) / "summary.json"

    def job_json(self) -> str:
        """The job as one JSON line for the child process."""
        return json.dumps(self._job, sort_keys=True)

    def run(self) -> MeshTrainResult:
        """Runs the whole job and returns the final loss and the summary."""
        config = self._to_c()
        summary = _core.MeshTrainSummary()
        before = _core.last_error(self._lib)
        loss = float(self._lib.nanochat_mesh_train_run(
            ctypes.byref(config), ctypes.byref(summary)))
        after = _core.last_error(self._lib)
        if after and after != before:
            raise _core.NanochatError("nanochat_mesh_train_run", after)
        checkpoint = summary.checkpoint.decode("utf-8", "replace")
        return MeshTrainResult(loss=loss, steps=int(summary.steps),
                               interrupted=bool(summary.interrupted),
                               mean_loss=float(summary.mean_loss),
                               steady_mfu=float(summary.steady_mfu),
                               checkpoint=checkpoint)

    def close(self) -> None:
        """Releases the mesh handle and the cached config."""
        if getattr(self, "_mesh", None):
            self._lib.nanochat_mesh_free(self._mesh)
            self._mesh = None
        self._config = None
        self._keepalive = []

    def __enter__(self) -> "MeshTrainConfig":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _to_c(self) -> "_core.MeshTrainConfig":
        """Builds the C config once and caches it."""
        if self._config is not None:
            return self._config
        job = self._job
        model, pattern = _core_model_config(_model_from_dict(job["model"]))
        self._keepalive.append(pattern)
        core = _core.MeshTrainConfig()
        core.model = model
        self._set_string(core, "preset", job["preset"])
        core.flops_per_token = float(job["flops_per_token"])
        core.peak_flops = float(job["peak_flops"])
        count = max(1, int(job["mesh"]["count"]))
        indices = (ctypes.c_int * count)(*range(count))
        self._keepalive.append(indices)
        self._mesh = self._lib.nanochat_mesh_create(indices, count)
        _core.check_handle(self._lib, self._mesh, "nanochat_mesh_create")
        core.mesh = self._mesh
        data = job["data"]
        self._set_string(core.data, "tokenizer_path", data["tokenizer_path"])
        core.data.data_seed = int(data["data_seed"])
        path_bytes = [str(path).encode("utf-8") for path in data["parquet"]]
        self._keepalive.extend(path_bytes)
        path_array = (ctypes.c_char_p * len(path_bytes))(*path_bytes)
        self._keepalive.append(path_array)
        core.data.parquet = path_array
        core.data.parquet_count = len(path_bytes)
        run = job["run"]
        core.run.batch = int(run["batch"])
        core.run.grad_accum = int(run["grad_accum"])
        core.run.seq = int(run["seq"])
        core.run.seed = int(run["seed"])
        core.run.num_iterations = int(run["num_iterations"])
        core.run.log_every = int(run["log_every"])
        core.run.save_every = int(run["save_every"])
        core.run.resume = int(run["resume"])
        output = job["output"]
        self._set_string(core.output, "out_dir", output["out_dir"])
        self._set_string(core.output, "checkpoint_path",
                         output["checkpoint_path"])
        self._config = core
        return core

    def _set_string(self, container, name: str, value) -> None:
        """Stores one encoded string in the C struct and keeps it alive."""
        if value is None:
            encoded = None
        else:
            encoded = str(value).encode("utf-8")
            self._keepalive.append(encoded)
        setattr(container, name, encoded)
