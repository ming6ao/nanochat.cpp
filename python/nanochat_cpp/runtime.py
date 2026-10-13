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
import ctypes
import dataclasses

from . import _core, _lib
from .mesh import Mesh

__all__ = ["MeshRuntime", "StepMetrics"]


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

    def load(self, path) -> None:
        """Load one checkpoint into every replica, optimizer state included.

        The runtime :attr:`step` moves to the stored step. Call :meth:`step`
        after this to learn where to resume.
        """
        self._void("nanochat_runtime_load", str(path).encode("utf-8"))

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
