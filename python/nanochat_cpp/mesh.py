"""The device mesh and the placement language for the single-client
multi-device runtime.

This module is the programming-model surface described in
``docs/client-runtime-design.md``. It mirrors JAX on one host: ``devices()``
enumerates the local accelerators, ``Mesh`` gives them one name, and
``NamedSharding`` with ``PartitionSpec`` states where a tensor lives.

The module imports the Python standard library and the shared-library loader
only. The runtime itself (the ``Trainer`` over the mesh) is a later phase; this
module carries the device view and the placement rules.
"""

from __future__ import annotations

import dataclasses

from . import _core, _lib

__all__ = [
    "DATA_AXIS",
    "Device",
    "Mesh",
    "MeshError",
    "NamedSharding",
    "PartitionSpec",
    "devices",
    "validate_batch_sharding",
    "validate_parameter_sharding",
]

#: The one supported mesh axis. The fixed graph is data parallel, so the mesh
#: is rank 1 and the axis is named ``"data"``
#: (docs/client-runtime-design.md section 5.2).
DATA_AXIS = "data"


class MeshError(ValueError):
    """A mesh or a placement outside the supported set."""


@dataclasses.dataclass(frozen=True)
class Device:
    """One local accelerator.

    ``ordinal`` is the position in the ``devices()`` list. ``index`` is the
    backend device index. ``is_device`` is false for the reference CPU device
    and true for an accelerator.
    """

    ordinal: int
    index: int
    name: str
    total_memory_bytes: int = 0
    compute_major: int = 0
    compute_minor: int = 0
    is_device: bool = False

    @property
    def is_gpu(self) -> bool:
        """True when this device is an accelerator."""
        return self.is_device

    def __str__(self) -> str:
        return f"cuda:{self.index}" if self.is_device else f"cpu:{self.index}"


def devices() -> tuple[Device, ...]:
    """Every local accelerator, in ordinal order.

    On the reference backend this is one CPU device. On the CUDA backend it is
    every visible card. The call loads the shared library and reads the device
    table; it does not create a device context in the calling thread.
    """
    library = _lib.load()
    count = _core.device_count(library)
    result: list[Device] = []
    for ordinal in range(count):
        info = _core.device_at(library, ordinal)
        name = (
            info.device_name.decode("utf-8", "replace")
            if info.device_name
            else "unknown"
        )
        result.append(
            Device(
                ordinal=ordinal,
                index=int(info.device_index),
                name=name,
                total_memory_bytes=int(info.total_memory_bytes),
                compute_major=int(info.compute_major),
                compute_minor=int(info.compute_minor),
                is_device=int(info.compute_major) > 0,
            )
        )
    return tuple(result)


class PartitionSpec:
    """The sharded axes of one tensor.

    A value of ``None`` means the axis is replicated. A string names a mesh
    axis. ``PartitionSpec()`` is fully replicated; ``PartitionSpec("data")``
    shards tensor axis 0 on the ``"data"`` axis.
    """

    def __init__(self, *dims) -> None:
        self._dims = tuple(dims)

    @property
    def dims(self) -> tuple:
        """The per-axis placement, in tensor-axis order."""
        return self._dims

    def is_replicated(self) -> bool:
        """True when every axis is replicated."""
        return all(dim is None for dim in self._dims)

    def __eq__(self, other) -> bool:
        return isinstance(other, PartitionSpec) and other._dims == self._dims

    def __hash__(self) -> int:
        return hash(self._dims)

    def __repr__(self) -> str:
        return f"PartitionSpec{self._dims!r}"


class NamedSharding:
    """A `PartitionSpec` bound to a mesh. The mesh axes are checked here."""

    def __init__(self, mesh: Mesh, spec: PartitionSpec) -> None:
        if not isinstance(spec, PartitionSpec):
            raise MeshError("spec must be a PartitionSpec")
        for dim in spec.dims:
            if dim is not None and dim not in mesh.axis_names:
                raise MeshError(
                    f"unknown mesh axis {dim!r}; the mesh axes are "
                    f"{mesh.axis_names!r}"
                )
        self._mesh = mesh
        self._spec = spec

    @property
    def mesh(self) -> Mesh:
        """The mesh the placement binds to."""
        return self._mesh

    @property
    def spec(self) -> PartitionSpec:
        """The placement itself."""
        return self._spec

    def __repr__(self) -> str:
        return f"NamedSharding(mesh={self._mesh!r}, spec={self._spec!r})"


class Mesh:
    """A named device grid.

    The runtime supports a rank-1 mesh with the one axis named ``"data"``. Any
    other shape or name raises ``MeshError`` at construction, so an unsupported
    placement fails before a run starts.
    """

    def __init__(self, device_list, axis_names=(DATA_AXIS,)) -> None:
        if isinstance(axis_names, str):
            axis_names = (axis_names,)
        names = tuple(axis_names)
        mesh_devices = tuple(device_list)
        if len(names) != 1:
            raise MeshError(
                "the runtime supports a rank-1 mesh only; axis_names has "
                f"{len(names)} entries"
            )
        if names[0] != DATA_AXIS:
            raise MeshError(
                f"the one mesh axis must be named {DATA_AXIS!r}; got "
                f"{names[0]!r}"
            )
        if not mesh_devices:
            raise MeshError("the mesh has no devices")
        self._devices = mesh_devices
        self._axis_names = names

    @property
    def devices(self) -> tuple[Device, ...]:
        """The devices in the one axis."""
        return self._devices

    @property
    def axis_names(self) -> tuple[str, ...]:
        """The axis names; one entry, ``"data"``."""
        return self._axis_names

    @property
    def shape(self) -> tuple[int, ...]:
        """The mesh shape; one entry, the device count."""
        return (len(self._devices),)

    @property
    def size(self) -> int:
        """The number of devices in the mesh."""
        return len(self._devices)

    def __len__(self) -> int:
        return len(self._devices)

    def __repr__(self) -> str:
        return (
            f"Mesh(devices={len(self._devices)}, "
            f"axis_names={self._axis_names!r})"
        )


def validate_parameter_sharding(sharding: NamedSharding) -> NamedSharding:
    """Checks the placement of model parameters and optimizer state.

    The fixed graph keeps one full replica per device, so every parameter must
    be replicated. A sharded parameter is tensor parallel, which is out of
    scope (docs/client-runtime-design.md section 5.4).
    """
    if not sharding.spec.is_replicated():
        raise MeshError(
            "model parameters must be replicated (PartitionSpec()); sharded "
            "parameters are tensor parallel, which is out of scope"
        )
    return sharding


def validate_batch_sharding(sharding: NamedSharding) -> NamedSharding:
    """Checks the placement of the global batch.

    The controller splits one global batch into contiguous slices, so the batch
    must be sharded on the one ``"data"`` axis: ``PartitionSpec("data")``.
    """
    if sharding.spec.dims != (DATA_AXIS,):
        raise MeshError(
            "the global batch must be sharded on the 'data' axis: "
            "PartitionSpec('data')"
        )
    return sharding
