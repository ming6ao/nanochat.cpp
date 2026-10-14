"""Frozen data classes for a notebook run.

:class:`ModelArgs` holds the architecture and maps to ``nc.Config``.
:class:`TrainArgs` holds the optimizer and schedule knobs and maps to
``nc.plan.compute_plan``. These two classes are the one place a notebook
names its model and its training hyperparameters.

Every class is frozen, so a variant is one ``dataclasses.replace`` call. The
JSON methods use ``dataclasses.asdict``, so a run directory records a
specification and can replay it. Do not add a derived field: the C library
owns every derived quantity.

The module imports ``nanochat_cpp`` inside the methods, not at module load. The
pure helpers then import without the shared library.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import nanochat_cpp as nc

__all__ = [
    "ModelArgs",
    "TrainArgs",
]

#: The minimum hidden width the smear and value gates read.
_MIN_HIDDEN_DIM = 24

#: The ``TrainArgs`` fields that are not optimizer or schedule fields.
_TRAIN_NON_OPTIMIZER_FIELDS = frozenset({
    "device_batch_size", "total_batch_size", "model_seed",
})

#: The optimizer and schedule fields a trial may override. Each name is a
#: ``compute_plan`` keyword. A ``None`` field keeps the plan default.
RATE_FIELDS = (
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
    "weight_decay", "warmup_steps", "warmdown_ratio", "final_lr_frac",
)


@dataclasses.dataclass(frozen=True)
class ModelArgs:
    """The architecture. One field per ``nanochat::Config`` field.

    ``padded_vocab_size`` is absent on purpose: ``nc.Config`` derives it from
    ``vocab_size``. ``hidden_dim`` must be divisible by ``num_heads``, and
    ``num_heads`` must be divisible by ``num_kv_heads``.
    """

    num_layers: int = 8
    hidden_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    seq_len: int = 512
    vocab_size: int = 32768
    window_pattern: str = "SSSL"
    rope_base: float = 100000.0
    value_embedding: bool = False

    def __post_init__(self) -> None:
        """Reject a bad shape early. The C check still runs at model create."""
        if self.num_layers < 1 or self.seq_len < 1:
            raise ValueError("num_layers and seq_len must be positive")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim < _MIN_HIDDEN_DIM:
            raise ValueError(
                f"hidden_dim must be at least {_MIN_HIDDEN_DIM}")
        if self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must divide by num_heads")
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must divide by num_kv_heads")

    @property
    def head_dim(self) -> int:
        """The width of one attention head."""
        return self.hidden_dim // self.num_heads

    def to_config(self) -> "nc.Config":
        """Build the handle the C library takes."""
        import nanochat_cpp as nc

        return nc.Config(
            num_layers=self.num_layers, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            seq_len=self.seq_len, vocab_size=self.vocab_size,
            window_pattern=self.window_pattern, rope_base=self.rope_base,
            value_embedding=self.value_embedding)

    def to_dict(self) -> dict:
        """A JSON-serializable mapping of every field."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelArgs":
        """Rebuild an instance from :meth:`to_dict` output."""
        return cls(**dict(data))

    def to_json(self) -> str:
        """A canonical JSON text of every field."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModelArgs":
        """Rebuild an instance from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    """The training knobs of one run.

    A ``None`` rate keeps the plan default. A value overrides the base rate.
    ``compute_plan`` then applies the batch scale.
    """

    device_batch_size: int = 8
    total_batch_size: int = 4096
    num_iterations: int = 50
    embedding_lr: float | None = None
    unembedding_lr: float | None = None
    matrix_lr: float | None = None
    scalar_lr: float | None = None
    weight_decay: float | None = None
    warmup_steps: int | None = None
    warmdown_ratio: float | None = None
    final_lr_frac: float | None = None
    model_seed: int = 42

    def __post_init__(self) -> None:
        if self.device_batch_size < 1 or self.total_batch_size < 1:
            raise ValueError("the batch sizes must be positive")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")

    def rate_overrides(self) -> dict[str, float | int]:
        """The non-``None`` rate fields, as ``compute_plan`` keywords."""
        return {name: getattr(self, name) for name in RATE_FIELDS
                if getattr(self, name) is not None}

    def plan(self, model: ModelArgs) -> "nc.plan.TrainPlan":
        """The full plan. ``seq_len`` and ``vocab_size`` come from ``model``."""
        import nanochat_cpp as nc

        return nc.plan.compute_plan(
            depth=model.num_layers, seq_len=model.seq_len,
            window_pattern=model.window_pattern, vocab_size=model.vocab_size,
            num_kv_heads=model.num_kv_heads, rope_base=model.rope_base,
            device_batch_size=self.device_batch_size,
            total_batch_size=self.total_batch_size,
            num_iterations=self.num_iterations,
            **self.rate_overrides())

    def optimizer_kwargs(self) -> dict[str, float | int]:
        """Every optimizer and schedule field, as ``nc.Optimizer`` keywords.

        A ``None`` rate is dropped, so the C library keeps its default. The
        batch sizes and the model seed stay out, because the optimizer does
        not read them.
        """
        fields = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in _TRAIN_NON_OPTIMIZER_FIELDS
        }
        fields = {name: value for name, value in fields.items()
                  if value is not None}
        fields["num_iterations"] = int(self.num_iterations)
        return fields

    def to_dict(self) -> dict:
        """A JSON-serializable mapping of every field."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainArgs":
        """Rebuild an instance from :meth:`to_dict` output."""
        return cls(**dict(data))

    def to_json(self) -> str:
        """A canonical JSON text of every field."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrainArgs":
        """Rebuild an instance from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))
