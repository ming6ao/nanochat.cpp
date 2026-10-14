"""The Python API for nanochat.cpp (docs/python.md).

One Python surface with three layers:

* the compute layer (``nanochat_cpp.api``) trains, evaluates, and generates in
  one process through the shared library and ``ctypes``;
* the planning layer (``nanochat_cpp.plan``) derives the model shape, the
  horizon, the batch size, and the rates;
* the orchestration layer (``nanochat_cpp.toolchain``) calls ``tools/nanochat``
  for build, test, lint, and doctor.

The package imports only the Python standard library. The compute layer never
imports ``torch``. ``chat`` owns the chat evaluation front end. ``rl`` owns
the reinforcement-learning bridge: it renders the prompts, drives the
persistent C++ worker, and scores the rollouts (docs/rl-notebook.md).
"""

from . import _build, _core, _lib, api, chat, distributed, mesh, plan, rl, runtime, sft, sft_data, toolchain
from ._build import BuildError
from ._core import NanochatError
from .api import (
    Config,
    EvalReport,
    Evaluator,
    GeneratedRow,
    Model,
    Optimizer,
    ParamView,
    ScoreReport,
    TensorView,
    TokenData,
    Tokenizer,
    Trainer,
    build,
    evaluate,
    no_grad,
    optimizer_config,
    preset_optimizer_config,
)
from .chat import render_conversation
from .mesh import (
    Device,
    Mesh,
    MeshError,
    NamedSharding,
    PartitionSpec,
    devices,
)
from .runtime import MeshRuntime
from .sft_data import build_train_mixture, build_val_mixture

__all__ = [
    "BuildError",
    "Config",
    "Device",
    "EvalReport",
    "Evaluator",
    "GeneratedRow",
    "Mesh",
    "MeshError",
    "MeshRuntime",
    "Model",
    "NanochatError",
    "NamedSharding",
    "Optimizer",
    "ParamView",
    "PartitionSpec",
    "ScoreReport",
    "TensorView",
    "TokenData",
    "Tokenizer",
    "Trainer",
    "build",
    "devices",
    "evaluate",
    "no_grad",
    "optimizer_config",
    "preset_optimizer_config",
    "render_conversation",
    "build_train_mixture",
    "build_val_mixture",
    "api",
    "chat",
    "distributed",
    "mesh",
    "plan",
    "rl",
    "runtime",
    "sft_data",
    "toolchain",
]
