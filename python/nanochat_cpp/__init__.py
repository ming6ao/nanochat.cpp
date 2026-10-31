"""The Python API for nanochat.cpp (docs/python.md).

One Python surface with three layers:

* the compute layer (``nanochat_cpp.api``) trains, evaluates, and generates in
  one process through the shared library and ``ctypes``;
* the planning layer (``nanochat_cpp.plan``) derives the model shape, the
  horizon, the batch size, and the rates;
* the orchestration layer (``nanochat_cpp.toolchain``) calls ``tools/nanochat``
  for build, test, lint, and doctor.

The package imports only the Python standard library (rule 1 of
``docs/dependency-layering-plan.md`` section 4), and the compute layer never
imports ``torch``. ``chat`` owns the chat evaluation front end. ``rl`` owns
the reinforcement-learning bridge: it renders the prompts, drives the
persistent C++ worker, and scores the rollouts (docs/rl-notebook.md).

Importing the package loads **no** submodule (section 2.3). Every name below --
a submodule and a public class or function -- resolves on first access through
the module ``__getattr__`` of PEP 562, so a caller that needs only ``api`` does
not pay for ``chat``, ``rl``, ``sft``, ``sft_data``, or ``toolchain``::

    import nanochat_cpp                       # one module, nothing else
    model = nanochat_cpp.Model(config)        # loads nanochat_cpp.api
    plan = nanochat_cpp.plan.compute_plan(..)  # loads nanochat_cpp.plan

Three things need a package the base install does not carry, and the core
reaches each through a callable a host registers, never through an import:

* ``seam`` -- the registration seam. ``seam.register_hub_loader(loader)``
  supplies the hub dataset loader behind the task classes, and
  ``seam.register_code_executor(executor)`` supplies the executor behind the
  HumanEval criterion. Nothing registered raises ``seam.MissingAdapter`` with
  the extra to install (section 6.2).
* ``checkpoint`` -- the checkpoint container read and write. A conversion from a
  PyTorch ``state_dict`` runs in the Z2 package of section 5.
* ``data`` -- the portable ``NCTOKEN1`` reader. The ``tiktoken`` encoding over
  it lives in the Z2 package of section 5.

The package never imports the Z2 package and never names it (rule 5 of section
8, gate G5).
"""

from __future__ import annotations

import importlib

#: The public names, in the order they were exported before the package became
#: lazy. Every one of them still resolves through ``__getattr__``.
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
    "MeshTrainConfig",
    "MissingAdapter",
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
    "build_train_mixture",
    "build_val_mixture",
    "devices",
    "evaluate",
    "no_grad",
    "optimizer_config",
    "preset_optimizer_config",
    "register_code_executor",
    "register_hub_loader",
    "render_conversation",
    "seam",
    "api",
    "chat",
    "distributed",
    "mesh",
    "pipeline",
    "plan",
    "rl",
    "runtime",
    "sft_data",
    "toolchain",
]

#: Every public name and the module that defines it. A name missing here is
#: tried as a submodule of this package by :func:`__getattr__`.
_EXPORTS = {
    "BuildError": ("_build", "BuildError"),
    "Config": ("api", "Config"),
    "Device": ("mesh", "Device"),
    "EvalReport": ("api", "EvalReport"),
    "Evaluator": ("api", "Evaluator"),
    "GeneratedRow": ("api", "GeneratedRow"),
    "Mesh": ("mesh", "Mesh"),
    "MeshError": ("mesh", "MeshError"),
    "MeshRuntime": ("runtime", "MeshRuntime"),
    "MeshTrainConfig": ("runtime", "MeshTrainConfig"),
    "MissingAdapter": ("seam", "MissingAdapter"),
    "Model": ("api", "Model"),
    "NanochatError": ("_core", "NanochatError"),
    "NamedSharding": ("mesh", "NamedSharding"),
    "Optimizer": ("api", "Optimizer"),
    "ParamView": ("api", "ParamView"),
    "PartitionSpec": ("mesh", "PartitionSpec"),
    "ScoreReport": ("api", "ScoreReport"),
    "TensorView": ("api", "TensorView"),
    "TokenData": ("api", "TokenData"),
    "Tokenizer": ("api", "Tokenizer"),
    "Trainer": ("api", "Trainer"),
    "build": ("api", "build"),
    "build_train_mixture": ("sft_data", "build_train_mixture"),
    "build_val_mixture": ("sft_data", "build_val_mixture"),
    "devices": ("mesh", "devices"),
    "evaluate": ("api", "evaluate"),
    "no_grad": ("api", "no_grad"),
    "optimizer_config": ("api", "optimizer_config"),
    "preset_optimizer_config": ("api", "preset_optimizer_config"),
    "register_code_executor": ("seam", "register_code_executor"),
    "register_hub_loader": ("seam", "register_hub_loader"),
    "render_conversation": ("chat", "render_conversation"),
}

#: The submodule names this package offers, so ``dir()`` and an attribute miss
#: stay honest without importing any of them.
_SUBMODULES = (
    "_build", "_core", "_entry", "_lib", "api", "chat", "checkpoint", "cli",
    "data", "distributed", "eval_fixture", "gpu_queue", "lint", "mesh",
    "pipeline", "plan", "prune", "reference", "rl", "run", "runtime",
    "sandbox", "seam", "sft", "sft_data", "tasks", "toolchain",
)


def _load(module_name: str):
    """Import one submodule of this package and return it."""
    return importlib.import_module(f".{module_name}", __name__)


def __getattr__(name: str):
    """Resolve a public name or a submodule on first access (PEP 562).

    The lookup order is the exports table, then the submodule list. Anything
    else raises ``AttributeError`` as usual, so ``hasattr`` and ``getattr``
    keep their normal meaning. The resolved object is cached in the module
    globals, so a second access costs a dictionary lookup.
    """
    export = _EXPORTS.get(name)
    if export is not None:
        module_name, attribute = export
        value = getattr(_load(module_name), attribute)
    elif name in _SUBMODULES:
        value = _load(name)
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names and the submodules without importing any."""
    return sorted(set(__all__) | set(_SUBMODULES))
