"""Python bridge for nanochat.cpp.

This package provides two entry points to the C++ implementation:

* the process bridge, which locates the reference nanochat checkout for the
  tokenizer and configuration math and launches the C++ binaries through
  ``tools/nanochat`` (``base_train``, ``base_eval``, ``data``, ``config``);
* the in-process API, which loads the shared library with ``ctypes`` and
  trains, evaluates, and generates inside one Python process
  (``nanochat_cpp.api``, docs/python-api.md).

Neither entry point is a runtime dependency of the C++ binary. Both import
only the Python standard library; the in-process API never imports ``torch``.
"""

from . import _build, _core, _lib, api
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
)

__all__ = [
    "BuildError",
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
    "base_eval",
    "checkpoint",
    "config",
    "data",
    "eval_fixture",
    "launcher",
    "reference",
]
