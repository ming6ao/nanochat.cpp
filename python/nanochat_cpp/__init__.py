"""Python bridge for nanochat.cpp.

This package provides entry points that mirror the PyTorch nanochat scripts
(``python -m scripts.base_train``) while running the C++ implementation. It
locates the reference nanochat checkout for the tokenizer and configuration
math, passes the parquet dataset and the ``NCTOKEN1`` artifact to the C++
binaries (which tokenize during the run; docs/parquet-native.md), and launches
them under ``tools/nanochat`` so the resource sandbox and the GPU broker still
apply.

Nothing here is a runtime dependency of the C++ binary; it is orchestration
around it. See ``python/nanochat_cpp/base_train.py`` and docs/testing.md.
"""

__all__ = ["base_eval", "checkpoint", "config", "data", "eval_fixture",
           "launcher", "reference"]
