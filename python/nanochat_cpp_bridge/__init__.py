"""The adapter package: every heavy import of the Python tree has one home.

The tree has four boundaries (``docs/build.md``, the Python distribution), and
the arrow points down: a boundary imports a lower boundary and never a higher
one.

````text
the hosts        modal, kaggle, jupyter       tools/**, notebooks/**
the adapters     torch, numpy, tiktoken,     this package
                 pyarrow, filelock,
                 huggingface_hub, nanochat
the Python core  the standard library only   nanochat_cpp
the C++ core     libm, cudart, cuBLAS, zstd  src/**, include/**, backends/**
````

This package is the adapter package. It imports the standard library,
``nanochat_cpp``, and the extra packages the caller installed, and it never
imports ``tools/`` or ``notebooks/``. Nothing imports a host, so a
module here works the same in a notebook, in a Modal image, and in the test
loop.

The package body imports no submodule. ``import nanochat_cpp_bridge`` costs one
module and needs no extra, and ``nanochat_cpp_bridge.<module>`` resolves on
first access through the PEP 562 hook below. Every module then imports its
heavy package inside the function that needs it, so an installed extra is
required to *call*, not to *import*::

    import nanochat_cpp_bridge                      # no extra needed
    records = nanochat_cpp_bridge.torch_checkpoint.build_records(state_dict)

One distribution, two import packages: the base install carries this package
and the Python core with an empty dependency list, and the extras supply what
a task needs. A missing package raises
:class:`nanochat_cpp_bridge._extras.MissingExtra` naming the extras line.

Modules
-------
``torch_checkpoint``
    The torch half of the reference-checkpoint conversion: the lazy torch
    imports, the state-dict reading, the record builders, and the converter
    command line that ``tools/convert_checkpoint.py`` and the console script
    share.
``reference_data``
    The reference data surface: the parquet listing of ``nanochat.dataset``
    and the ``tiktoken`` tokenizer built from an ``NCTOKEN1`` artifact.
``reference_tasks``
    The hub dataset loader (``pyarrow`` and ``filelock``), the ``numpy``
    shuffle, and the reference code executor of HumanEval, behind the named
    seams the core is handed.
``hf_hub``
    The Hugging Face Hub save and load path of a checkpoint stage.
``sft_reference``
    The reference SFT mixture and renderer, the counterpart of the core's
    mixture builders.
``fixtures``
    The offline parity producers, run by ``tools/nanochat parity``.

This package imports the Python core and never the reverse: the core reaches
this package by being handed a callable, never by naming it.
"""

from __future__ import annotations

import importlib

__all__ = [
    "fixtures",
    "hf_hub",
    "reference_data",
    "reference_tasks",
    "sft_reference",
    "torch_checkpoint",
]


def __getattr__(name: str):
    """Import one named submodule on first access (PEP 562)."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Offer the module names to ``dir()`` without importing them."""
    return sorted(__all__)
