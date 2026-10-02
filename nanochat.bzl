"""Shared build definitions for nanochat.cpp.

Every per-directory BUILD file loads its CUDA rules and the project's common
compiler options from here, so the toolchain wiring (rules_cuda) and the
warning set live in exactly one place. See docs/build.md.
"""

load(
    "@rules_cuda//cuda:defs.bzl",
    _cuda_binary = "cuda_binary",
    _cuda_library = "cuda_library",
    _cuda_test = "cuda_test",
)

# Re-exported so kernel packages have a single import point and the rule set
# can be swapped (for example to a hand-rolled nvcc rule) without touching every
# BUILD file.
cuda_binary = _cuda_binary
cuda_library = _cuda_library
cuda_test = _cuda_test

# Warnings every target builds with. The Definition of Done requires a clean
# build; -Werror is enabled by the integrator at the merge gate rather than in
# the inner development loop.
NANOCHAT_COPTS = [
    "-Wall",
    "-Wextra",
]

# Device-side options. Host warning flags must be forwarded with -Xcompiler,
# because nvcc does not recognise them directly.
NANOCHAT_CUDA_COPTS = [
    "-Xcompiler",
    "-Wall",
    "-Xcompiler",
    "-Wextra",
]

def nanochat_copts():
    """Host compiler options shared by every target."""
    return list(NANOCHAT_COPTS)

def nanochat_cuda_copts():
    """CUDA compiler options shared by every kernel target."""
    return list(NANOCHAT_CUDA_COPTS)

def precision_defines():
    """The #define for the selected precision (`--config=fp32` / `fp16`).

    The config_settings live in the root package, which is the aggregation
    point every per-directory BUILD file can rely on.
    """
    return select({
        "//:fp16": ["NANOCHAT_PRECISION_FP16"],
        "//:fp32": ["NANOCHAT_PRECISION_FP32"],
    })
