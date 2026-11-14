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

# Warnings every target builds with. A warning is an error in both passes, so
# the inner loop and the merge gate build the same way and a warning cannot sit
# unnoticed in a log. `docs/build.md` states the policy.
#
# `-O2` is explicit: Bazel's default `fastbuild` mode adds no optimization
# flag, so without it the host code (the whole workflow, the data loader, and
# the host-side reductions) is compiled at `-O0`. See docs/performance.md.
NANOCHAT_COPTS = [
    "-O2",
    "-Wall",
    "-Wextra",
    "-Werror",
]

# Device-side options. Host warning flags must be forwarded with -Xcompiler,
# because nvcc does not recognise them directly. `-O2` optimises the device
# code; `-Xcompiler -O2` optimises the host pass of each .cu translation unit.
#
# The deprecation warning for the legacy cuBLAS enums that `gemm.cu` still names
# is the one exception: the interposer reports those enums directly, so the
# warning carries no information here. A target that compiles against the
# CUTLASS headers states its own `copts` instead of using this list, because
# those headers emit thousands of warnings of their own.
#
# `20012` is the redundancy notice of a `__host__ __device__` annotation on a
# function that is defaulted on its first declaration. `include/nanochat/tensor.h`
# states the pair on the defaulted `Bf16` constructor: the annotation documents
# the intent, the device pass ignores it, and a newer toolkit reports that fact
# as a warning. The header is frozen, and the diagnostic says nothing about the
# code, so the option below suppresses that one number and not the class.
NANOCHAT_CUDA_COPTS = [
    "-O2",
    "-Xcompiler",
    "-O2",
    "-Xcompiler",
    "-Wall",
    "-Xcompiler",
    "-Wextra",
    "-Xcompiler",
    "-Werror",
    "-Xcompiler",
    "-Wno-deprecated-declarations",
    "-Werror",
    "all-warnings",
    "-diag-suppress",
    "20012",
]

def nanochat_copts():
    """Host compiler options shared by every target."""
    return list(NANOCHAT_COPTS)

def nanochat_cuda_copts():
    """CUDA compiler options shared by every kernel target."""
    return list(NANOCHAT_CUDA_COPTS)

def precision_defines():
    """The #define for the build precision.

    One value, not one boolean per precision. `include/nanochat/tensor.h` maps
    it to `nanochat::BuildPrecision`.
    """
    return select({
        "//:fp32": ["NANOCHAT_PRECISION=0"],
        "//:bf16": ["NANOCHAT_PRECISION=1"],
        "//:fp8": ["NANOCHAT_PRECISION=2"],
    })
