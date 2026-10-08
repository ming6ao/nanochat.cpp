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
#
# `-O2` is explicit: Bazel's default `fastbuild` mode adds no optimization
# flag, so without it the host code (the whole workflow, the data loader, and
# the host-side reductions) is compiled at `-O0`. The Makefile fallback already
# uses `-O2`; this keeps the Bazel path equivalent. See docs/performance.md.
NANOCHAT_COPTS = [
    "-O2",
    "-Wall",
    "-Wextra",
]

# Device-side options. Host warning flags must be forwarded with -Xcompiler,
# because nvcc does not recognise them directly. `-O2` optimises the device
# code; `-Xcompiler -O2` optimises the host pass of each .cu translation unit.
NANOCHAT_CUDA_COPTS = [
    "-O2",
    "-Xcompiler",
    "-O2",
    "-Xcompiler",
    "-Wall",
    "-Xcompiler",
    "-Wextra",
]

# The simulator compile gate (docs/simulator.md section 8). The host pass turns
# every warning into an error, except the deprecation warning for the legacy
# cuBLAS enums that `gemm.cu` still names; the interposer reports those enums
# directly. The device pass does the same through nvcc's warning classes. The
# last two entries ask ptxas for the per-kernel resource use.
SIM_GATE_CUDA_COPTS = [
    "-Xcompiler",
    "-Werror",
    "-Xcompiler",
    "-Wno-deprecated-declarations",
    "-Werror",
    "all-warnings",
    "-Xptxas",
    "-v",
]

def nanochat_copts():
    """Host compiler options shared by every target."""
    return select({
        "//:sim_gate": list(NANOCHAT_COPTS) + ["-Werror"],
        "//conditions:default": list(NANOCHAT_COPTS),
    })

def nanochat_cuda_copts():
    """CUDA compiler options shared by every kernel target."""
    return select({
        "//:sim_gate": list(NANOCHAT_CUDA_COPTS) + SIM_GATE_CUDA_COPTS,
        "//conditions:default": list(NANOCHAT_CUDA_COPTS),
    })

def precision_defines():
    """The #define for the selected precision (`--config=fp32` / `fp16`).

    The config_settings live in the root package, which is the aggregation
    point every per-directory BUILD file can rely on.
    """
    return select({
        "//:fp16": ["NANOCHAT_PRECISION_FP16"],
        "//:fp32": ["NANOCHAT_PRECISION_FP32"],
    })
