# Status

Per-workstream state for `nanochat.cpp`. Maintained by the architect/integrator.
States: `todo`, `wip`, `cpu-green`, `gpu-green`, `done`.

| Phase | Workstream | Owner (directory) | State | Notes |
|---|---|---|---|---|
| P0 | Build, Bazel wiring, CUDA/Pascal spike | `p0-build` (`nanochat.cpp/`) | gpu-green | `MODULE.bazel`, `.bazelrc`, `.clang-format`, root `BUILD.bazel`, `nanochat.bzl`; sm_61 default, sm_75 and fp32/fp16 configs; Pascal spike + gpu-tagged test. |
| P0 | Freeze public API and host headers | `p0-headers` (`include/nanochat`) | todo | Depends on the build. |
| P0 | CPU reference backend | `p0-cpu-backend` (`backends/cpu`) | todo | Depends on headers. |
| P0 | Oracle fixture and CPU oracle test | `p0-oracle-fixture` (`tests`, `tools`) | todo | Depends on headers. |
| P0 | Model skeleton on the CPU backend | `p0-model-skeleton` (`src`) | todo | Depends on CPU backend and oracle. |
| P1 | CUDA device runtime and cuBLAS GEMM | `w1-cuda-runtime` (`backends/cuda`) | todo | Depends on the build and headers. |
| P1 | CUDA row/elementwise kernels | `w1-cuda-kernels-row` (`backends/cuda/kernels`, `dev/kernels`) | todo | Depends on the CUDA runtime. |
| P1 | CUDA attention kernels | `w1-cuda-kernels-attn` (`backends/cuda/kernels`, `dev/kernels`) | todo | Depends on the CUDA runtime. |
| P1 | CUDA optimizer kernels | `w1-cuda-kernels-optim` (`backends/cuda/kernels`, `dev/kernels`) | todo | Depends on the CUDA runtime. |
| P2 | Workflow, optimizer, harness | `w2-src-*` (`src`) | todo | Depends on P0. |
| P3 | Oracle parity, Turing port, fusion | `w3-*` (`tests`, `backends/cuda`, `dev/kernels`) | todo | Wave-boundary gates. |

## Build notes

- The workspace is `nanochat.cpp/`; `cd nanochat.cpp && tools/nanochat build`.
- CUDA is the Debian/Ubuntu toolkit; the host compiler is pinned to `gcc-12`
  in `.bazelrc` because CUDA 12.0 rejects gcc 13.
- Per-directory BUILD files load their CUDA rules and common options from
  `//:nanochat.bzl`; the precision `config_setting`s and the `:all_tests`
  aggregation point live in the root `BUILD.bazel`.
