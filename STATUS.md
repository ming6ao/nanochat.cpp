# Status

Per-workstream state for `nanochat.cpp`. Maintained by the architect/integrator.
States: `todo`, `wip`, `cpu-green`, `gpu-green`, `done`.

Campaign `nanochat-cpp`, feature branch `nanochat-cpp`. Waves 0-4 are integrated;
wave 5 is in progress. Live resume state is in `.sliceme/RESUME.md`.

| Phase | Workstream | Owner (directory) | State | Notes |
|---|---|---|---|---|
| P0 | Build, Bazel wiring, CUDA/Pascal spike | `p0-build` (`nanochat.cpp/`) | done | Landed. |
| P0 | Freeze public API and host headers | `p0-headers` (`include/nanochat`) | done | Landed. |
| P0 | CPU reference backend | `p0-cpu-backend` (`backends/cpu`) | done | Landed; every seam symbol as a naive loop. |
| P0 | Oracle fixture and CPU oracle test | `p0-oracle-fixture` (`tests`, `tools`) | done | Landed; `tests/data/debug_state.bin` committed. |
| P0 | Model skeleton on the CPU backend | `p0-model-skeleton` (`src`) | done | Landed; forward `2.79e-09`, backward `2.98e-08` vs oracle. |
| P1 | CUDA device runtime and cuBLAS GEMM | `w1-cuda-runtime` (`backends/cuda`) | done | Landed; fixed an unsigned-wraparound bug in its test. |
| P1 | CUDA row/elementwise kernels | `w1-cuda-kernels-row` (`backends/cuda/kernels`, `dev/kernels`) | done | Landed; rms_norm, qk_prep, pointwise, global_norm. |
| P1 | CUDA attention kernels | `w1-cuda-kernels-attn` (`backends/cuda/kernels`, `dev/kernels`) | done | Landed; attention, classifier, embedding. |
| P1 | CUDA optimizer kernels | `w1-cuda-kernels-optim` (`backends/cuda/kernels`, `dev/kernels`) | wip | Worker done, changes uncommitted in its worktree; not yet landed. |
| P2 | Workflow (ops, model, generate) | `w2-src-workflow` (`src`) | done | Landed; prefill/decode consistency bit-identical. |
| P2 | Optimizer grouping and schedules | `w2-src-optim` (`src`) | wip | Worker running at save time. |
| P2 | Training harness | `w2-src-harness` (`src`) | todo | Wave 6. |
| P3 | Oracle parity (CUDA) | `w3-oracle-parity` (`tests`) | todo | Wave 7; the wave-boundary gate. |
| P3 | Turing port (sm_75, fp16) | `w3-turing-port` (`backends/cuda`) | todo | Wave 8. |
| P3 | Fusion/tuning | `w3-fusion-tuning` (`dev/kernels`) | todo | Wave 8. |

## Build and resource notes

- Run everything through `tools/nanochat` (`build`, `test`, `test --gpu`,
  `check`, `shutdown`). Builds are serialized by a shared lock and bounded in
  `.bazelrc`; see [docs/sandbox.md](docs/sandbox.md).
- The host is one GTX 1080 Ti (sm_61) under WSL2; `tools/nanochat test --gpu`
  holds the exclusive broker.
- `.bazelignore` keeps Bazel out of `.sliceme/worktrees`, which otherwise breaks
  `bazel build //...` at the campaign root.
