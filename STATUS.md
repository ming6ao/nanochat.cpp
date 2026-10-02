# Status

Per-workstream state for `nanochat.cpp`. Maintained by the architect/integrator.
States: `todo`, `wip`, `cpu-green`, `gpu-green`, `done`.

Campaign `nanochat-cpp`, feature branch `nanochat-cpp`. Waves 0-8 are
integrated. Live resume state is in `.sliceme/RESUME.md`.

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
| P1 | CUDA optimizer kernels | `w1-cuda-kernels-optim` (`backends/cuda/kernels`, `dev/kernels`) | done | Landed; AdamW 1.49e-08, Muon 4.25e-07 device-vs-reference. |
| P2 | Workflow (ops, model, generate) | `w2-src-workflow` (`src`) | done | Landed; prefill/decode consistency bit-identical. |
| P2 | Optimizer grouping and schedules | `w2-src-optim` (`src`) | done | Landed; `//src:optim_test` green. |
| P2 | Training harness | `w2-src-harness` (`src`) | done | Landed; data/train/eval/CLI, 9 CPU tests green. |
| — | Backend link selection (`--config=cuda`) | architect (`BUILD.bazel`, `.bazelrc`, `src/BUILD.bazel`, `backends/cuda/BUILD.bazel`, `tools/nanochat`) | done | Landed; `//src:model` selects CPU or CUDA at link time. |
| P2.5 | CUDA model bring-up (host/device correctness) | `w3-cuda-model-bringup` (`src`) | done | Landed; `//src:model_oracle_gpu_test` forward `4.66e-09`, backward `5.96e-08`. |
| P3 | Oracle parity (CUDA) | `w3-oracle-parity` (`tests`) | done | Landed; forward `4.66e-09`, backward `5.96e-08`, optimizer step-1 `2.38e-07`; multi-step trajectory behavioral only (fixture ill-conditioned at `eps=1e-10`). |
| P3 | Turing port (sm_75, fp16) | `w3-turing-port` (`backends/cuda`) | done | Landed; fp16 runs on sm_61 (GEMM tol 1e-2, kernels 2e-3); sm_75 compile-only (no Turing device). Fixed Pascal fp16 atomics and fp16 cuBLAS compute type. |
| P3 | Fusion/tuning | `w3-fusion-tuning` (`dev/kernels`) | done | Landed; fused QkPrep legal, wins on launch-bound tiny shapes (fwd 2.8x), ties on medium backward; decode graph baseline 1.10x. Backend promotion is a follow-up. |

## Build and resource notes

- Run everything through `tools/nanochat` (`build`, `test`, `test --gpu`,
  `check`, `shutdown`). Builds are serialized by a shared lock and bounded in
  `.bazelrc`; see [docs/sandbox.md](docs/sandbox.md).
- The host is one GTX 1080 Ti (sm_61) under WSL2; `tools/nanochat test --gpu`
  holds the exclusive broker.
- `.bazelignore` keeps Bazel out of `.sliceme/worktrees`, which otherwise breaks
  `bazel build //...` at the campaign root.
- Backend selection is a link-time config: default `--define=backend=cpu`,
  `--config=cuda` for CUDA. `tools/nanochat test --gpu` adds `--config=cuda` so
  GPU-tagged workflow tests exercise the CUDA backend.
