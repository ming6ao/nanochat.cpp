# Build

## File layout

```
nanochat.cpp/
  .clang-format
  MODULE.bazel  .bazelrc
  README.md  DESIGN.md  AGENTS.md  CONTRIBUTING.md
  docs/                        # reference and how-to
  include/nanochat/
    config.h  tensor.h  kernels.h  model.h  optim.h  data.h
    tokenizer.h  dataloader.h  rand.h  sampler.h  scheduler.h  logger.h  mfu.h
  src/
    ops.cc  model.cc  generate.cc  optim.cc  train.cc  data.cc  eval.cc
    *_main.cc
  backends/
    cpu/kernels.cc
    cuda/device.cu  cuda/gemm.cu
    cuda/kernels/{rms_norm,qk_prep,attention,pointwise,classifier,
                  embedding,adamw,muon,global_norm}.cu
    cuda/kernels/device_utils.cuh
  dev/kernels/                 # standalone test+benchmark per kernel
  tests/oracle_test.cc  tests/debug_state.bin
  tools/
    gpu.sh  sandbox.sh         # broker + sandboxed launcher
    sandbox/profiles.conf      # per-profile cgroup budgets
    sandbox/nanochat.slice     # aggregate user-slice budget (drop-in)
    sandbox/README.md
  data/*.bin
```

## Bazel

**Bazel** is the primary driver: fine-grained targets map to kernel families,
and the action cache keeps parallel work from rebuilding the world. Config
settings select backend/precision/arch; `select()` handles linking.

- `.bazelrc` sets `--@rules_cuda//cuda:archs=sm_61` (or `sm_75`).
- `CUDA_HOME=/usr` on this machine (no `/usr/local/cuda`).
- GPU tests are tagged `gpu` + `manual` and run outside the Bazel sandbox.
- Test actions inherit resource limits in one of two ways (see
  [sandbox.md](sandbox.md)): GPU and `manual` targets already run with
  `--spawn_strategy=local --no-sandbox --local_test_jobs=1`, so
  `--run_under='tools/sandbox.sh --profile=t1-gpu --'` reaches the user manager
  directly; for hermetic CPU tests, start the Bazel server inside a sandbox
  scope so every test process inherits the same cgroup.

## Makefile fallback

A single Makefile mirrors llm.c's autodetection (nvcc presence, arch via
`nvidia-smi`, optional cuDNN/NCCL/OpenMP). The source layout is identical, so
the build file is swappable.

Runtime links against `cudart` + cuBLAS only on CUDA, and `libm` on CPU.
