# Build

## File layout

```
nanochat.cpp/
  .clang-format
  MODULE.bazel  .bazelrc
  README.md  DESIGN.md  AGENTS.md  CONTRIBUTING.md
  docs/                        # reference and how-to
  include/nanochat/
    config.h  tensor.h  kernels.h  model.h  optim.h  data.h  sandbox.h
    tokenizer.h  dataloader.h  rand.h  sampler.h  scheduler.h  logger.h  mfu.h
  src/
    ops.cc  model.cc  generate.cc  optim.cc  train.cc  data.cc  eval.cc
    *_main.cc
  backends/
    cpu/kernels.cc  cpu/kernels_test.cc
    cuda/device.cu  cuda/gemm.cu  cuda/cuda_runtime_test.cc
    cuda/kernels/{rms_norm,qk_prep,attention,pointwise,classifier,
                  embedding,adamw,muon,global_norm}.cu
    cuda/kernels/<family>_test.cc + <family>_benchmark.cc next to each kernel
    cuda/kernels/device_utils.cuh
    cuda/kernels/testing/        # host reference headers and test helpers
  dev/kernels/                   # unpromoted prototypes and the toolchain spike
  tests/oracle_test.cc  tests/data/debug_state.bin
  tools/
    nanochat                   # the single execution entry point
    gpu.sh  sandbox.sh         # broker + sandboxed launcher
    sandbox/profiles.conf      # per-profile cgroup budgets
    sandbox/verify.sh          # read-back check of the applied limits
    sandbox/nanochat.slice     # aggregate user-slice budget (drop-in)
    sandbox/README.md
  data/*.bin
```

## Bazel

**Bazel** is the primary driver: fine-grained targets map to kernel families,
and the action cache keeps parallel work from rebuilding the world. Config
settings select backend/precision/arch; `select()` handles linking.

The tree pins **Bazel 9.2.0** in `.bazelversion`. Bazel does not read that file
itself, so `tools/nanochat` runs Bazelisk when it is on `PATH` and falls back to
a system `bazel` otherwise. Install Bazelisk once per host; see
[sandbox.md](sandbox.md).

- `.bazelrc` sets `--@rules_cuda//cuda:archs=sm_61` (or `sm_75`).
- `CUDA_HOME=/usr` on this machine (no `/usr/local/cuda`).
- GPU tests are tagged `gpu` + `manual` and run outside the Bazel sandbox.
- Run tests through `tools/nanochat test`; it adds `--run_under` so each test
  action is sandboxed, and forces `--spawn_strategy=local` so the wrapper can
  reach the systemd user manager. See [sandbox.md](sandbox.md).

### Resource bounds

Every worktree and verifier is its own Bazel workspace, so it starts its own
persistent server. On a small shared host that adds up, so `.bazelrc` bounds
each server and `tools/nanochat` serializes builds:

- `startup --host_jvm_args=-Xmx1500m` caps the server heap.
- `startup --max_idle_secs=600` and `startup --shutdown_on_low_sys_mem` let an
  idle server exit instead of lingering for the three-hour default.
- `build --jobs=4 --local_resources=memory=3072` bounds one server's local
  action pool.
- `tools/nanochat build` and `tools/nanochat test` take an exclusive lock
  (`/tmp/nanochat-build.lock`) so at most one worktree compiles at a time. The
  lock descriptor is closed on the Bazel client so the detached server cannot
  inherit it.
- `tools/nanochat shutdown` stops the current workspace's server immediately.

See [sandbox.md](sandbox.md) for the host budget this protects.

## Makefile fallback

A single Makefile mirrors llm.c's autodetection (nvcc presence, arch via
`nvidia-smi`, optional cuDNN/NCCL/OpenMP). The source layout is identical, so
the build file is swappable.

Runtime links against `cudart` + cuBLAS only on CUDA, and `libm` on CPU.
