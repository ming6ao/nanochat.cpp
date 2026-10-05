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
    tokenizer/                 # the tokenizer host package (see src/tokenizer/)
      tokenizer.cc  split_pattern.{h,cc}  utf8.{h,cc}  unicode_tables.inc
      bpe_trainer.cc  parquet_reader.{h,cc}  shard_writer.{h,cc}
      tok_train_main.cc  tok_shard_main.cc
      *_test.cc
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

## Backends and selection

Backend and precision are build and link choices, not runtime switches.

Selection is a build/link choice:

```
--backend=cpu|cuda        link the backend library
--precision=fp32|fp16     select ComputeType
--arch=sm_61|sm_75        CUDA --generate-code
```

| Backend | Implements | Notes |
|---|---|---|
| `backends/cpu` | all of `kernels.h` with naive loops | reference, CI, oracle baseline; `-lm` (+ OpenMP) |
| `backends/cuda` | all of `kernels.h`; cuBLAS for GEMM | Pascal fp32, Turing fp16; cuDNN/NCCL optional |

The shared workflow (`ops.cc`, `model.cc`, `generate.cc`, `optim.cc`,
`train.cc`) compiles once and links against either backend. This is the one
improvement over llm.c, which duplicates its top-level files per backend.

### Adding a backend

Adding hardware = add `backends/<x>/` implementing `kernels.h` plus a config
value. `src/` and `include/` normally need no changes; a new kernel family
(below the replacement boundary) does touch the frozen headers, under
[DESIGN.md §7](../DESIGN.md).

The interface to implement is [kernels.md](kernels.md). The dependency rules a
backend must respect are in [DESIGN.md §2](../DESIGN.md): `backends/**` includes
only `nanochat/kernels.h` and `nanochat/tensor.h`, and a kernel knows nothing
about GPT.

## Precision

`-DNANOCHAT_PRECISION=FP32|FP16` selects `using ComputeType = ...;` at build
time. One precision per build; no runtime dtype dispatch.

- Pascal (sm_61): fp32.
- Turing (sm_75): fp16 permitted, with loss scaling.
- The backend reports supported dtypes via `GetCaps()`; an invalid combination
  fails fast at startup.

Precision is chosen at build time alongside the backend; see the
backends section above.

## Turing (T4, sm_75)

Design for the Turing target. The T4 card is the second GPU family that
`nanochat.cpp` supports. Read the backends and precision sections above for the
selection rules.

Status: partially implemented. The build configs exist. The correctness gates
on Turing are not recorded yet.

### 1. Goal

T4 is a first-class target. A Turing port uses the existing backend seam. No
new backend directory is necessary.

### 2. Capability difference

| Feature | Pascal sm_61 | Turing sm_75 |
|---|---|---|
| fp32 arithmetic | yes | yes |
| fp16 storage | software | native `__half` |
| fp16 tensor cores | no | yes |
| `__half` atomic add | fallback loop | native, sm_70 and newer |
| cuBLAS fp16 path | HGEMM, 16F compute | tensor-op, 32F compute |
| `mma` instructions | no | yes |

`GetCaps()` reports the differences. `caps.has_tensor_cores` is true for
`major >= 7`. `caps.supports_fp16` is true for sm_53 and newer.

### 3. Build configuration

| Config | Arch | Precision | Use |
|---|---|---|---|
| `sm_75` | sm_75 | fp32 | fp32 correctness gate |
| `t4` | sm_75 | fp16 | default Kaggle build |
| `t4-fp32` | sm_75 | fp32 | explicit fp32 GPU build |

The `t4` config includes `--config=cuda`. The half-precision build uses the
tensor-op GEMM path on this card.

The T4 supports both precisions. The fp32 build is the correctness gate. The
fp16 build is the performance build.

### 4. GEMM behavior

`backends/cuda/gemm.cu` selects the compute type from the device capability.
On sm_75 the fp16 build uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` with
`CUBLAS_COMPUTE_32F`. The accumulation stays in fp32. The fp32 build uses the
classic `cublasSgemm` path for a single GEMM.

The host graph and the optimizer state stay fp32 wherever the seam requires
it. See the precision section above.

### 5. Attention

The current attention kernel uses fp32 statistics and an online softmax. It
does not use tensor cores. A `mma` variant is later work under
[DESIGN.md §3](../DESIGN.md). A candidate belongs in `dev/kernels` first, and
it must beat the current kernel at the training shape.

### 6. Correctness gates

| Tier | Target | Precision |
|---|---|---|
| T1 | `//backends/cuda/kernels:precision_gpu_test` | fp32, fp16 |
| T1 | `//backends/cuda:cuda_runtime_gpu_test` | fp32 |
| T2 | `//tests:oracle_cuda_test` | fp32 |
| T2 | `//tests:train_parity_cuda_test` | fp32 |
| T3 | `//backends/cuda/kernels:attention_benchmark` | fp16 |

The first fp16 run may exceed the recorded Pascal tolerance. Re-measure the
worst error and record the new value in `precision_test.cc`. Do not widen the
tolerance without a measurement.

Extend `//backends/cuda:cuda_runtime_gpu_test` to assert three things on
sm_75: the device name, `has_tensor_cores`, and `supports_fp16`.

### 7. Two cards

A Kaggle Notebook gives two T4 cards. See
[host-portability.md](host-portability.md) for device selection and
[host-portability.md §8](host-portability.md) for the two options. The short form:

- Option A: two independent jobs, one per card.
- Option B: data-parallel training. Later work.

### 8. Definition of done

1. `tools/nanochat build --config=t4` succeeds.
2. Every T1 GPU test passes on the T4 in fp32 and fp16.
3. The oracle and train-parity gates pass on sm_75.
4. The recorded fp16 tolerance matches a fresh measurement.
5. `cuda_runtime_gpu_test` asserts the Turing capabilities.
6. [performance.md](performance.md) holds one T4 row at the training shape.
