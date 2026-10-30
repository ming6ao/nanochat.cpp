# Build

## File layout

```
nanochat.cpp/
  .clang-format
  MODULE.bazel  .bazelrc
  README.md  DESIGN.md  AGENTS.md  CONTRIBUTING.md
  docs/                        # reference and how-to
  spec/                        # the contract vocabulary: prose and data, no code
    versions.md  capability_bits.md  block_kinds.md  pointwise_ops.md  decisions.md
  models/                      # the model axis: one directory per model package
    nanochat-default/
      spec.json                # the architecture description (the block list)
      tolerances.json          # this model's numeric contract
      fixtures/*.bin           # the committed fixtures its gates compare
      BUILD.bazel              # exports the package label
  include/nanochat/
    capi.h  config.h  tensor.h  kernels.h  model.h  optim.h  data.h  sandbox.h
    tokenizer.h  dataloader.h  rand.h  sampler.h  scheduler.h  logger.h  mfu.h
    model_spec.h               # the description as data, and ModelSpec -> Config
    plan.h                     # the plan structure and the pure BuildPlan
    fields.h  checkpoint_metadata.h  # the metadata block; host-only, vendor-free
  src/
    ops.cc  model.cc  generate.cc  optim.cc  train.cc  data.cc  eval.cc
    model_spec.cc              # the spec reader, the validator, the derivation
    plan.cc  plan_main.cc      # the planner, and the probe `doctor` prints
    capability_names.h/.cc     # the vocabulary name to kFeature* bit table
    device_profile.cc          # derives the device table from backends/*/caps.json
    fields.cc  checkpoint_metadata.cc  # the visitor and the metadata block
    fields_test.cc                # the ported visitor tests
    *_main.cc
    parquet/                   # the native parquet reader (docs/parquet-native.md)
      thrift.{h,cc}  format.{h,cc}  levels.{h,cc}  reader.{h,cc}
      *_test.cc  testdata/text.parquet
    tokenizer/                 # the tokenizer host package (see src/tokenizer/)
      tokenizer.cc  split_pattern.{h,cc}  utf8.{h,cc}  unicode_tables.inc
      bpe_trainer.cc
      tok_train_main.cc
      *_test.cc
  backends/
    cpu/kernels.cc  cpu/kernels_test.cc
    cpu/caps.json              # the reference device declaration
    cuda/caps.json             # the declaration of every device this backend serves
    cuda/fused_overrides.cc    # the fused families, registered as overrides
    cuda/device.cu  cuda/gemm.cu  cuda/cuda_runtime_test.cc
    cuda/row_reduce.cu  cuda/span_reduce.cu  # the two primitive reductions
    cuda/kernels/{rms_norm,qk_prep,attention,pointwise,classifier,
                  embedding,adamw,muon,grad_norm,value_gate}.cu
    cuda/kernels/<family>_test.cc + <family>_benchmark.cc next to each kernel
    cuda/kernels/device_utils.cuh
    cuda/kernels/testing/        # host reference headers and test helpers
  dev/kernels/                   # unpromoted prototypes and the toolchain spike
  examples/                      # the smallest examples of the public API
    hello_world/                 # inference: config, model, generate (docs/examples.md)
    checkpoint_roundtrip/        # the container: save, load, metadata (docs/examples.md)
  bindings/                      # the C ABI for the in-process Python API
    nanochat_capi.cc  nanochat_capi_test.cc  BUILD.bazel
  python/
    nanochat_cpp/                # the single Python surface
      __init__.py  _core.py  _lib.py  _build.py  api.py
      plan.py  chat.py  toolchain.py  _entry.py  checkpoint.py
      data.py  tasks.py  eval_fixture.py  reference.py
      cli.py                     # the tools/nanochat command line
      prune.py                   # the Bazel output-base reaper
      gpu_queue.py               # the GPU job queue (worker + client)
    tests/{core,build,api,plan,chat,toolchain,data,checkpoint,eval_fixture}_test.py
    tests/gpu_queue_test.py      # T0 test for the queue
    tests/prune_test.py          # T0 test for the reaper
    tests/contract_test.py       # T0 test for the host-tool contract
    BUILD.bazel
  tests/oracle_test.cc  tests/data/debug_state.bin
  tools/
    nanochat                   # the single execution entry point
    sandbox/profiles.json      # per-profile cgroup budgets
    sandbox/nanochat.slice     # aggregate user-slice budget (drop-in)
    sandbox/README.md
    dump_fixtures.py           # the offline fixture generator
    fixture_common.py          # the shared fixture helpers
    fixture_format.py          # the shared fixture record writer
    gen_matrix.py              # the section 11 matrix: rendered, and checked
    isolation_gate.py          # the section 10.4 regenerate-and-diff gate
  data/*.parquet
```

`include/nanochat/fields.h` is host-only and vendor-free. It is the field
visitor the checkpoint container's metadata block is built on: plain scalar
code over a `uint32` word stream, with no device, SIMD, or backend header, and
no kernel header includes it. `include/nanochat/checkpoint_metadata.h` sits
beside it and is host-only for the same reason. Neither one appears in
`kernels.h`, and neither one reaches `backends/`.

`spec/` holds prose and data and **no code**. It carries no build target, no
header, and no source file: it is the vocabulary the two axes declare against
and the authority those declarations are reviewed by — version numbers,
capability bits, block kinds, op codes and their accuracy bounds, and the
decision records. `docs/` describes what the tree computes; `spec/` names the
words it says it in, and a name is only ever spelled once. A document that
needs a name links into `spec/` rather than copying the table
([model-and-device-bringup.md](model-and-device-bringup.md) §3 and §17).

`backends/<name>/caps.json` is **the single device declaration**. The profile
table the simulator reads — `include/nanochat/device_profile.h` and its
`src/device_profile.cc` derivation — has no hand-written row: every entry comes
from a `caps.json`, so a device bring-up adds a file under `backends/` and no
row in a header. The same file is what the planner reads, and
`spec/capability_bits.md` is the authority for the bit names it spells.

`models/` is the other axis: one directory per model package, holding the
architecture description, the tolerances, and the fixtures the model's gates
compare. A model bring-up adds a directory and edits no shared file
([model-and-device-bringup.md](model-and-device-bringup.md) §6 and §12.2).
Each package is its own Bazel package, so a diff inside one rebuilds only that
package's targets and its matrix row.

Every example under `examples/` is one package with a `cc_binary`, a `cc_test`
over the same source, and a `test_suite(name = "all")`, so an example cannot
rot. `docs/examples.md` says what each one shows and how long it stays.

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

### Python targets

The in-process Python API (docs/python.md) adds two packages.

- `//bindings` holds the C shim over the C++ graph:
  - `//bindings:nanochat_capi` — the shim as a `cc_library`, for the drift test.
  - `//bindings:nanochat_shared` — the shim as a `linkshared` `cc_binary`.
    This target builds `libnanochat_shared.so`, the library `ctypes` loads.
  - `//bindings:nanochat_capi_test` — the C surface drift test.
- `//python` holds the `ctypes` package and the `py_test` coverage:
  - `//python:nanochat_cpp` — the `py_library`.
  - `//python:core_test`, `//python:build_test`, and `//python:api_test` — the
    three T0 `py_test` targets.

`MODULE.bazel` declares `rules_python` 1.7.0 directly. The package imports
only the Python standard library, so the root module registers the local
`python3` toolchain through `local_runtime_repo` and
`local_runtime_toolchains_repo`. The flag `dev_dependency = True` keeps the
local toolchain out of a downstream module. The host tool requires Python 3.10
or later, and it checks that floor at startup. Build the shared library with
`tools/nanochat build //bindings:nanochat_shared --config=cpu`; the config
selects the CPU backend and fp32.

## Makefile fallback

A single Makefile builds the P0 toolchain spike for an environment where
`rules_cuda` is unavailable. It autodetects nvcc and the architecture in the
spirit of llm.c. It does not build the kernel families.

The CUDA build links `cudart` and cuBLAS. The CPU build links `libm`. Neither
links cuDNN or NCCL.

## Backends and selection

Backend and precision are build and link choices, not runtime switches.

Selection is a build/link choice:

```
--backend=cpu|cuda        link the backend library
--precision=fp32|bf16     select ComputeType
--arch=sm_61|sm_75|sm_80|sm_90  CUDA --generate-code
```

| Backend | Implements | Notes |
|---|---|---|
| `backends/cpu` | all of `kernels.h` with naive loops | reference, CI, oracle baseline; `-lm` (+ OpenMP) |
| `backends/cuda` | all of `kernels.h`; cuBLAS for GEMM | Pascal fp32, Turing fp32, Ampere and newer bf16; no cuDNN or NCCL |

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

`-DNANOCHAT_PRECISION=0` selects fp32, `=1` selects bf16, and `=2` selects fp8.
The fp8 compute type is bf16. `include/nanochat/tensor.h` maps the value to
`BuildPrecision`, so one precision exists per build and no runtime dtype
dispatch occurs. The fp16 build and its half-storage path are removed
([precision.md](precision.md)).

- Pascal (sm_61), Turing (sm_75), and the CPU: fp32.
- Ampere (sm_80) and Hopper (sm_90): bf16.
- The persistent state stays fp32 in every build, so the parameter gradient is
  always `GradientType = float` even when the compute type is bf16
  ([precision.md](precision.md) section 5).
- The backend reports supported dtypes through `Caps::Supports`. A bf16 build on
  a device without bf16 fails fast: `Model::Create` returns null with
  "compute dtype is not supported by <device>".

`--config=bf16` selects the define. `--config=a100`, `--config=h100`, and
`--config=h200` are the complete Ampere and Hopper targets; each selects the
matching architecture, `--config=bf16`, and the CUDA backend.

## Turing (T4, sm_75)

Design for the Turing target. The T4 card is the second GPU family that
`nanochat.cpp` supports. Read the backends and precision sections above for the
selection rules.

Status: partially implemented. The build configs exist, but the Turing
correctness gates have no recorded result yet. The `t4` build uses fp32 for the
compute, so each parameter keeps one fp32 weight and the optimizer updates that
weight in place. The build needs no master-weight mirror and no loss scaling.
See the precision section above and [precision.md](precision.md).

### 1. Goal

Turing is a supported target. A Turing port uses the existing backend seam. No
new backend directory is necessary.

### 2. Capability difference

| Feature | Pascal sm_61 | Turing sm_75 |
|---|---|---|
| fp32 arithmetic | yes | yes |
| `mma` instructions | no | yes |

`GetCaps()` reports the differences. `caps.has_tensor_cores` is true for
`major >= 7`. Turing has fp16 tensor cores, but no build uses them
([precision.md](precision.md)); the T4 runs fp32.

### 3. Build configuration

| Config | Arch | Precision | Use |
|---|---|---|---|
| `sm_75` | sm_75 | fp32 | fp32 correctness gate |
| `t4` | sm_75 | fp32 | default Kaggle build |
| `bf16` | default | bf16 | bf16 compute for Ampere and newer |
| `a100` | sm_80 | bf16 | first Ampere target |
| `h100`, `h200` | sm_90 | bf16 | Hopper target and compile gate |

The `t4` config includes `--config=cuda`. The `t4` build uses the fp32 GEMM
path on this card.

The T4 build uses fp32. The fp32 build is the correctness gate and the default
Kaggle build. The Turing hardware supports fp16, but no build uses it
([precision.md](precision.md)). Stage 1 removed the separate `t4-fp32` config.

### 4. GEMM behavior

`backends/cuda/gemm.cu` uses fp32 accumulation on every device
(`CUBLAS_COMPUTE_32F`). The `t4` build uses the classic `cublasSgemm` path for a
single GEMM. A bf16 build uses `cublasGemmEx` with `CUDA_R_16BF` operands. A
forward or dgrad GEMM writes a bf16 result; `GemmWgrad` writes the fp32
parameter gradient.

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
| T1 | `//backends/cuda/kernels:precision_gpu_test` | fp32 |
| T1 | `//backends/cuda:cuda_runtime_gpu_test` | fp32 |
| T2 | `//tests:oracle_cuda_test` | fp32 |
| T2 | `//tests:train_parity_cuda_test` | fp32 |
| T3 | `//backends/cuda/kernels:attention_benchmark` | fp32 |

The CUDA `precision_gpu_test` target builds at the fp32 precision only. The T4
gate runs the fp32 build with the recorded fp32 tolerance. Measure the worst
error again and record the new value in `precision_test.cc`. Do not widen the
tolerance without a measurement.

The T2 targets compare against committed fp32 PyTorch goldens, so they run only
in the fp32 build. The bf16 gates are listed in
[precision.md](precision.md) section 6.

Extend `//backends/cuda:cuda_runtime_gpu_test` to assert two things on sm_75:
the device name and `has_tensor_cores`.

### 7. Two cards

A Kaggle Notebook gives two T4 cards. See
[host-portability.md](host-portability.md) for device selection and
[host-portability.md §8](host-portability.md) for the two options. The short form:

- Option A: two independent jobs, one per card.
- Option B: data-parallel training. Later work.

### 8. Definition of done

1. `tools/nanochat build --config=t4` succeeds.
2. Every T1 GPU test passes on the T4 in fp32.
3. The oracle and train-parity gates pass on sm_75.
4. The recorded fp32 tolerance matches a fresh measurement.
5. `cuda_runtime_gpu_test` asserts the Turing capabilities.
6. [performance.md](performance.md) holds one T4 row at the training shape.
