# Build

One Bazel workspace builds every host and every precision. `tools/nanochat`
selects the configuration, applies the sandbox, and runs the command.

```bash
tools/nanochat build                      # the default fp32 build
tools/nanochat build --config=bf16        # the bf16 compute type
tools/nanochat build --config=hopper      # sm_90
tools/nanochat test                       # every non-GPU test
tools/nanochat lint                       # the style gate
```

## Layers

Dependencies point strictly downward. There is exactly one interchangeable
seam, and it is the kernel API at L1.

```text
L6  App / CLI        one binary, nine subcommands        nanochat_cli.cc
L5  Train driver     TrainLoop Checkpointer Logger Mfu   train.cc
L4  Optim            AdamW Muon Scheduler param groups   optim.cc
L3  Model / graphs   Model, Prefill, Decode, KvCache     model.cc generate.cc
L2  Ops (internal)   Linear Block Mlp ValueResidual      ops.cc
L1  KERNEL API       <-- THE BACKEND SEAM          nanochat/kernels.h
L0  Backend          alloc stream launch           backends/cpu/ backends/cuda/
    Vendor           libm / OpenMP | cudart + cuBLAS
```

Host utilities sit beside L2 to L5 and never above the kernel API: `Tokenizer`,
`DataLoader`, `Rand` (xorshift), `Sampler`, `Scheduler`, `Logger`, and `Mfu`.
The seam itself is [kernels.md](kernels.md) § Kernel API sketch.

### Dependency invariants

The build graph enforces these five rules.

1. `src/**` includes only `nanochat/**`. **No vendor header appears outside
   `backends/`, with one exception: `src/parquet/` is the one host package that
   may include the host-only ZSTD header.** The model, the kernels, and the
   training runtime never link DuckDB. Parquet input uses the native reader
   ([data.md](data.md)).
2. `backends/**` includes only `nanochat/kernels.h` and `nanochat/tensor.h`.
   **A kernel knows nothing about the model.** One additive exception exists for
   the simulator: a backend may include `nanochat/device_profile.h`, a leaf
   header that declares only the simulated capability table and itself depends
   only on `nanochat/tensor.h` ([simulator.md](simulator.md), tier S0). That is
   what lets `GetCaps()` answer from a simulated profile. `backends/cpu/` links
   the matching leaf unit `//src:device_profile` for it. The edge points *down*
   to a leaf and not up into the workflow layer. The leaf carries no model code
   and no kernel arithmetic, so the seam stays vendor-free and the reference
   backend still computes every value. `nanochat/mesh.h` is a second leaf of the
   same shape: it declares the local device list and the mesh spec, and it
   depends only on `nanochat/tensor.h`. A backend implements `EnumerateDevices`
   there, so the multi-device runtime can enumerate devices without a change to
   `kernels.h`. A new edge of this shape needs the same review as any other
   change to the layering rule.
3. `kernels.h` contains no vendor type: pointers are `ComputeType*`, streams are
   opaque, and no CUDA header is ever included from it.
4. **No cross naming.** Nothing above the kernel seam names a specific model or
   a specific device, and nothing below it names a graph. The seam names
   neither.
5. **A bring-up adds a directory.** A model bring-up edits only
   `models/<name>/**`; a device bring-up edits only `backends/<name>/**`.
   Neither edits a frozen header, a shared build file, or another package
   without a contract extension first.

The first three rules are the load-bearing invariants: they let a backend be
replaced as a build and link decision rather than as a source change. Stable API
is therefore the default and cheap by construction, and not a promise that the
API cannot move. When it must move, see
[model-and-device-bringup.md](model-and-device-bringup.md) §18. Rules 4 and 5
are the isolation promise. The bring-up contract that makes them operational is
[model-and-device-bringup.md](model-and-device-bringup.md) §4 and §13.

## File layout

```
nanochat.cpp/
  .clang-format
  MODULE.bazel  .bazelrc
  README.md  AGENTS.md  CONTRIBUTING.md
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
    # The one stage binary. `nanochat_cli.cc` is the entry point, `command.h`
    # and `commands.cc` are the subcommand seam, and each `<name>_command.cc`
    # holds one handler. `cli.{h,cc}` is the one flag parser.
    nanochat_cli.cc  command.h  commands.cc  cli.h  cli.cc
    *_command.cc
    plan_main.cc               # the probe `doctor` prints; alone on purpose
    parquet/                   # the native parquet reader (docs/data.md)
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
`tools/nanochat build //bindings:nanochat_shared`; `.bazelrc` already selects
the CPU backend at fp32, so no named config is necessary.

### Python environments

The host tools and the `py_test` targets need the standard library and the
system `python3` only. Keep that interpreter free of third-party packages.

A task that needs an extra installs it in one environment outside the tree.
Then no worktree carries a second copy. The command names an extra of
[build.md](build.md) Part 1, section 6.2, not a
package list:

```bash
uv venv --python 3.12 ~/.venvs/nanochat-cpp
uv pip install --python ~/.venvs/nanochat-cpp/bin/python -e "python[modal,hub]"
```

Do not add an extra to the reference environment. `tools/nanochat parity
produce` resolves the reference interpreter itself. A producer records that
environment as its provenance.

Do not activate the environment for `tools/nanochat test` or `build`. Bazel
resolves `python3` through `PATH` (`local_runtime_repo` above). An active
environment therefore becomes the test interpreter. Its site-packages can then
hide an import that breaks the layering rule.

## Link dependencies

Bazel is the only supported build. The P0 toolchain spike is
`//dev/kernels:pascal_spike_gpu_test`.

The CUDA build links `cudart` and cuBLAS. The CPU build links `libm`. Neither
links cuDNN or a collective library: the native ring needs no vendor library.

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
| `backends/cuda` | all of `kernels.h`; cuBLAS for GEMM | Pascal fp32, Turing fp32, Ampere and newer bf16; no cuDNN, no vendor collective |

The shared workflow (`ops.cc`, `model.cc`, `generate.cc`, `optim.cc`,
`train.cc`) compiles once and links against either backend. This is the one
improvement over llm.c, which duplicates its top-level files per backend.

### Adding a backend

Adding hardware = add `backends/<x>/` implementing `kernels.h` plus a config
value. `src/` and `include/` normally need no changes; a new kernel family
(below the replacement boundary) does touch the frozen headers, under the
evolution rules in [model-and-device-bringup.md](model-and-device-bringup.md)
§18.

The interface to implement is [kernels.md](kernels.md). The dependency rules a
backend must respect are the
[dependency invariants](#dependency-invariants):
`backends/**` includes only `nanochat/kernels.h` and `nanochat/tensor.h`, and a
kernel knows nothing about the model.

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

`--config=bf16` selects the define. `--config=h100` and `--config=h200` are the
two Hopper targets; each selects sm_90, `--config=bf16`, the CUDA backend, and
the simulator.

### `hopper`, beside `h100`

| Config | Arch | Precision | Backend | Simulator | Collective |
|---|---|---|---|---|---|
| `h100` | sm_90 | bf16 | cuda | on (`--define=simulator=1`) | `native` |
| `hopper` | sm_90 | bf16 | cuda | off | `native` (the only collective) |

The two configs are not the same build. `h100` is the *simulator* profile: it
compiles in the capability override and answers from the device profile table
instead of from the card. `hopper` is a real Hopper build for a hand run at
bf16, with the simulator off. The native ring is the one collective, so both
configs link it and the chain's `world_size=1` reduces to a no-op
([post-training.md](post-training.md) Part 3, section 3, trap 1). A
check that needs a card chooses `hopper`; a report read without a card chooses
`h100`.

`hopper` is a hand-run shorthand only. `_build._build_command` still emits its
explicit flag triple, `--config=cuda`, `--define=precision=<p>`, and
`--@rules_cuda//cuda:archs=<a>`, and never a named config, so the chain's H100
leg, the build test, and a hand run are the same build.

## Turing (T4, sm_75)

Design for the Turing target. The T4 card is the second GPU family that
`nanochat.cpp` supports. Read the backends and precision sections above for the
selection rules.

The build configs exist for Turing, but the
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
| `bf16` | default | bf16 | bf16 compute for Ampere and newer |
| `h100`, `h200` | sm_90 | bf16 | Hopper simulator profile and compile gate |
| `hopper` | sm_90 | bf16 | real Hopper build for a hand run (the Precision section) |

The tree keeps no named Turing config. A Turing run passes
`--config=cuda --@rules_cuda//cuda:archs=sm_75` itself, which is the flag triple
`tools/modal/app.py` emits for its T4 leg. The Turing build uses the fp32 GEMM
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

The current attention kernel uses fp32 statistics and an online softmax. It does
not use tensor cores. A `mma` variant is later work under
[kernels.md](kernels.md) § Compute strategy. A candidate belongs in
`dev/kernels/` first, and it must beat the current kernel at the training shape.

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

## Part 1. Dependency layering and host packaging

_Absorbed from the former `dependency-layering-plan.md`. Its section numbers
are unchanged, so a reference to a section of that document resolves to the
same number here._

### 1. Goals and non-goals

#### Goals

- Keep `pip install nanochat-cpp` free of third-party packages.
- Make a broken dependency boundary fail a test, not a review.
- Drive a notebook, a Modal container, and a test gate from one package.
- Compare with the PyTorch reference without a torch install at gate time.
- Install the package at image build, and the weights at run start.

#### Non-goals

- Adding numpy to the core. The current `array` and `ctypes` code is why the
  wheel stays pure.
- Moving a source file under `src/`, `include/`, or `backends/`.
- A language server, a formatter, or a packaging rewrite.
- A published binary wheel for every platform. The first release covers sm_75
  and sm_61 only.

### 2. The defect

The C++ core is already lean. The
[dependency invariants](#dependency-invariants)
enforce the layering, and `backends/**` links no vendor library beyond CUDA. The
leak is in the Python tree.

#### 2.1 Leak one: lazy third-party imports in core modules

A lazy import hides a dependency from every tool. The import succeeds in the
test loop only because the module name resolves late.

| File | Line | Import | Correct zone |
|---|---|---|---|
| `python/nanochat_cpp/checkpoint.py` | 943 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 962 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 986 | `torch` | Z2 |
| `python/nanochat_cpp/checkpoint.py` | 1005 | `torch` | Z2 |
| `python/nanochat_cpp/data.py` | 42 | `nanochat.dataset` | Z2 |
| `python/nanochat_cpp/data.py` | 163 | `tiktoken` | Z2 |
| `python/nanochat_cpp/tasks.py` | 65 | `numpy` | Z2 |
| `python/nanochat_cpp/tasks.py` | 409 | `nanochat.execution` | Z2 |

#### 2.2 Leak two: no package description

The tree has no `pyproject.toml`. A reader cannot tell which packages the base
install needs. A host cannot install the package without a repository clone.
`tools/modal/app.py` and `notebooks/kaggle_setup.py` both clone and build the
tree for that reason.

#### 2.3 Leak three: an eager package import

`python/nanochat_cpp/__init__.py` line 18 imports thirteen modules. A caller
that needs only `api` pays for `chat`, `rl`, `sft`, `sft_data`, and `toolchain`.
The cost is small today. It grows when `Z2` re-exports arrive.

#### 2.4 Leak four: no provenance in a fixture

`models/nanochat-default/fixtures/manifest.json` records a hash, a size, and a
record count. It does not record the torch version or the reference commit that
produced the bytes. A parity claim in [parity.md](parity.md) then rests on an
environment that the manifest cannot name.

### 3. Decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | How many import packages? | Two: `nanochat_cpp` and `nanochat_cpp_bridge`. |
| D2 | How many distributions? | One. The base list is empty and the extras carry the rest. |
| D3 | Where is the boundary tested? | In the test loop, by five gates. See section 8. |
| D4 | Which module moves first? | `checkpoint.py`. It has the largest torch surface. |
| D5 | Does the bridge live under `python/`? | Yes. One `py_library`, one test area, one owner. |
| D6 | How does a wheel carry the shared library? | A per-architecture wheel, `nanochat_cpp_cuda_<arch>`. |
| D7 | Where does the wheel sit in the search order? | New item 5, before the cache and the on-demand build. |
| D8 | How does Modal get the weights? | A volume keyed by the stage revision, not the image. |
| D9 | How does a notebook install? | One `pip install` cell and one environment cell. |
| D10 | How does a parity gate compare? | It reads a committed file, never an import. |
| D11 | Where does the producer code live? | `python/nanochat_cpp_bridge/fixtures/`, run by `tools/nanochat parity`. |
| D12 | Is a public PyPI release in scope? | No. The first channel is a Hugging Face repository. |

The owner decides D1, D2, and D12. D6 and D7 need a decision from the Build
workstream, because a wheel changes `.bazelrc` and the release command.

### 4. The four zones

```text
Z3  Hosts        modal, kaggle, jupyter        tools/**, notebooks/**
Z2  Adapters     torch, numpy, tiktoken,      nanochat_cpp_bridge
                 safetensors, huggingface_hub,
                 datasets, nanochat
Z1  Python core  standard library only        nanochat_cpp
Z0  C++ core     libm, cudart, cuBLAS,        src/**, include/**, backends/**
                 zstd (host only), RDMA
```

The arrow points down. A zone imports a lower zone and never a higher one.

| Zone | May import | Must not import | Counted dependency |
|---|---|---|---|
| Z0 | the C runtime and the vendor runtime | any Python | 3 to 5 libraries |
| Z1 | the standard library, and `Z0` through `ctypes` | anything in `Z2` | 0 packages |
| Z2 | the standard library, `Z1`, and its extras | `Z3` | 1 to 6 packages |
| Z3 | any zone | nothing imports it | per host |

Five rules hold the zones.

1. `python/nanochat_cpp/**` imports only the standard library.
2. `python/nanochat_cpp_bridge/**` imports the standard library, `nanochat_cpp`,
   and its extra packages.
3. `tools/**` and `notebooks/**` import either package.
4. No package imports `tools/` or `notebooks/`.
5. A header under `include/` names no vendor type. This rule already holds.

### 5. Target tree

```text
python/
  pyproject.toml                  # new: base list empty, extras below
  BUILD.bazel                     # one new py_library and one test area
  nanochat_cpp/                   # Z1, same role
    __init__.py                   # lazy __getattr__ re-exports
    api.py chat.py rl.py mesh.py runtime.py plan.py toolchain.py
    data.py                       # the NCTOKEN1 reader stays
    checkpoint.py                 # the container read and write stay
    sft_data.py tasks.py eval_fixture.py
    _core.py _lib.py _build.py
  nanochat_cpp_bridge/            # Z2, new
    __init__.py
    torch_checkpoint.py           # from checkpoint.py lines 915 to 1046
    reference_data.py             # from data.py lines 42 and 163
    reference_tasks.py            # from tasks.py lines 65 and 409
    hf_hub.py                     # the Hub save and load path
    sft_reference.py              # the reference renderer and mixture
    fixtures/                     # the producers. See section 7
```

The model manifests gain one field. `models/<name>/fixtures/manifest.json`
records the producer beside the hash. See section 7.2.

### 6. Packaging

#### 6.1 The base install

```toml
[project]
name = "nanochat-cpp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []          # Z1. This line is the invariant.

[project.optional-dependencies]
hub       = ["huggingface_hub>=0.24", "safetensors>=0.4"]
reference = ["torch>=2.4", "numpy>=1.26", "tiktoken>=0.8"]
nanochat  = ["nanochat @ git+https://github.com/karpathy/nanochat"]
hfdata    = ["datasets>=2.20", "tokenizers>=0.20", "pyarrow>=16"]
modal     = ["modal>=0.64"]
notebook  = ["jupyterlab", "ipykernel"]

[tool.setuptools.packages.find]
include = ["nanochat_cpp*", "nanochat_cpp_bridge*"]
```

The base install carries both import packages. The bridge files are inert
without an extra, because a bridge module imports its heavy package inside the
function that needs it.

#### 6.2 Extras by task

| Task | Command |
|---|---|
| Notebook, evaluate a stage | `pip install "nanochat-cpp[hub]"` |
| Modal image, train | `pip install "nanochat-cpp[hub]"` and a binary wheel |
| Produce a parity fixture | `pip install "nanochat-cpp[reference,nanochat]"` |
| SFT data from the Hub | `pip install "nanochat-cpp[reference,nanochat,hfdata]"` |
| Workstation development | the source tree, plus Bazel |

#### 6.3 The shared library

A pure wheel cannot carry one `.so` for every card. Use one binary wheel per
architecture. The package name states the architecture, as `nanochat_cpp_cuda_sm75`.

`_lib.py` searches in this order after the change:

1. the path in `NANOCHAT_CPP_LIB`;
2. the Bazel runfiles tree;
3. the development tree under `bazel-bin/bindings`;
4. a library next to the package;
5. an installed `nanochat_cpp_cuda_<arch>` wheel;  **new**
6. the cache or an on-demand build.

A wrong architecture fails at `nanochat_init`, with the corrective command.

#### 6.4 Modal

Install the package at image build. Fetch the weights at run start.

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.0-devel-ubuntu22.04")
    .pip_install("nanochat-cpp[hub] @ https://.../nanochat_cpp_cuda_sm75-0.1.0.whl")
)
runs   = modal.Volume.from_name("nanochat-runs",   create_if_missing=True)
weights = modal.Volume.from_name("nanochat-weights", create_if_missing=True)
```

The image holds the package. The weights volume holds `<repo_id>/<stage>/`, keyed
by the stage revision. A new checkpoint then costs a download, not an image
build.

The current `tools/modal/app.py` bakes the repository and the Bazel build into
the image. Keep that path as a fallback until the wheels exist. Add the weights
volume now, because the image path costs minutes per run today.

#### 6.5 Notebook

```python
%pip install -q "nanochat-cpp[hub]" "https://.../nanochat_cpp_cuda_sm75-0.1.0.whl"
import os; os.environ.setdefault("NANOCHAT_SANDBOX_BACKEND", "none")
```

The cell does no clone and no build. The logic stays in `notebooks/session.py`
and `notebooks/target.py`. A cell holds no logic.

### 7. The parity seam

#### 7.1 The rule

A comparison with the PyTorch world is a file, not an import.

```text
      producer (Z2, offline)              consumer (Z1, no torch)
torch nanochat  -->  fixtures/*.bin  -->  tests/
HF datasets     -->  packed rows     -->  nanochat_cli sft
```

The producer runs in a virtual environment with the reference checkout. The
consumer runs in the T0 loop with no third-party package. The tree already
follows this pattern for `debug_state.bin`, `train_parity.bin`,
`tokenizer_fixture.bin`, and `numerics_golden_10l.bin`.

#### 7.2 Provenance

Add two fields to each manifest record.

```json
"producer": {
  "torch": "2.5.1",
  "numpy": "1.26.4",
  "nanochat_commit": "<sha>",
  "python": "3.12.4"
},
"generated_by": "nanochat_cpp_bridge.fixtures.train_parity"
```

A gate reads the manifest and fails when a record has no producer. The claim in
[parity.md](parity.md) then names an environment that a reader can rebuild.

#### 7.3 The producer entry point

`tools/nanochat parity produce <kind>` is the only producer command. It resolves
the reference virtual environment and runs the bridge fixture module. No
producer runs under `tools/nanochat test`. The lock and the sandbox therefore
still cover every long job.

#### 7.4 What this unblocks

| Row | Rung | Fixture |
|---|---|---|
| `P4` | SFT data parity | `sft_rows_fixture.bin` |
| `P5` | SFT arithmetic parity | `sft_trajectory.bin` |
| `P7` | SFT validation parity | `sft_val.bin` |

Do not write those fixtures before stage S4 lands. They must land on the
consumer side of the boundary.

### 11. What not to do

- Do not add torch to `Z1` "just for conversion". Conversion is an adapter in
  both directions.
- Do not add numpy to the base list. It breaks the pure wheel.
- Do not add a root `requirements.txt`. It repeats the extras and drifts.
- Do not let a notebook become a gate. A notebook is a host, not a test.
- Do not put reference task logic in `tests/`. It belongs in `Z2`, behind the
  fixture.
- Do not import the bridge from the core to save one function.
