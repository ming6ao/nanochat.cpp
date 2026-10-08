# Simulator for Hopper-class targets

This document plans a simulator for the API calls and the numerics of
`nanochat.cpp` on H100, H200, and later accelerator targets. The development
host has no Hopper device, so the simulator must work without one.

Status: implemented for the S0 tier. The tree holds the device profile table,
the reference engine, the emulation engine, the API interposer, and the
collective mock. The `tools/nanochat simulate` command runs them. Tests cover
each part. The sections below are the plan; each carries the status note where
it matters. See "Status by section" at the end of this file.

The host engine in section 9 is the strongest of the three. It runs the
unchanged `backends/cuda/kernels/*.cu` bodies. So it checks the arithmetic
that the reference engine cannot.

## 1. Purpose

The project supports Pascal `sm_61` and Turing `sm_75` today. Hopper appears
only in the peak-FLOPS table of `src/train.cc` and as a design reference in
`DESIGN.md`.

The simulator has two jobs:

1. **Numerical correctness.** Run the model, the backward pass, the optimizer,
   and the correctness suites against a Hopper-class capability profile.
2. **API behavior.** Check the runtime, cuBLAS, and collective calls against
   the same profile.

The numerical job uses the CPU reference backend, in the same way that PyTorch
uses its CPU backend as a portable reference. The reference backend computes
real values. The device profile changes the reported capabilities, so the host
logic sees a Hopper device.

The simulator covers three API families:

1. The CUDA runtime: memory, copies, device queries, and launches.
2. The cuBLAS GEMM entry points.
3. The gradient-sync collective path.

A third engine runs the real device code on the host. Section 9 defines it.
The emulation engine is in scope.

## 2. Evidence from the current tree

These facts shape the plan. I verified each one on the development host.

- The kernel seam lives at `include/nanochat/kernels.h`. The project keeps it
  frozen and vendor-free.
- `backends/cpu/kernels.cc` implements every symbol of the seam as a reference.
  It passes the oracle and the training-parity suites today.
- `backends/cuda/device.cu` calls the runtime. `backends/cuda/gemm.cu` calls
  cuBLAS. Every kernel family launches through `cuda_backend::Launch` in
  `backends/cuda/device.h`.
- `GetCaps()` in `backends/cuda/device.cu` maps `cudaDeviceProp` to `Caps`.
  `GetCaps()` in `backends/cpu/kernels.cc` reports the CPU reference.
- The tensor-core rule is `major >= 7`. The fp16 rule is `sm_53` or newer.
  Hopper reports `major == 9`, so both rules already include it.
- `backends/cuda/gemm.cu` uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and the
  `cublasGemmAlgo_t` type. Both are legacy since CUDA 11.0. The fp32 path uses
  `CUBLAS_GEMM_DEFAULT` and `cublasSgemm`, so no algorithm enum reaches Hopper.
- The attention and value-gate kernels use `extern __shared__`. No launch in the
  tree exceeds 48 KB. The fused attention forward uses 37,120 bytes at
  `head_dim == 128`. No file calls `cudaFuncSetAttribute`.
- Host nvcc is CUDA 12.0. Both `-arch=sm_90` and `-arch=sm_90a` compile.
- The CUDA runtime and cuBLAS link dynamically. I confirmed this with `readelf`
  on `bazel-bin/backends/cuda/cuda_runtime_gpu_test`. The binary has
  shared-library dependencies on `libcudart.so.12` and `libcublas.so.12`.
- The undefined symbols carry version tags. Examples are
  `cublasCreate_v2@libcublas.so.12` and
  `cudaGetDeviceProperties_v2@libcudart.so.12`. Each device translation unit
  also references `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`, and
  `__cudaUnregisterFatBinary`.
- The production CUDA backend uses no stream, event, or graph symbol. The host
  `Stream()` helper returns a null handle. Only the prototype
  `dev/kernels/decode_fused_bench.cc` and the test scaffold
  `backends/cuda/kernels/testing/bench_utils.h` use streams and events.
- The gradient-sync seam already exists at `src/distributed.h`. The class is
  `GradientSync`, and the method is `AllReduceSum`. The host reference lives in
  `src/distributed.cc`. The plan for NCCL is `backends/cuda/nccl_sync.cu`
  (docs/distributed-design.md sections 4 and 5).
- The correctness suites already exist: `//tests:oracle_test`,
  `//tests:train_parity_test`, and the finite-difference kernel tests.

The dynamic linkage and the version tags make a preloaded interposer possible.
The CPU reference makes the numerical engine possible.

## 3. Simulator structure

The simulator has three engines. The first two are primary.

| Engine | Job | Numerics | Device needed | Tier |
|---|---|---|---|---|
| Reference engine | Run the graph and the correctness suites | Yes | No | S0 |
| API interposer | Validate the runtime and cuBLAS calls | No | No | S0 |
| Emulation engine | Run the real device code on the host | Yes | No | S0 |

A shared device profile table gives every engine the target capability data.

- The reference engine reports the profile and computes on the CPU reference.
- The API interposer reports the profile and records the real API calls.
- The emulation engine runs the compiled device kernels on the host.

Tier S0 is new. It is a CPU-only tier that exercises Hopper-class behavior but
never touches the GPU. Section 10 defines it.

## 4. Part 1: reference engine

### 4.1 Mechanism

The reference engine reuses the CPU reference backend. A new build config
`--config=sim` selects that backend and defines `NANOCHAT_SIMULATOR`.

`GetCaps()` in `backends/cpu/kernels.cc` then reads `NANOCHAT_SIM_PROFILE` and
returns `CapsFromProfile`. Every other line of the reference backend stays the
same. The host graph therefore sees a Hopper-class device, and the compute
still runs on the CPU.

```bash
# The oracle and the training trajectory on a simulated H100, fp32.
tools/nanochat simulate --device h100 --precision fp32 --suite correctness

# The same on an H200. A numerical run defaults to fp32: the committed fixtures
# are the fp32 CPU case, so the fp16 correctness suite diverges from the fp32
# oracle fixture for reasons that predate the simulator (see docs/testing.md).
# The half-storage path is still covered, by //tests:sim_numerics_test in the
# fp16 build and by `--device h200 --mode compile`.
tools/nanochat simulate --device h200 --precision fp32 --suite correctness
```

### 4.2 What the reference engine validates

- The model, the backward pass, and the optimizer under the profile caps.
- The oracle fixture within the recorded tolerance.
- The training-parity trajectory within the recorded tolerance.
- The finite-difference gradient check for every kernel family.
- The capability gates: the precision gate, the workspace sizing, and the MFU
  computation.
- The fp16 storage path, when the build selects fp16 (see the precision note in
section 4.1's command list).

The precision matters. The CPU backend compiles with `ComputeType = Fp16` under
`--config=fp16`. It then rounds storage to half and computes in float. So the
simulator exercises half-storage numerics for the fp16 profile.

### 4.3 What the reference engine does not validate

The reference engine does not test CUDA kernel arithmetic. The CPU loops are
not the device loops. The device-code emulation engine in section 9 covers
that case.

The reference engine also does not test tensor-core accumulation. The CPU
reference always accumulates in float. Only real hardware or the emulation
engine can test that.

### 4.4 Invariants

- The reference engine must produce the same numerics with and without the
  profile override. A test asserts this equality. The profile changes the caps,
  not the math.
- The tolerance stays the CPU tolerance from docs/testing.md. The profile does
  not change the tolerance.

## 5. Part 2: API interposer

### 5.1 Mechanism

The interposer is a shared library that exports the CUDA runtime and cuBLAS
symbols. The `tools/nanochat simulate` command sets `LD_PRELOAD` to that
library. The dynamic linker then binds the calls from `device.cu`, `gemm.cu`,
and every launch stub to the interposer.

The interposer serves host memory from `malloc` and `free`. It maps
`cudaMemcpy` to `memcpy` and `cudaMemset` to `memset`. It does not call the real
CUDA runtime. The real `libcudart.so.12` may stay loaded, because the binary
has a `NEEDED` entry for it. That is safe, because no call reaches it.

A version script is optional for binding. The dynamic linker resolves a
versioned reference to a preloaded definition in the normal way. The project
still uses a version script, because the script pins the exported set for the
symbol gate. The gate reads the exact version nodes from `readelf -V`. The
current nodes are `libcudart.so.12` and `libcublas.so.12`.

The kernel launch stub is compiler-generated. A preprocessor macro cannot
redirect it. The interposer therefore intercepts the runtime launch call and
the fat-binary hooks at load time.

Two handle contracts are load-bearing. The mock `cublasCreate_v2` must write a
non-null handle into its output argument, because `gemm.cu` creates the handle
only when the stored handle is null. The mock `__cudaRegisterFatBinary` must
return a non-null token, because the registration hooks pass it back.

### 5.2 Symbol coverage

The symbol gate compares the mock exports against the exact binary under test.
The reference binary is `//backends/cuda:cuda_runtime_gpu_test`, which links the
CUDA backend; the preload test `//tests:cuda_sim_preload_test` links the same
backend and is the binary `LD_PRELOAD` actually runs. The gate requires the mock
to export every undefined CUDA symbol of a binary. The mock may export more.
(`//tests:cuda_sim_test` links the mock's implementation directly and is an
in-process unit test of the mock, not a binding test.)

| Family | Symbols | Production use |
|---|---|---|
| Memory | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset` | yes |
| Device | `cudaGetDevice`, `cudaGetDeviceProperties_v2`, `cudaGetErrorString`, `cudaGetLastError`, `cudaDeviceSynchronize` | yes |
| Launch | `cudaLaunchKernel`, `cudaFuncSetAttribute`, `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`, `__cudaUnregisterFatBinary`, `__cudaRegisterFunction` | yes |
| cuBLAS | `cublasCreate_v2`, `cublasSgemm_v2`, `cublasGemmStridedBatchedEx`, `cublasGetStatusString` | yes |
| Streams | `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamBeginCapture`, `cudaStreamEndCapture` | reserved |
| Events | `cudaEventCreate`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime` | reserved |
| Graphs | `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy` | reserved |
| cuBLAS | `cublasDestroy`, `cublasSetStream` | reserved |

`cudaFuncSetAttribute` is in the table with a production use of "yes" for the
rule, not for the current tree. No current call uses it. Section 12.3 gives the
synthetic trigger.

### 5.3 Validation rules

The interposer records each call and checks it against the device profile and
the CUDA API rules.

| Check | Rule |
|---|---|
| Binding canary | The first log record names the simulated profile. The record proves the mock ran. |
| Container bounds | Every pointer range is inside a live allocation. |
| Launch limits | `blockDim` is within the profile limit. Grid dimensions are positive. |
| Shared memory | The dynamic size is within the profile limit. A size above 48 KB needs the device attribute. |
| Attribute order | A large-shared-memory launch must follow the matching `cudaFuncSetAttribute` call. |
| cuBLAS compute type | fp16 on a tensor-core device uses `CUBLAS_COMPUTE_32F`. A Pascal-only path must not appear. |
| Legacy cuBLAS | The interposer reports `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and `cublasGemmAlgo_t` as legacy. |
| Transpose and leading dimension | The `lda`, `ldb`, and `ldc` values are consistent with the transpose flags. |
| Handle order | `cublasCreate` runs before any cuBLAS call. A destroyed handle is not reused. |
| Copy direction | The kind matches the source and destination memory spaces. |

The interposer writes one JSON record per call. The record holds the call name,
the arguments, the verdict, and the profile. The test asserts named semantic
invariants, not byte equality with a committed log. The test omits or
normalizes pointer values. Section 12.2 lists the invariants.

### 5.4 API scope

The interposer does not compute. It records and validates each kernel launch,
then returns. The numerical results come from the reference engine.

The API mode and the numerical mode are separate runs. The API mode proves API
conformance. The reference engine proves numerics. Together they cover both
jobs. The emulation engine in section 9 unifies them for the kernel-level case.

The numerical CUDA tests `//backends/cuda:cuda_runtime_gpu_test`,
`//tests:oracle_cuda_test`, and `//tests:train_parity_cuda_test` do not run
under the interposer. They compare values, and the API-only interposer cannot
satisfy them. The reference engine covers their numerics on the CPU.

## 6. Part 3: collective path

### 6.1 The existing seam

The plan does not add a second seam. The seam already exists at
`src/distributed.h`:

- `DistributedConfig` holds `rank`, `world_size`, `master`, and `port`.
- `GradientSync` has `rank()`, `world_size()`, and
  `AllReduceSum(ComputeType* buffer, std::int64_t count)`.
- `CreateGradientSync(config)` builds the sync. A `world_size` of 1 returns a
  no-op object.

`src/distributed.cc` holds the host reference. The NCCL implementation is
planned for `backends/cuda/nccl_sync.cu`, and it calls `ncclAllReduce` with
`ncclSum`. The test is `//backends/cuda:nccl_sync_test`. See
docs/distributed-design.md sections 4 and 5.

### 6.2 Collective numerics

The host reference in `src/distributed.cc` computes the real sum. It runs on
the CPU with no vendor library. The simulator therefore tests the collective
numerics directly:

- Run `N` ranks of the host reference.
- Each rank holds a distinct buffer.
- Assert that every rank ends with the elementwise sum.
- Assert the result against an independently computed reference.

This extends the existing `//src:distributed_test` to the simulated profiles.
The numerical correctness of the seam is already real, not mocked.

### 6.3 Collective API coverage

The interposer mocks the collective library with the same method as part 1. The
NCCL symbols live in weak definitions, so the mock loads without a vendor
library.

| Symbols | Purpose |
|---|---|
| `ncclGetVersion`, `ncclGetErrorString` | Version and error text |
| `ncclGetUniqueId` | Communicator identity |
| `ncclCommInitRank`, `ncclCommInitAll`, `ncclCommDestroy`, `ncclCommCount`, `ncclCommUserRank` | Communicator lifecycle |
| `ncclAllReduce` | The only collective the seam needs today |
| `ncclBroadcast`, `ncclAllGather`, `ncclReduceScatter`, `ncclReduce` | Reserved for later |
| `ncclSend`, `ncclRecv`, `ncclGroupStart`, `ncclGroupEnd` | Reserved for later |

### 6.4 Multi-rank model

The simulator runs `N` ranks as separate processes. This matches the existing
`src/distributed_test.cc`, which forks processes over loopback TCP. Each rank
links the same NCCL mock. The environment variables `NANOCHAT_SIM_RANK` and
`NANOCHAT_SIM_WORLD` identify the rank and the world size. Every rank and the
coordinator run through `tools/nanochat run t0-cpu -- ...`.

Each rank gets its own simulated device from the profile table. The mock appends
one JSON record per call to the shared `NANOCHAT_SIM_LOG`. A coordinator reads
the merged log back (`ReadCollectiveLog`) and checks the cross-rank rules.

| Check | Rule |
|---|---|
| Rank consistency | `rank < world_size` on every rank. |
| Communicator matching | All ranks use the same communicator identity. |
| Collective matching | Count, data type, operation, and root agree across ranks. |
| Group nesting | `ncclGroupStart` and `ncclGroupEnd` are balanced. |
| Buffer range | `count` times the data-type size fits in the rendezvous buffer. |
| Deadlock | A rank that waits on a collective fails when no peer issues the same call. |

The mock does not model stream *ordering* (it has no stream state; the seam uses
no streams), so there is no ordering rule. The multi-rank model validates the
wire protocol that the real pattern must follow.

## 7. Part 4: device profile table

The profile table holds one entry per target. Each entry records the name, the
compute capability, the memory size, the tensor-core flag, the fp16 and bf16
flags, and the peak rates.

Path: `include/nanochat/device_profile.h` and `src/device_profile.cc`.
The header is additive and architect-owned.

The table stays **beside** the MFU table in `src/train.cc`, by operator
decision. The two tables do not merge.

The consistency test is objective:

- The join key is the lowercased device name.
- The test matches a profile name against the `PeakFlopsForDevice` patterns.
- The test asserts equal fp32 and fp16 peak rates for every shared entry.
- The test asserts that every profile name resolves to one MFU entry.

`PeakFlopsForDevice` stays the authority for the MFU string patterns. The
profile table is the authority for the capability fields.

The bf16 field is device metadata only. No dtype in the tree consumes it.
The profile records the TF32 gap: the fp32 path uses `CUBLAS_COMPUTE_32F` and
`cublasSgemm`, and it never calls `cublasSetMathMode`. So the fp32 reference
tests true fp32, not TF32.

The profile table also feeds a `doctor` line that reports the simulated
profile.

## 8. Part 5: compile gate

The compile gate is secondary. It needs no interposer and no GPU.

- New `.bazelrc` configs: `h100` (`sm_90`, fp32, cuda) and `h200` (`sm_90`,
  fp16, cuda).
- `tools/nanochat simulate --device h100 --mode compile` builds
  `//backends/cuda:all_kernels` and the host graph for that architecture.
- The gate holds the build lock. It takes no GPU broker.
- The gate enables `-Werror` through `--config=sim-gate`. `nanochat.bzl` adds
  `-Xcompiler -Werror` and the nvcc `-Werror all-warnings` class, and it
  excludes the deprecation warning for the legacy cuBLAS enums through
  `-Xcompiler -Wno-deprecated-declarations`.
- The gate records the `ptxas` register and shared-memory use per kernel in
  `/tmp/nanochat-sim-ptxas-<device>.log`. Set `NANOCHAT_SIM_PTXAS_LOG` to
  change the path. A cached build compiles nothing, so the gate keeps the
  previous log instead of erasing it.
- The gate leaves `-Werror` out of the inner loop. Only `--config=sim-gate`
  turns it on.

The gate does not add an `sm_90a` config. The tree uses no `sm_90a` feature,
so the config can only fail. The gate does not add an `a100` config. That
device is outside the stated scope.

## 9. Part 6: device-code emulation engine

Status: implemented for RmsNorm and Pointwise.

The reference engine tests the graph, and the interposer tests the API. Neither
tests the arithmetic of the `.cu` kernels. The emulation engine closes that gap.

### 9.1 Mechanism

- A host execution header maps the CUDA subset to C++
  (`tools/cuda_sim/emu/cuda_emu.h`, with the stub `<cuda_runtime.h>`,
  `<cuda_fp16.h>`, and `<math_constants.h>` beside it).
- The build compiles the `backends/cuda/kernels/*.cu` sources with the host
  compiler and the header (`//tools/cuda_sim/emu:emulated_kernels`). A genrule
  copies each `.cu` to a `.cc` name, because Bazel's C++ rules do not accept a
  `.cu` file in `srcs`; the text stays the same.
- The engine intercepts the launch at `cuda_backend::Launch`, not at
  `cudaLaunchKernel`. It force-includes `emu_prelude.h` ahead of
  `backends/cuda/device.h`. That header guards its `<<<...>>>` template with
  `__CUDACC__`, so the template does not compile. The prelude replacement
  iterates the grid and the block, then runs the kernel body on the host.
- Each block runs as a set of cooperative fibers on the launching host thread.
  The scheduler keeps every CUDA thread of the block live at the same time.
  A barrier releases when all live threads arrive. A block that cannot release
  a barrier aborts with a diagnostic instead of hanging. See the
  execution-model comment in `cuda_emu.h` for the divergences from a device.
- `emu_prelude.h` renames the seam entry points with an `Emu` prefix
  (`nanochat::kernels::EmuRmsNormForward` and so on). The emulated device code
  and the CPU reference then link side by side, and `//tests:kernel_emu_test`
  compares them.

### 9.2 Supported subset

The header maps these constructs:

| CUDA construct | Host mapping |
|---|---|
| `__global__`, `__device__`, `__forceinline__` | empty or `inline` |
| `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | plain structs |
| `__syncthreads` | a barrier over the block threads |
| `__shared__` | a per-block host buffer |
| `atomicAdd`, `atomicCAS` | host atomics |
| `__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync` | host shuffle within the warp |
| `__half`, `__half2` | the project `Fp16` type |
| `rsqrtf`, `tanhf`, `expf` | the standard math library |
| `__launch_bounds__` | empty |

Not yet mapped: `extern __shared__` (attention and the ResFormer value gate use
it), texture and surface objects, and `__constant__` arrays. Those families are
outside the first cut.

### 9.3 Acceptance

The emulated kernel must match the CPU reference kernel for the same input,
within the recorded tolerance. This check validates the emulator and the device
code together.

### 9.4 Risk

An emulator can diverge from device semantics. The plan marks this engine as
experimental. It never replaces the oracle. It is a stronger check than the API
mode alone, and a weaker check than real hardware.

## 10. Running the simulator

Tier S0 is new. An S0 run exercises Hopper-class behavior but needs no GPU.
The fast CPU-only S0 tests also ride the default `tools/nanochat test` loop,
because they carry the `sim` tag. The `simulate` command runs them under a
`--config=sim` capability override. For the API suite, it also runs them under
`LD_PRELOAD`.

```bash
# Numerical correctness on a simulated H100, fp32.
tools/nanochat simulate --device h100 --precision fp32 --suite correctness

# API conformance for the CUDA backend.
tools/nanochat simulate --device h100 --suite api

# The NCCL mock and the collective API checks.
tools/nanochat simulate --device h100 --suite collective

# Compile the CUDA backend for the target architecture.
tools/nanochat simulate --device h200 --mode compile

# The optional emulated-kernel check.
tools/nanochat simulate --device h100 --suite kernel
```

The command does this:

1. Builds the needed targets under the build lock.
2. Runs the tests with the `t0-cpu` sandbox profile and no GPU broker.
3. For the correctness suite, adds `--config=sim` and the selected precision.
4. For the API suite, builds `//tools/cuda_sim:cuda_sim_interposer` and
   `//tests:cuda_sim_preload_test` under `--config=cuda`, resolves the mock's
   absolute `.so` path, and runs the preload test with
   `--test_env=LD_PRELOAD=<abs .so>` and
   `--test_env=NANOCHAT_SIM_LOG=<abs temp>`. It also runs
   `tools/cuda_sim/check_symbols.sh` against the mock, against a real
   CUDA-linked binary (`//backends/cuda:cuda_runtime_gpu_test`), and with
   `--negative-self-test`, which rejects a missing symbol.
5. Passes `--test_env=NANOCHAT_SIM_PROFILE=<name>`.

The `.bazelrc` gains `test --test_env=NANOCHAT_SIM_PROFILE`. The `simulate`
command passes `LD_PRELOAD` per invocation, because the path changes with the
build.

The S0 tests carry the `sim` tag. They are CPU-only and fast, so they stay in the
default `tools/nanochat test` loop; the tag lets a merge gate select them by
name. The preload test additionally carries `manual`, because it must only run
under `tools/nanochat simulate --suite api`, which supplies `LD_PRELOAD` and the
log path. docs/testing.md defines the S0 tier.

## 11. Deliverables and owners

The plan needs a new ownership row. `AGENTS.md` section 1 gains a **Simulator**
workstream for `tools/cuda_sim/**`. The entry point `tools/nanochat` moves under
the Architect / Integrator row.

| Deliverable | Path | Owner |
|---|---|---|
| Ownership rows | `AGENTS.md` | Architect |
| Sim config and capability override | `.bazelrc`, `backends/cpu/kernels.cc` | Runtime |
| Reference-engine suite wiring | `tools/nanochat` | Architect |
| Interposer library | `tools/cuda_sim/` | Simulator |
| Version script | `tools/cuda_sim/cuda_sim.map` | Simulator |
| Symbol gate | `tools/cuda_sim/check_symbols.sh` | Simulator |
| Binding canary | `tools/cuda_sim/` | Simulator |
| Collective mock and coordinator | `tools/cuda_sim/collective/` | Simulator |
| Device profile header | `include/nanochat/device_profile.h` | Architect |
| Device profile table | `src/device_profile.cc` | Harness |
| API-log test | `tests/cuda_sim_test.cc` | Oracle |
| Reference-engine test | `tests/sim_numerics_test.cc` | Oracle |
| Collective test | `tests/collective_sim_test.cc` | Oracle |
| Profile and MFU consistency test | `tests/device_profile_test.cc` | Oracle |
| Emulation header | `tools/cuda_sim/emu/cuda_emu.h` | Simulator |
| Emulated-kernel test | `tests/kernel_emu_test.cc` | Oracle |
| Simulate command | `tools/nanochat` | Architect |
| Compile configs and env forwarding | `.bazelrc` | Architect / Build |
| `doctor` profile line | `tools/nanochat` | Architect |
| Design document | `docs/simulator.md` | Architect |
| Documentation index | `docs/README.md` | Architect |
| Design invariants citation | `DESIGN.md` | Architect |
| Testing and portability updates | `docs/testing.md`, `docs/host-portability.md` | Architect |

## 12. Validation of the simulator

### 12.1 Reference engine

1. `//tests:oracle_test` passes under `--config=sim` for fp32 and fp16.
2. `//tests:train_parity_test` passes under `--config=sim` for fp32 and fp16.
3. The finite-difference checks pass under `--config=sim`.
4. The reference engine gives identical values with and without the profile
   override.
5. The model test passes when the profile reports a tensor core and a small
   memory budget, and it fails the precision gate when the budget is too small.

### 12.2 API interposer

6. The symbol gate reads the real binary and the mock. It fails when the mock
   misses a requested symbol. The gate allows extra mock symbols.
7. The binding canary fails when the log holds no header record, or when the
   record names a profile other than `NANOCHAT_SIM_PROFILE`.
8. The API-log test asserts named invariants over the normalized log:
   - The log holds at least one `cudaMalloc`, one kernel launch, and one cuBLAS
     call.
   - Every `cublasGemmStridedBatchedEx` record under a tensor-core profile has
     `compute_type == CUBLAS_COMPUTE_32F`.
   - No launch above 48 KB lacks a preceding `cudaFuncSetAttribute`.
   - No pointer value appears in the assertion.
9. A synthetic interposer test issues a 64 KB launch with no attribute. The
   check must fail.
10. The legacy-enum case builds with `--config=cuda --config=fp16` under a
    simulated tensor-core profile. The interposer then reports the legacy enum.

### 12.3 Collective and kernel

11. The NCCL-mock test (`//tests:collective_sim_test`) runs four ranks, merges
    their `NANOCHAT_SIM_LOG` records, and judges the merged timeline; every rank
    ends with the elementwise sum. `//src:distributed_test` owns the
    gradient-sync *seam* numerics; this test does not.
12. The collective deadlock check fails when one rank skips the all-reduce, and
    the coordinator rejects an unclosed group and an oversized buffer range.
13. The emulated kernel matches the CPU reference kernel within tolerance.
14. The compile gate passes for `sm_90` in fp32 and fp16.

## 13. Phasing

The profile table comes first, because the other engines consume it.

| Phase | Work | Result |
|---|---|---|---|
| 1 | Device profile table and the MFU consistency test | Target capability data |
| 2 | Reference engine: sim config, capability override, numerics suite | Numerical correctness |
| 3 | Emulation engine | Device-code numerics |
| 4 | API interposer for memory, device, and launch | Runtime API conformance |
| 5 | cuBLAS checks and the legacy-enum report | GEMM API conformance |
| 6 | Collective numerics and the multi-rank mock | Collective correctness |
| 7 | Compile gate and the `doctor` line | Architecture and warning gate |
| 8 | Documentation and merge-gate wiring | Repeatable check |

Phases 1, 2, 3, 4, 5, 6, and 8 have code and tests. Phase 7 now works:
`--config=h100` and `--config=h200` exist, and `tools/nanochat simulate
--mode compile` builds the whole CUDA backend for the target architecture.
The gate enables `-Werror` through `--config=sim-gate`. It records the `ptxas`
register and shared-memory use in `/tmp/nanochat-sim-ptxas-<device>.log`.

### Status by section

| Section | Status |
|---|---|
| 4 Reference engine | Implemented (`//tests:sim_numerics_test`, `--config=sim`) |
| 5 API interposer | Implemented (`//tools/cuda_sim`, `//tests:cuda_sim_test`) |
| 6 Collective path | Implemented (`//tools/cuda_sim/collective`, `//tests:collective_sim_test`) |
| 7 Device profile table | Implemented (`//tests:device_profile_test`) |
| 8 Compile gate | Configs and command present; `-Werror` and `ptxas` reporting outstanding |
| 9 Emulation engine | Implemented for RmsNorm and Pointwise |

## 14. Risks and non-goals

- The reference engine does not test device arithmetic. The emulation engine
  covers that case, and it is experimental.
- The reference engine does not test tensor-core accumulation. Only real
  hardware does.
- The interposer must track the toolkit version. A pin and the symbol gate
  control that risk.
- The real CUDA runtime may load beside the mock. No call reaches it, but the
  load order needs one early test.
- The collective dependency waits for architect approval. The plan covers the
  numerics, the mock, and the seam.
- A functional PTX interpreter, such as GPGPU-Sim, is a non-goal. It is a large
  external dependency, and it does not target CUDA 12 and `sm_90`. The
  emulation engine uses the source, not the PTX.
- The S0 tier adds a build path away from the GPU. The merge gate runs it, so
  it cannot hide a compile error.

## 15. Open questions

1. Does the symbol gate use one binary or the union of the S0 binaries?
2. Does a canary header record satisfy the binding proof, or does the test need
   a deeper check?
3. How does the fabricated `cudaDeviceProp` interact with `GetCaps()` and the
   profile table on a host with a real device?
4. Which kernel families does the emulation engine support in the first cut?
