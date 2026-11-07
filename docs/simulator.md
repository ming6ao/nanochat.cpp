# Simulator for Hopper-class targets

The simulator runs the API calls and the numerics of `nanochat.cpp` on H100,
H200, and later accelerator targets. The development host has no Hopper device,
so the simulator works without one: it is the only way to check a Hopper build
on this host.

The tree holds the device profile table, the reference engine, the emulation
engine, and the API interposer, and `tools/nanochat simulate` runs them.

```bash
tools/nanochat simulate --device h100 --suite all
tools/nanochat simulate --device h100 --suite correctness
tools/nanochat simulate --suite api
tools/nanochat simulate --suite compile
```

The host engine in section 9 is the strongest of the three. It runs the
unchanged `backends/cuda/kernels/*.cu` bodies. So it checks the arithmetic
that the reference engine cannot.

## 1. Purpose

The project supports Pascal `sm_61` and Turing `sm_75` today. Hopper appears
only in the peak-FLOPS table of `src/train.cc` and as a design reference in
[kernels.md](kernels.md) § Direct-MMA and persistent kernels.

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
- The tensor-core rule is `major >= 7`. Hopper reports `major == 9`, so the
  rule already includes it.
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
  `src/distributed.cc`, and the one collective is the native ring in
  `backends/cuda/collective/` (docs/distributed.md sections 4 and 5).
- The correctness suites already exist: `//tests:oracle_test`,
  `//tests:train_parity_test`, and the finite-difference kernel tests.

The dynamic linkage and the version tags make a preloaded interposer possible.
The CPU reference makes the numerical engine possible.

## 3. Simulator structure

The simulator has three engines. The first two are primary. The reference
engine is not a separate numeric implementation. It is the CPU backend from
`backends/cpu/` with the `NANOCHAT_SIMULATOR` define. The profile changes the
reported capabilities and nothing else (section 4.4).

| Engine | Job | Numerics | Device needed | Tier |
|---|---|---|---|---|
| Reference engine (CPU backend) | Run the graph and the correctness suites | Yes | No | S0 |
| API interposer | Validate the runtime and cuBLAS calls | No | No | S0 |
| Emulation engine | Run the real device code on the host | Yes | No | S0 |

A shared device profile table gives every engine the target capability data.

- The reference engine is the CPU backend with the capability override. It
  reports the profile and computes on the CPU reference. Only the `GetCaps()`
  branch changes.
- The API interposer reports the profile and records the real API calls.
- The emulation engine runs the compiled device kernels on the host.

Tier S0 is new. It is a CPU-only tier that exercises Hopper-class behavior but
never touches the GPU. Section 10 defines it.

## 4. Part 1: reference engine

4.1 Mechanism

The reference engine reuses the CPU reference backend. There is no second
numeric implementation. A new build config `--config=sim` selects that backend
and defines `NANOCHAT_SIMULATOR`.

`GetCaps()` in `backends/cpu/kernels.cc` then reads `NANOCHAT_SIM_PROFILE` and
returns `CapsFromProfile`. Every other line of the reference backend stays the
same. The host graph therefore sees a Hopper-class device, and the compute
still runs on the CPU.

```bash
# The oracle and the training trajectory on a simulated H100.
tools/nanochat simulate --device h100 --suite correctness

# The same on a simulated T4.
tools/nanochat simulate --device t4 --suite correctness
```

The reference engine runs the fp32 build, so every run matches the committed
fp32 fixtures. The simulated device still reports the bf16 capability of its
profile through `Caps::supports_bf16` ([precision.md](precision.md)).

4.2 What the reference engine validates

- The model, the backward pass, and the optimizer under the profile caps.
- The oracle fixture within the recorded tolerance.
- The training-parity trajectory within the recorded tolerance.
- The finite-difference gradient check for every kernel family.
- The capability gates: the precision gate, the workspace sizing, and the MFU
  computation.

The reference engine runs the fp32 build. The simulated profile changes the
reported caps and never a value ([precision.md](precision.md)).

4.3 What the reference engine does not validate

The reference engine does not test CUDA kernel arithmetic. The CPU loops are
not the device loops. The device-code emulation engine in section 9 covers
that case.

The reference engine also does not test tensor-core accumulation. The CPU
reference always accumulates in float. Only real hardware or the emulation
engine can test that.

4.4 Invariants

- The reference engine must produce the same numerics with and without the
  profile override. A test asserts this equality. The profile changes the caps,
  not the math.
- The tolerance stays the CPU tolerance from docs/testing.md. The profile does
  not change the tolerance.

## 5. Part 2: API interposer

5.1 Mechanism

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

5.2 Symbol coverage

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
| Device | `cudaGetDevice`, `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDeviceProperties_v2`, `cudaGetErrorString`, `cudaGetLastError`, `cudaDeviceSynchronize` | yes |
| Launch | `cudaLaunchKernel`, `cudaFuncSetAttribute`, `__cudaPushCallConfiguration`, `__cudaPopCallConfiguration`, `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`, `__cudaUnregisterFatBinary`, `__cudaRegisterFunction` | yes |
| Cooperative | `cudaLaunchCooperativeKernel`, `cudaDeviceGetAttribute`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | yes |
| cuBLAS | `cublasCreate_v2`, `cublasSgemm_v2`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasGetStatusString` | yes |
| Streams | `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamBeginCapture`, `cudaStreamEndCapture` | reserved |
| Events | `cudaEventCreate`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime` | reserved |
| Graphs | `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy` | reserved |
| cuBLAS | `cublasDestroy`, `cublasSetStream` | reserved |

`cudaFuncSetAttribute` is in the table with a production use of "yes" for the
rule, not for the current tree. No current call uses it. Section 12.3 gives the
synthetic trigger.

`cudaLaunchCooperativeKernel`, `cudaDeviceGetAttribute`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` are the three names the fused
Muon build references (`docs/kernels.md` Part 1 section 8). It references all
three even when it ends up on the phase driver, so the symbol gate requires all
three. On a device that answers 0 for `cudaDevAttrCooperativeLaunch`, a run
reaches only the attribute query: the other two sit inside the cooperative
driver, which the phase driver never enters.

The answers are the design's: **0** for `cudaDevAttrCooperativeLaunch` and
**1** for `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. A simulated run
therefore takes the **phase loop** -- one launch per phase execution, no grid
barrier -- which is also the only driver the emulator's block-at-a-time engine
can run (section 9). The cooperative-kernel stub records the call it was given
(grid, block, and dynamic shared size) instead of doing nothing, so a test can
assert that a driver attempted a cooperative launch, and the fallback that a
refused one takes.

`__cudaPushCallConfiguration` and `__cudaPopCallConfiguration` are not runtime
API calls. nvcc's `<<<>>>` prologue calls them around `cudaLaunchKernel`, so a
translation unit that instantiates a `<<<>>>` launch asks the mock for both.
`backends/cuda/device.h` holds such a launch, and
`backends/cuda/launch_probe.cu` instantiates it, so the gate sees the pair. The
mock keeps the pushed configuration on a per-thread slot and returns it from the
pop, so the launch that follows carries the configuration the program set.

`cublasGemmEx` is the one-batch form of `cublasGemmStridedBatchedEx`, which is
what `backends/cuda/gemm.cu` reaches when a single GEMM's operands are not all
fp32 (the bf16 build; [build.md](build.md), "GEMM behavior"). An fp32 build
links the name but never calls it, so the mock implements both through one
shared body and records the same arguments.

5.3 Validation rules

The interposer records each call and checks it against the device profile and
the CUDA API rules.

| Check | Rule |
|---|---|
| Binding canary | The first log record names the simulated profile. The record proves the mock ran. |
| Container bounds | Every pointer range is inside a live allocation. |
| Launch limits | `blockDim` is within the profile limit. Grid dimensions are positive. |
| Shared memory | The dynamic size is within the profile limit. A size above 48 KB needs the device attribute. |
| Attribute order | A large-shared-memory launch must follow the matching `cudaFuncSetAttribute` call. |
| cuBLAS compute type | A GEMM on a tensor-core device must accumulate with `CUBLAS_COMPUTE_32F`. A Pascal-only compute type must not appear. |
| Legacy cuBLAS | The interposer reports `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and `cublasGemmAlgo_t` as legacy. |
| Transpose and leading dimension | The `lda`, `ldb`, and `ldc` values are consistent with the transpose flags. |
| Handle order | `cublasCreate` runs before any cuBLAS call. A destroyed handle is not reused. |
| Copy direction | The kind matches the source and destination memory spaces. |

The interposer writes one JSON record per call. The record holds the call name,
the arguments, the verdict, and the profile. The test asserts named semantic
invariants, not byte equality with a committed log. The test omits or
normalizes pointer values. Section 12.2 lists the invariants.

5.4 API scope

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

6.1 The seam and its one implementation

The simulator adds no second seam. The seam is `src/distributed.h`:

- `DistributedConfig` holds `rank`, `world_size`, `master`, and `port`.
- `GradientSync` has `rank()`, `world_size()`, and
  `AllReduceSum(ComputeType* buffer, std::int64_t count)`.
- `CreateGradientSync(config)` builds the sync. A `world_size` of 1 returns a
  no-op object.

Two implementations ship. `src/distributed.cc` holds the host reference for the
CPU backend, and `backends/cuda/collective/native_sync.cu` holds the native ring
for the CUDA backend. The native ring is the only collective, so the simulator
needs no collective mock: a vendor library is not in the tree.
See docs/distributed.md sections 4 and 5.

6.2 Collective numerics

The host reference in `src/distributed.cc` computes the real sum, and the ring
schedule in `backends/cuda/collective/ring_schedule.cc` computes the same sum
over a transport. Both run on a central processing unit, so the simulator tests
the collective numerics directly:

- Run `N` ranks over the memory transport.
- Each rank holds a distinct buffer.
- Assert that every rank ends with the elementwise sum.
- Assert the result against an independently computed reference.

This is `//backends/cuda/collective:schedule_test`, which covers world sizes 1,
2, 4, and 8 plus the missing-peer timeout. `//src:distributed_test` covers the
host reference over sockets. Neither needs a card.

6.3 Multi-rank model

A rank is a process, which matches `src/distributed_test.cc`. Every process runs
through `tools/nanochat run t0-cpu -- ...`. The ring test uses the memory
transport, so two ranks share one address space and no card is necessary.

## 7. Part 4: device profile table

The profile table holds one row per declared device. Each row records the name,
the compute capability, the memory size, the capability flags, the three peak
rates, the bandwidth, the launch cost, and the reference rate the MFU line
divides by.

Path: `include/nanochat/device_profile.h` and `src/device_profile.cc`. The
header is additive and architect-owned, and it declares the *shape* of a row and
the lookups over the derived list. It holds no row.

The rows are not maintained in the header. Every row comes from a `devices[]`
entry in `backends/<name>/caps.json`, so a device bring-up adds one entry and no
row: not in the header, not in `src/`, not in a table in a document
([model-and-device-bringup.md](model-and-device-bringup.md) section 6).

The simulator, the reference backend's `GetCaps()`, the API interposer, the
benchmark report, and the roofline estimator all read this one table. A
simulated device is therefore described in exactly one place, and a rate the
simulator reports is the rate the estimator divides by.

### 7.1 One rate table, not two

The tree used to hold a second table: a string-pattern list in `src/train.cc`
that named about thirty cards and one rate each. The two disagreed by design on
a card whose reference recipe counts at tensor cores, so a run could log an MFU
line against one number and be compared with a roofline bound computed from
another.

The declaration now states both numbers, in two fields:

| Field | Meaning |
|---|---|
| `reference_peak_flops` | The rate the reference recipe's MFU line divides by |
| `reference_precision` | The precision that rate is quoted at |

`PeakFlopsForDevice` reads the first field. `PeakFlopsForCompute` reads the
declared rate at the precision it is asked for. Both go through the one
accessor, `PeakFlopsFor(profile, precision)`, so a card has one number per
precision and one reference number. `//tests:device_profile_test` enforces it.

A card with no declaration has no rate. The MFU line then reads 0 and the
benchmark prints that it states no share, which is the honest answer and the one
the header's caption reports.

### 7.2 One answer for an unknown device

A name no declaration owns is refused. `DeviceNameError` renders the message and
`DeviceNameList` gives the declared set, so `tools/nanochat simulate` and
`tools/nanochat roofline` refuse the same name with the same text.

`tools/nanochat devices` prints the declared set, and the simulator validates
against it. That is why the command line holds no device tuple: the accepted set
is the declared set, so a new device needs one `devices[]` entry and no edit to
the tool.

### 7.3 The lookup is deterministic

`FindDeviceProfile` lowercases and strips the whitespace of a name. It looks for
an exact match first, then for a declared name inside the given string, and the
**longest** declared name wins. A vendor spelling such as
`NVIDIA H100 80GB HBM3` therefore reaches `h100` on every build, whatever order
the table is sorted in.

A sub-SKU with its own rated peak needs its own `devices[]` entry. The tree
declares no `h100nvl` row, so an H100 NVL resolves to `h100`.

### 7.4 The consistency test

`//tests:device_profile_test` is a plain CPU test and needs no GPU. It asserts:

- The join key is the lowercased device name, and every declared name resolves.
- `reference_peak_flops` is positive and equals one of the declared rates.
- `reference_precision` is the precision the documented rule selects: bf16 when
  the card has bf16, and fp32 otherwise.
- `PeakFlopsForDevice` reads exactly the declared reference rate.
- Every declared field of a declared device appears in its derived row, and the
  derived table holds exactly one row per declared device.
- A declaration that is not an accelerator contributes no row, because it has no
  launch cost, no HBM rate, and no peak rate.
- An undeclared name has no row, so a device is added by a declaration and never
  by a table edit.

The bf16 field is device metadata only. No dtype in the tree consumes it.
The profile records the TF32 gap: the fp32 path uses `CUBLAS_COMPUTE_32F` and
`cublasSgemm`, and it never calls `cublasSetMathMode`. So the fp32 reference
tests true fp32, not TF32.

The profile table also feeds a `doctor` line that reports the simulated
profile.

## 8. Part 5: compile gate

The compile gate is secondary. It needs no interposer and no GPU.

- New `.bazelrc` configs: `h100` (`sm_90`, fp32, cuda) and `h200` (`sm_90`,
  fp32, cuda).
- `tools/nanochat simulate --device h100 --mode compile` builds
  `//backends/cuda:all_kernels` and the host graph for that architecture.
- The gate holds the build lock. It takes no GPU queue.
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

The engine covers RmsNorm and Pointwise, the fp8 tier (Part 7), the value gate,
attention, and the fused Muon phase driver.

The reference engine tests the graph, and the interposer tests the API. Neither
tests the arithmetic of the `.cu` kernels. The emulation engine closes that gap.

9.1 Mechanism

- A host execution header maps the CUDA subset to C++
  (`tools/cuda_sim/emu/cuda_emu.h`, with the stub `<cuda_runtime.h>` and
  `<math_constants.h>` beside it).
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

9.2 Supported subset

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

9.3 Acceptance

The emulated kernel must match the CPU reference kernel for the same input,
within the recorded tolerance. This check validates the emulator and the device
code together.

9.4 Risk

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

# Compile the CUDA backend for the target architecture.
tools/nanochat simulate --device h100 --mode compile

# The optional emulated-kernel check.
tools/nanochat simulate --device h100 --suite kernel
```

The command does this:

1. Builds the needed targets under the build lock.
2. Runs the tests with the `t0-cpu` sandbox profile and no GPU queue.
3. For the correctness suite, adds `--config=sim` and the selected precision.
4. For the API suite, builds `//tools/cuda_sim:cuda_sim_interposer` and
   `//tests:cuda_sim_preload_test` under `--config=cuda`, resolves the mock's
   absolute `.so` path, and runs the preload test with
   `--test_env=LD_PRELOAD=<abs .so>` and
   `--test_env=NANOCHAT_SIM_LOG=<abs temp>`. It also runs
   `tools/cuda_sim/check_symbols.py` against the mock, against a real
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

## 12. Validation of the simulator

12.1 Reference engine

1. `//tests:oracle_test` passes under `--config=sim` at fp32.
2. `//tests:train_parity_test` passes under `--config=sim` at fp32.
3. The finite-difference checks pass under `--config=sim`.
4. The reference engine gives identical values with and without the profile
   override.
5. The model test passes when the profile reports a tensor core and a small
   memory budget, and it fails the precision gate when the budget is too small.

12.2 API interposer

6. The symbol gate reads the real binary and the mock. It fails when the mock
   misses a requested symbol. The gate allows extra mock symbols.
7. The binding canary fails when the log holds no header record, or when the
   record names a profile other than `NANOCHAT_SIM_PROFILE`.
8. The API-log test asserts named invariants over the normalized log:
   - The log holds at least one `cudaMalloc`, one kernel launch, and one cuBLAS
     call.
   - Every accepted `cublasGemmStridedBatchedEx` or `cublasGemmEx` record under a
     tensor-core profile has `compute_type == CUBLAS_COMPUTE_32F`, and a
     `cublasGemmEx` record holds `batch_count == 1`.
   - No launch above 48 KB lacks a preceding `cudaFuncSetAttribute`.
   - No pointer value appears in the assertion.
9. A synthetic interposer test issues a 64 KB launch with no attribute. The
   check must fail.
10. The legacy-enum case builds with `--config=cuda` under a simulated
    tensor-core profile. The interposer then reports the legacy enum.

12.3 Collective and kernel

11. The ring schedule test (`//backends/cuda/collective:schedule_test`) runs
    world sizes 1, 2, 4, and 8 and checks the elementwise sum on every rank. A
    rank that skips the all-reduce fails with a reported timeout, not a hang.
12. The emulated kernel matches the CPU reference kernel within tolerance.
14. The compile gate passes for `sm_90` at fp32.

## 14. Risks and non-goals

- The reference engine does not test device arithmetic. The emulation engine
  covers that case, and it is experimental.
- The reference engine does not test tensor-core accumulation. Only real
  hardware does.
- The interposer must track the toolkit version. A pin and the symbol gate
  control that risk.
- The real CUDA runtime may load beside the mock. No call reaches it, but the
  load order needs one early test.
- The collective path needs no mock: the native ring runs on the central
  processing unit over the memory transport.
- A functional PTX interpreter, such as GPGPU-Sim, is a non-goal. It is a large
  external dependency, and it does not target CUDA 12 and `sm_90`. The
  emulation engine uses the source, not the PTX.
- The S0 tier adds a build path away from the GPU. The merge gate runs it, so
  it cannot hide a compile error.

## Part 7. The simulator at bf16 and fp8

_Absorbed from the former `simulator-precision.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

This part is the simulator at bf16 and fp8. Parts 1 to 6
are in [simulator.md](simulator.md): the reference engine, the API interposer,
the collective path, the device profile table, the compile gate, and the
emulation engine.

The subject of this document is the precision coverage. The S0 tier runs the
fp32 build, and Stage 2 of [precision.md](precision.md) adds the bf16 build.
The fp8 path and the bf16 leg of the numeric trace are in the tree, except the
open items of section 13. The table below records which surface covers what.

- Sections 1 to 10 above — the three engines and the S0 tier.
- [precision.md](precision.md) — the fp8 design this document supports.
- [testing.md](testing.md) Part 2 — the trace and the tolerances, and the test
  tiers.

1. Goal and scope

**Goal.** Verify the correctness and the call shape of the bf16 and fp8 paths
without a GPU.

**In scope.**

- The bf16 and fp8 fields in the device profile table.
- The fp8 rounding emulation in the CPU reference backend.
- The `cublasLt` coverage in the API interposer, plus the symbol gate.
- The fp8 conversion intrinsics in the emulation engine.
- The fp8 leg of the numeric trace, with its tolerance.
- The emulator gaps that block the attention work.

**Out of scope.**

- A functional PTX interpreter. `docs/simulator.md` section 14 rejects it.
- A model of the tensor-core accumulation order. Only hardware has it.
- A cycle-accurate model. See [performance-model.md](performance-model.md).
- An NVIDIA-only library outside cuBLAS and `cublasLt`.

2. The four surfaces today

| Surface | Path | Covers today |
|---|---|---|
| Device profile table | `include/nanochat/device_profile.h`, `src/device_profile.cc` | fp32 and bf16 rates; no fp8 |
| Reference engine | `backends/cpu/kernels.cc` under `--config=sim` | fp32, and bf16 under `--config=bf16` |
| API interposer | `tools/cuda_sim/cuda_sim.map`, `interposer.cc` | The CUDA runtime, and four cuBLAS calls |
| Emulation engine | `tools/cuda_sim/emu/` | `RmsNorm` and `Pointwise`, in fp32 and bf16 |
| Numeric trace | `tests/numerics_trace*` | The exact class, on fp32, bf16, and fp8, on the CPU, the simulator, and the GPU |

3. The device profile table

The table carries `supports_bf16`, and `CapsFromProfile` passes it to `Caps`, so
`Caps::Supports(DType::kBf16)` answers for a simulated device (Stage 2 of
[precision.md](precision.md)).

Add three things:

1. `supports_fp8` to `DeviceProfile`.
2. `peak_fp8_flops` to `DeviceProfile`.
3. `supports_fp8` to `Caps`, and extend `Caps::Supports` with it.

The rate is dense and non-sparse. Use 1,979 TFLOP/s for the H100 and the H200.
Use 0 for every other entry in the table today.

| Device | fp32 | bf16 | fp8 | fp8 support |
|---|---:|---:|---:|---|
| h100 | 66.9 TFLOP/s | 989 TFLOP/s | 1,979 TFLOP/s | yes |
| t4 | 8.1 TFLOP/s | 0 | 0 | no |

The gate is `//tests:device_profile_test`. Extend it with three assertions:

- every profile with an fp8 rate has `supports_fp8` true;
- every profile with `supports_fp8` true reports compute capability 8.9 or
  higher;
- a device without fp8 support rejects an fp8 build at startup.

The last assertion is the important one. It ties the table to the runtime gate in
`src/train.cc`.

4. The reference engine

The reference engine runs the CPU backend with a profile override. It computes
real values, so it can verify an fp8 graph.

Add an fp8 rounding emulation to the CPU backend. Model it as two functions:

```cpp
// One shared header, used by the CPU backend and the emulation engine.
float Fp8E4M3Round(float value);   // nearest, with saturation at 448
float Fp8E5M2Round(float value);   // nearest, with saturation
```

The scale stays on the host side. The backend applies the scale, rounds, and
unrounds. Keep the accumulator fp32, as the design requires.

The one important rule: **share the conversion header** between the CPU backend
and the host emulator. Then the simulator leg is exact by construction, as
`docs/testing.md` Part 2, section 3 clause 2 requires. Two separate
conversion implementations would break that clause.

Add a build config for the test:

```bash
tools/nanochat simulate --device h100 --precision fp8 --suite correctness
```

5. The API interposer

#5.1 The new symbol set

The fp8 path calls `cublasLtMatmul`, not `cublasGemmEx`. The interposer must
export the whole family, or the dynamic linker fails at load time.

Add these symbols to `tools/cuda_sim/cuda_sim.map` under a new version node,
`libcublasLt.so.12`:

| Symbol | Purpose |
|---|---|
| `cublasLtCreate` | The library handle |
| `cublasLtDestroy` | The handle release |
| `cublasLtMatmul` | The scaled GEMM |
| `cublasLtMatmulDescCreate` | The operation descriptor |
| `cublasLtMatmulDescDestroy` | The descriptor release |
| `cublasLtMatmulDescSetAttribute` | The scale pointers |
| `cublasLtMatmulDescGetAttribute` | The read-back for the test |
| `cublasLtMatrixLayoutCreate` | The operand layout |
| `cublasLtMatrixLayoutDestroy` | The layout release |
| `cublasLtMatrixLayoutSetAttribute` | The layout attributes |
| `cublasLtMatmulPreferenceCreate` | The preference object |
| `cublasLtMatmulPreferenceDestroy` | The preference release |
| `cublasLtMatmulPreferenceSetAttribute` | The preference attributes |
| `cublasLtMatmulAlgoGetHeuristic` | The algorithm choice |
| `cublasLtGetStatusString` | The error text |

`tools/cuda_sim/check_symbols.py` reads the map as the required set, and compares
it against the mock's dynamic symbol table and the binary's undefined symbols.
The gate therefore fails until every symbol resolves. That is the intent.

#5.2 The new validation rules

The interposer records and checks each call. Add these rules to the table in
`docs/simulator.md` section 5.3.

| Check | Rule |
|---|---|
| fp8 operand type | On an fp8 profile, A and B are `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2` |
| fp8 accumulator | C is `CUDA_R_32F` on every fp8 call |
| fp8 compute type | The compute type is `CUBLAS_COMPUTE_32F` |
| Scale presence | A non-null scale pointer exists for A and for B |
| Scale range | Each scale is finite and greater than zero |
| Capability | No fp8 call appears on a profile without fp8 support |
| Descriptor order | A scale attribute is set after the descriptor exists |
| Handle lifetime | No call uses a destroyed `cublasLt` handle |

The scale-pointer check is the direct analogue of the compute-type check
that the interposer already applies.

#5.3 A design limit

The interposer does not compute. It proves the call shape. The numeric proof
comes from the reference engine. Keep the two runs separate, as
`docs/simulator.md` section 5.4 states.

6. The emulation engine

The emulation engine runs the unchanged `.cu` sources on the host. It is the only
off-GPU check of the device arithmetic.

#6.1 The fp8 intrinsics

Add a header `tools/cuda_sim/emu/cuda_fp8.h` in `tools/cuda_sim/emu/`. Map:

| CUDA construct | Host mapping |
|---|---|
| `__nv_fp8_e4m3` | The project type for the format |
| `__nv_fp8_e5m2` | The project type for the format |
| `__nv_cvt_float_to_fp8` | The shared rounding function from section 4 |
| `__nv_cvt_fp8_to_float` | The inverse |
| `__nv_cvt_fp8x2_to_half2` | Two conversions, then a pack |
| `__nv_cvt_halfraw2_to_fp8x2` | Two conversions, then a pack |

Use the shared rounding function. Do not write a second one.

#6.2 The blocking gaps

Two gaps block the attention work:

1. `extern __shared__` is not mapped. `docs/simulator.md` section 9.2 records
   this. The fused attention forward and the value gate both use it.
2. There is no tensor-core primitive. An `mma` or `wgmma` instruction has no
   host mapping today.

The consequence is direct: **the emulator cannot check a tensor-core attention
tile.** It can check the cuBLAS attention path, because that path has no tile.

Two options close the gap:

- **Option 1: map `mma` to a scalar loop.** The mapping is slow, and it proves
  only the arithmetic order. It cannot prove the layout or the warp
  cooperation.
- **Option 2: buy one hardware leg.** Run the tile test on a real Ada or Hopper
  card.

Prefer option 2 for the tile, and option 1 only for a narrow case. Record the
choice in [attention.md](attention.md) Part 6, section 6.

Add `extern __shared__` to the emulator regardless. The value gate uses it, and
that family is small.

7. The numeric trace

The trace has one fixture with three sections and four consumers: the CPU leg,
the simulator leg, the CUDA leg, and the comparison. The fp8 leg reads the fp8
section.

| Leg | Comparison | Needs a GPU |
|---|---|---|
| CPU, fp8 emulation | Exact against the fp8 section | no |
| Simulator, fp8 profile | Exact against the fp8 section | no |
| CUDA, fp8 | Exact against the fp8 section, on the discrete records | yes |

Add one row per stage to the fp8 parity bounds in
[testing.md](testing.md). The trace compares the fp8 discrete results byte for
byte, so it carries no numeric row. Start with these values, and correct them
from the first measurement.

| Stage | Artifact | GPU against the CPU golden |
|---|---|---|
| Train | loss | 1e-2 relative |
| Train | gradient norm | 1e-2 relative |
| Train | parameter L2 after five steps | 1e-2 relative |
| Eval | bits per byte | 1e-2 relative |
| Eval | greedy argmax identifiers | Exact |

The fp8 tolerance is wide, and it must be. fp8 carries three mantissa bits.
State the tolerance in the document, and do not tighten it without evidence.

Keep the greedy-identifier row exact. A greedy step is a comparison, not an
accumulation.

8. The collective

No change. The reduction operand is fp32, and
`docs/distributed.md` Part 3, section 6 states that rule. Do not reduce an
fp8 gradient in fp8.

The ring schedule test already covers the multi-rank case at fp32. Extend
nothing.

9. Invariants

1. The simulator changes capabilities, never values.
2. One rounding function serves the CPU backend and the emulator.
3. The interposer proves the call shape and computes nothing.
4. The emulator never replaces the oracle.
5. The fp8 leg needs a real card for the final numeric proof.
6. The collective operand stays fp32.

10. Deliverables and owners

| Deliverable | Path | Owner |
|---|---|---|
| The profile fields | `include/nanochat/device_profile.h` | Architect |
| The profile rows | `src/device_profile.cc` | Harness |
| The capability flags | `include/nanochat/tensor.h` | Architect |
| The consistency test | `tests/device_profile_test.cc` | Oracle |
| The fp8 rounding | A shared header | Runtime |
| The CPU backend emulation | `backends/cpu/kernels.cc` | Runtime |
| The interposer symbols | `tools/cuda_sim/cuda_sim.map`, `interposer.cc` | Simulator |
| The symbol gate | `tools/cuda_sim/check_symbols.py` | Simulator |
| The emulator fp8 header | `tools/cuda_sim/emu/cuda_fp8.h` | Simulator |
| The trace leg | `tests/numerics_trace.h` | Oracle |
| The tolerance rows | `docs/testing.md` | Architect |
| This document | `docs/simulator.md` | Architect |

11. Validation

1. `//tests:device_profile_test` passes with the fp8 rows.
2. `tools/nanochat simulate --device h100 --precision fp8 --suite correctness`
   passes.
3. `tools/nanochat simulate --device h100 --suite api` passes, and the symbol
   gate resolves every new `cublasLt` symbol.
4. The negative self-test of `check_symbols.py` still fails on a missing symbol.
5. The fp8 kernel emulation matches the CPU fp8 reference within the recorded
   tolerance.
6. The fp8 trace leg passes on a real card.

Checks 1 to 5 need no GPU.

12. Phases

| Phase | Work | Gate | GPU |
|---|---|---|---|
| P0 | The profile fields, the rows, and the capability flags | The consistency test passes | no |
| P1 | The shared fp8 rounding function | A unit test covers the saturation and the rounding rule | no |
| P2 | The CPU backend fp8 emulation | The reference engine runs an fp8 graph | no |
| P3 | The interposer `cublasLt` coverage | The symbol gate and the API suite pass | no |
| P4 | The emulator fp8 header | The fp8 kernel test matches the CPU reference | no |
| P5 | The trace leg and the tolerance rows | The exact CPU and simulator legs pass | no |
| P6 | The `extern __shared__` mapping | The value-gate emulation matches the reference | no |
| P7 | The hardware confirmation | The fp8 trace passes on a real card | yes |

13. Status by section

| Section | Status |
|---|---|
| 3 The profile table | Delivered. `include/nanochat/device_profile.h` carries `supports_fp8`, `supports_bf16`, and the two peak rates |
| 4 The reference engine | Delivered. The shared rounding header is `include/nanochat/fp8.h` |
| 5 The API interposer | Delivered. `tools/cuda_sim/cuda_sim.map` and `tools/cuda_sim/check_symbols.py` gate it |
| 6 The emulation engine | Delivered, except the `mma` scalar loop |
| 7 The numeric trace | Delivered for the CPU and the simulator legs. The external fp8 golden is open |
| 12 Phases | Partly complete. See [testing.md](testing.md) Part 1, section 6 |

14. Risks and non-goals

- **The symbol gate tracks the toolkit.** A new CUDA version adds a symbol. Pin
  the toolkit, and let the gate fail on a mismatch.
- **The fp8 tolerance may hide a fault.** Prefer a narrow fp8 case with a known
  answer over a wide tolerance on a large case.
- **The emulator cannot prove the layout.** Only hardware proves a tensor-core
  layout.
- **Not a goal:** a PTX interpreter, a tensor-core model, and a second numeric
  implementation.

15. Open questions

1. Does the shared fp8 rounding header live under `include/nanochat/` or under
   `tools/cuda_sim/emu/`?
2. Does the interposer mock the `cublasLt` heuristic, or return a fixed
   algorithm?
3. Does the emulator map `mma` to a scalar loop, or wait for a hardware leg?
4. Is one fp8 trace case enough, or is a case per stage necessary?

See also

- Parts 1 to 6 above — the S0 simulator and its parts.
- [precision.md](precision.md) — the fp8 design.
- [testing.md](testing.md) — the trace and tolerances.
- [performance-model.md](performance-model.md) — the time bound.
