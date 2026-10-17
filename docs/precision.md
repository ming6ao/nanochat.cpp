# Precision policy

Status: accepted. Stage 1 is complete. Stage 2 is delivered for the C++ and
CUDA surfaces. This document defines the precision policy for every host. One
build has one compute type. The tree uses fp32 and bf16 only. It holds no fp16
code.

## 1. Decision

Use one rule for every device:

1. Use fp32 for the compute on Pascal, Turing, and the central processing unit.
2. Use bf16 for the compute on Ampere and newer.
3. Keep the persistent state in fp32 on every device.
4. Do not use loss scaling. Neither fp32 nor bf16 needs it.

Remove fp16 from the tree. The removal covers the production builds, the
simulator, the emulator, the type system, the test legs, and the estimator.

This decision reverses the earlier choice twice. Commit `02c1dd6` kept an fp32
master weight beside an fp16 compute copy. The first reversal removed that
compute copy from the production path. The second reversal removes the
test-only fp16 build as well. Section 6 lists the work.

## 2. Reason

fp16 is the odd format. It is the only format with a small exponent range.

- bf16 has the fp32 exponent range. A tiny gradient does not underflow. bf16
  needs no loss scaling.
- fp16 has five exponent bits. A tiny gradient underflows. fp16 therefore needs
  loss scaling and fp32 gradient storage.
- On Pascal, fp16 runs at 0.177 TFLOP/s. The fp32 rate is 11.34 TFLOP/s. The
  fp16 rate is 64 times slower.
- On Turing, fp16 is the only accelerated format, and Turing has no bf16. The
  fp16 peak there is 65 TFLOP/s. The fp32 peak is 8.1 TFLOP/s.

One card only gains from fp16. That card is the Turing T4. The gain costs a
second compute type, a second storage tag, a loss-scaling project, an emulator
header, and a second test leg.

The test-only fp16 build costs more than it earns. It carries its own emulator
header, its own tolerances, and its own simulator legs. The bf16 path replaces
it, and the bf16 path needs no loss scaling and no second master weight.

The PyTorch reference makes the same choice. It uses fp32 on pre-Ampere hardware
(`nanochat/common.py:17-32`).

## 3. Device capabilities

The table is from `src/device_profile.cc`. The rates are dense (non-sparse).

| Device | Major.Minor | Tensor cores | fp32 | bf16 |
|---|---|---|---|---|
| gtx1080ti | 6.1 | no | 11.34 TFLOP/s | none |
| t4 | 7.5 | yes | 8.1 TFLOP/s | none |
| a100 | 8.0 | yes | 19.5 TFLOP/s | 312 TFLOP/s |
| h100 | 9.0 | yes | 66.9 TFLOP/s | 989 TFLOP/s |

The table shows the decision. bf16 starts at Ampere. Turing has no bf16, so the
T4 runs fp32.

The device table drops the `supports_fp16` and `peak_fp16_flops` fields. The
estimator drops its fp16 mode with them. The table and the builds agree on one
precision set.

## 4. Policy per target

| Target | Arch | Compute | Persistent state | Config |
|---|---|---|---|---|
| Workstation | sm_61 | fp32 | fp32 | default |
| Central processing unit and continuous integration | none | fp32 | fp32 | `--config=cpu` |
| Kaggle T4 | sm_75 | fp32 | fp32 | `--config=t4` |
| Ampere and newer | sm_80 and sm_90 | bf16 | fp32 | `--config=bf16` |

The Ampere row is Stage 2, which is delivered (section 6).

The named targets follow the table. The `a100` target is sm_80 with bf16. The
`h100` and `h200` targets are sm_90 with bf16. `--config=bf16` selects the bf16
compute type on any architecture.

## 5. Storage rule

This rule applies to bf16 and to fp32. It also applies to a later format.

Keep these buffers in fp32:

- the master weights;
- the parameter gradients;
- the optimizer moments;
- the norm statistics and the attention statistics.

Use the compute dtype for the activations, the GEMM operands, and the key/value
cache.

bf16 has seven mantissa bits. A parameter update needs more precision than
that. bf16 has the fp32 exponent range, so no scale is necessary.

The checkpoint container is a storage format. It stores fp32 or bf16 records,
and the record dtype is separate from the build compute dtype. Store the
canonical checkpoint in fp32. A bf16 container is a derived compact artifact.

The `DType` codes 0, 1, and 2 stay. The code 3 belongs to `kFp16`. Stage 1
retires that tag and keeps the code reserved. Do not assign the code to a new
format later. The container never stored an fp16 record, so no file migration is
necessary.

This rule is stricter than the reference. The reference keeps the matrix
parameters in fp32, but it casts the embeddings to the compute dtype
(`gpt.py:262-266`). Its embedding gradients are therefore bf16. This rule keeps
every parameter gradient in fp32. That is a deliberate divergence.

A bf16 build needs a mixed GEMM. The weight-gradient GEMM reads bf16 operands
and writes an fp32 result. Use `cublasGemmEx` with `Atype = Btype =
CUDA_R_16BF`, `Ctype = CUDA_R_32F`, and `CUBLAS_COMPUTE_32F`.

## 6. Staged rollout

### Stage 1: remove fp16 from the tree (delivered)

The removal is one change. Keep the fp32 gates green in that change.

1. `include/nanochat/tensor.h`: delete the `Fp16` type, the half conversion
   helpers, `DType::kFp16`, `DTypeTraits<Fp16>`, `Caps::supports_fp16`, and the
   `NANOCHAT_PRECISION_FP16` branch of `ComputeType`.
2. `include/nanochat/device_profile.h` and `src/device_profile.cc`: delete
   `supports_fp16` and `peak_fp16_flops`, and the fp16 rate in every row.
3. `include/nanochat/mfu.h` and `src/performance_model.cc`: delete
   `ComputePrecision::kFp16` and the fp16 rows of the estimate.
4. `include/nanochat/kernels.h`: update the two comments that name an fp16
   build.
5. `.bazelrc`, `BUILD.bazel`, `nanochat.bzl`, and `Makefile`: delete the
   `--config=fp16` setting, the `//:fp16` config, and the `PRECISION=fp16`
   branch.
6. `backends/cuda/gemm.cu`: delete the fp16 branch, the `HasTensorCores`
   helper, and the `cuda_fp16.h` include.
7. `backends/cpu/kernels.cc`: delete the half-storage branch.
8. `tools/cuda_sim/emu/cuda_fp16.h`: delete the file, and delete its entry in
   `tools/cuda_sim/emu/BUILD.bazel`.
9. `tools/cuda_sim/emu/cuda_runtime.h` and `cuda_emu.h`: delete the fp16
   intrinsics and the half atomic add.
10. The kernel tests and the fixtures: delete the fp16 legs. Change
    `//backends/cuda/kernels:precision_gpu_test` to a single fp32 build.
11. `src/roofline_main.cc`: delete the `fp16` precision name.
12. `python/nanochat_cpp/_build.py`: delete the fp16 precision mapping.
13. `notebooks/kaggle_setup.py`: confirm that the default stays fp32.
14. The documents: update `docs/build.md`, `docs/host-portability.md`,
    `docs/simulator.md`, `docs/simulator-precision.md`, `docs/testing.md`, and
    `docs/python.md`.
15. `docs/performance-model.md`: delete the fp16 column, and keep the fp32 and
    bf16 columns.

Gate: `tools/nanochat build`, `tools/nanochat test`, and `tools/nanochat lint`
pass. The fp32 numeric trace passes on the central processing unit, the
simulator, and the graphics card. The `sm_90` compile gate passes at fp32. No
target name, flag, or type name contains "fp16".

Result: the gate passes. `tools/nanochat build`, `tools/nanochat test`, and
`tools/nanochat lint` pass. The fp32 numeric trace passes on the central
processing unit and the simulator. The `sm_90` compile gate passes at fp32. No
target name, flag, or type name contains "fp16".

### Stage 2: add the bf16 compute path for Ampere and newer (delivered)

The tree has the `Bf16` type, the `DType::kBf16` tag, device-profile bf16
support, and bf16 peak rates. This stage adds the compute path.

Use this type contract:

- Keep master weights, parameter gradients, optimizer state, loss values, and
  reduction statistics in fp32.
- Store parameter compute copies, activations, activation gradients, GEMM
  operands, and the key/value cache in bf16.
- Accumulate GEMM products and reductions in fp32. Store activation outputs as
  bf16, and store weight-gradient outputs as fp32.
- Accumulate embedding gradients directly into fp32 parameter-gradient buffers.
- Round each stored bf16 value to nearest-even. Do not use loss scaling.

Delivered:

1. `ComputeType` selection. `include/nanochat/tensor.h` selects `float` for
   `NANOCHAT_PRECISION_FP32` and `Bf16` for `NANOCHAT_PRECISION_BF16`. The bf16
   item adds `Bf16(float)`, `operator float`, and the compound assignments, and
   the conversion helpers are `__host__ __device__`.
2. `GradientType` is fp32 in every build. `ParamView::grad` and the optimizer
   buffers use it, so a parameter gradient is never bf16.
3. `Caps::supports_bf16`. `CapsFromProfile` copies it from the device profile,
   the CUDA backend sets it from `major >= 8`, and the CPU reference reports it.
   `Caps::Supports(DType::kBf16)` reports it. `Model::Create` fails fast when
   the build compute type is not supported by the device.
4. A CPU bf16 reference path. The reference stores bf16 where the compute type
   is bf16 and accumulates in fp32 or double, so the CPU records the same
   rounding the device does.
5. The bf16 GEMM. `Gemm` uses `cublasGemmEx` with `CUDA_R_16BF` operands and
   `CUBLAS_COMPUTE_32F`. The new `GemmWgrad` entry point writes an fp32 result,
   so the weight-gradient GEMM reads bf16 operands and accumulates into fp32.
6. The CUDA bf16 kernels. `AtomicAddDev` gains a bf16 compare-and-swap form
   (a half-word write cannot use the native atomic add), and the embedding
   scatter-add, the value-gate weight gradient, and the global norm write fp32.
7. The optimizer path. `AdamWUpdate` and `MuonUpdate` read fp32 gradients and
   keep fp32 moments, then write the bf16 compute copy.
8. The checkpoint path. The canonical record stays fp32. `CollectModelRecords`
   reads the master, a load writes the master and rebuilds the compute copy,
   and a load accepts an fp32 or a bf16 record in any build.
9. `--config=bf16`, and the `a100` target at sm_80. The `h100` and `h200`
   targets at sm_90 select bf16 too.
10. The Python surface. `Model.dtype` and `TensorView.dtype` report the compute
    type, and the parameter view reads and writes Python floats for both
    fp32 and bf16.
11. The simulator and the emulator. The simulated device reports
    `supports_bf16`, the emulator tests bf16 conversion, and the kernel
    emulator tolerance carries the bf16 storage granularity.

Gate: `tools/nanochat build`, `tools/nanochat build --config=bf16`,
`tools/nanochat test`, `tools/nanochat test --config=bf16`, and
`tools/nanochat lint` pass. The compile gate passes for `a100`, `h100`, and
`h200`. The fp32 build keeps every committed fp32 golden.

#### What the bf16 leg does not cover yet

A bf16 build cannot reproduce a committed fp32 golden, so a test that compares
against one prints a skip. Those tests keep their full strength in the fp32
build. The skipped legs are the numeric trace, the oracle, the training parity,
the scoring parity, the model oracle, the model gradient, and the two Python
fixtures. A bf16 build still runs every one of those code paths through the
tests that do not compare against a golden. The in-build two-path checks run in
both builds with a precision-aware bound.

The bf16 numeric trace is the remaining work. It needs its own committed golden,
produced by the bf16 CPU emitter, and its own tolerance table. The fp32 trace and
its tolerances stay unchanged. Other documents call this stage "Stage 2a".

## 7. What this avoids

Stage 1 removes the fp16 emulator header, the second set of tolerances, the
fp16 test leg, and the fp16 simulator legs.

Stage 2 removes the loss-scaling project:

- the dynamic scale, the growth, and the backoff;
- the inf and nan detection and the step skip;
- the half to fp32 gradient unscale.

Stage 2 keeps one fp32 master weight and adds a bf16 compute copy for bf16
builds. In an fp32 build, the master weight still aliases the parameter.

The checkpoint format is precision-independent. The model saves an fp32 master
with no dtype tag (`src/model.cc:1040`). No migration is necessary.

## 8. Risks

- The T4 is the Kaggle target (`docs/build.md`). The T4 loses the tensor-core
  rate, because Turing has no bf16. The peak fp16 rate is 65 TFLOP/s. The fp32
  rate is 8.1 TFLOP/s. Section 9 records the answer.
- The attention kernel does not use tensor cores today (`docs/build.md`).
  Attention runs at the fp32 rate in either case. The realized loss is below
  8 times.
- Stage 1 touches every backend, the simulator, and the tests. Stage 2 touches
  the same set again, because the fp32 build stays the correctness gate and
  must pass unchanged.
- The `h200` target is sm_90 with bf16, like `h100`.
- A bf16 build cannot reproduce a committed fp32 golden, so the golden-fixture
  gates run in the fp32 build and print a skip in the bf16 build. The bf16
  numeric trace is the remaining leg (section 6).
- Stage 2 is new code. The fp32 build is still the correctness gate, and the
  tree keeps every fp32 test at its recorded tolerance.

## 9. Decisions on the former open questions

1. **The T4 training speed.** The Kaggle T4 target runs fp32 and keeps it. Do
   not restore fp16. Do not start a loss-scaling project. Record the measured
   step time for the Kaggle shape in `docs/performance.md`. If the step time is
   not acceptable, reduce the model size. Do not add a second compute type.
2. **The fp16 central processing unit emulation.** Remove it. Remove fp16
   support everywhere. The simulator covers fp32 until Stage 2 adds bf16.
3. **The first Ampere target.** Use sm_80 as the first Ampere target. bf16
   starts at sm_80, and the `a100` profile is already in the device table. The
   `h100` and `h200` targets (sm_90) reuse the same precision config. The gate
   order is the `a100` profile under the simulator, then an sm_80 card, then an
   sm_90 card.
4. **The dangling references.** Fixed in Stage 1. `.bazelrc` points to
   `docs/build.md` and `docs/python.md`. `tools/sandbox/profiles.conf` points to
   `docs/host-portability.md`.
5. **The bf16 gate.** The fp32 build stays the golden gate. In a bf16 build the
   golden-fixture tests print a skip, and the in-build equality checks run with a
   bound derived from the storage precision. The bf16 numeric trace becomes the
   bf16 gate when it lands.

## See also

- [build.md](build.md) — backends and the build configuration.
- [host-portability.md](host-portability.md) — the host and the Kaggle procedure.
- [kernels.md](kernels.md) — the backend seam.
