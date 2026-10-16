# Precision policy

Status: accepted. Stage 1 is complete; Stage 2 comes later. This document
defines the precision policy for every host. One build has one compute type.
The tree uses fp32 and bf16 only. It holds no fp16 code.

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

The Ampere row is Stage 2. It comes later (section 6).

The named targets follow the table. The `a100` target takes bf16 in Stage 2. The
`h100` target is fp32 today and takes bf16 in Stage 2. The `h200` target is
fp16 today and takes fp32 in Stage 1, then bf16 in Stage 2.

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

### Stage 2: add the bf16 compute path for Ampere and newer (pending)

Some groundwork already exists. The tree holds the `Bf16` type, the
`DType::kBf16` tag, the `supports_bf16` field, and the bf16 rates. The compute
path is the remaining work.

1. Add `Caps::supports_bf16` and extend `Caps::Supports`.
2. Wire `supports_bf16` from `src/device_profile.cc` into `Caps`.
3. Add the bf16 GEMM. Use `cublasGemmEx` with `CUDA_R_16BF` and
   `CUBLAS_COMPUTE_32F`.
4. Add the mixed storage seam: bf16 activations, and fp32 master weights,
   gradients, moments, and statistics.
5. Add `--config=bf16`, and move the `a100`, `h100`, and `h200` targets to it.
6. Add the bf16 leg to the simulator, the emulator, and the numeric trace.
7. Keep the persistent state in fp32 (section 5).

Stage 2 needs the fp32 gradient seam change. It needs no loss scaling. Other
documents call this stage "Stage 2a".

## 7. What this avoids

Stage 1 removes the fp16 emulator header, the second set of tolerances, the
fp16 test leg, and the fp16 simulator legs.

Stage 2 removes the loss-scaling project:

- the dynamic scale, the growth, and the backoff;
- the inf and nan detection and the step skip;
- the half to fp32 gradient unscale.

Stage 2 removes the fp16 master-weight mirror. The fp32 build needs no mirror,
because the master weight aliases the parameter.

The checkpoint format is precision-independent. The model saves an fp32 master
with no dtype tag (`src/model.cc:1040`). No migration is necessary.

## 8. Risks

- The T4 is the Kaggle target (`docs/build.md`). The T4 loses the tensor-core
  rate, because Turing has no bf16. The peak fp16 rate is 65 TFLOP/s. The fp32
  rate is 8.1 TFLOP/s. Section 9 records the answer.
- The attention kernel does not use tensor cores today (`docs/build.md`).
  Attention runs at the fp32 rate in either case. The realized loss is below
  8 times.
- Stage 1 touches every backend, the simulator, and the tests. Land it as one
  change, and keep the fp32 gates green.
- The `h200` target is fp32 until Stage 2.
- Stage 2 is new code. Keep the fp32 build as the correctness gate.

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

## See also

- [build.md](build.md) — backends and the build configuration.
- [host-portability.md](host-portability.md) — the host and the Kaggle procedure.
- [kernels.md](kernels.md) — the backend seam.
