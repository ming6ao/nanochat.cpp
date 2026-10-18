# The simulator at bf16 and fp8

Status: proposed. This document is Part 7 of the simulator work. Parts 1 to 6
are in [simulator.md](simulator.md): the reference engine, the API interposer,
the collective mock, the device profile table, the compile gate, and the
emulation engine.

The subject of this document is the precision coverage. The S0 tier runs the
fp32 build, and Stage 2 of [precision.md](precision.md) adds the bf16 build.
The fp8 work and the bf16 leg of the numeric trace stay open: the table below
records which surface covers what today.

- [simulator.md](simulator.md) — the three engines and the S0 tier.
- [fp8-training.md](fp8-training.md) — the fp8 design this document supports.
- [numerics-integration.md](numerics-integration.md) — the trace and tolerances.
- [testing.md](testing.md) — the test tiers.

## 1. Goal and scope

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

## 2. The four surfaces today

| Surface | Path | Covers today |
|---|---|---|
| Device profile table | `include/nanochat/device_profile.h`, `src/device_profile.cc` | fp32 and bf16 rates; no fp8 |
| Reference engine | `backends/cpu/kernels.cc` under `--config=sim` | fp32, and bf16 under `--config=bf16` |
| API interposer | `tools/cuda_sim/cuda_sim.map`, `interposer.cc` | The CUDA runtime, and four cuBLAS calls |
| Emulation engine | `tools/cuda_sim/emu/` | `RmsNorm` and `Pointwise`, in fp32 and bf16 |
| Numeric trace | `tests/numerics_trace*` | fp32 on the CPU, the simulator, and the GPU |

## 3. Part 1: the device profile table

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
| h200 | 67.0 TFLOP/s | 989 TFLOP/s | 1,979 TFLOP/s | yes |
| h100 | 66.9 TFLOP/s | 989 TFLOP/s | 1,979 TFLOP/s | yes |
| a100 | 19.5 TFLOP/s | 312 TFLOP/s | 0 | no |
| t4 | 8.1 TFLOP/s | 0 | 0 | no |
| gtx1080ti | 11.34 TFLOP/s | 0 | 0 | no |

The gate is `//tests:device_profile_test`. Extend it with three assertions:

- every profile with an fp8 rate has `supports_fp8` true;
- every profile with `supports_fp8` true reports compute capability 8.9 or
  higher;
- a device without fp8 support rejects an fp8 build at startup.

The last assertion is the important one. It ties the table to the runtime gate in
`src/train.cc`.

## 4. Part 2: the reference engine

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
`docs/numerics-integration.md` section 3 clause 2 requires. Two separate
conversion implementations would break that clause.

Add a build config for the test:

```bash
tools/nanochat simulate --device h100 --precision fp8 --suite correctness
```

## 5. Part 3: the API interposer

### 5.1 The new symbol set

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

### 5.2 The new validation rules

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

### 5.3 A design limit

The interposer does not compute. It proves the call shape. The numeric proof
comes from the reference engine. Keep the two runs separate, as
`docs/simulator.md` section 5.4 states.

## 6. Part 4: the emulation engine

The emulation engine runs the unchanged `.cu` sources on the host. It is the only
off-GPU check of the device arithmetic.

### 6.1 The fp8 intrinsics

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

### 6.2 The blocking gaps

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
choice in [attention-tensor-core.md](attention-tensor-core.md) phase A3.

Add `extern __shared__` to the emulator regardless. The value gate uses it, and
that family is small.

## 7. Part 5: the numeric trace

The trace has one golden file and four consumers: the CPU leg, the simulator
leg, the CUDA leg, and the comparison. Add an fp8 leg.

| Leg | Comparison | Needs a GPU |
|---|---|---|
| CPU, fp8 emulation | Exact against the golden | no |
| Simulator, fp8 profile | Exact against the CPU leg | no |
| CUDA, fp8 | Within the fp8 tolerance | yes |

Add one row per stage to the tolerance table in
`docs/numerics-integration.md` section 4. Start with these values, and correct
them from the first measurement.

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

## 8. Part 6: the collective

No change. The reduction operand is fp32, and
`docs/distributed-native-plan.md` section 6 states that rule. Do not reduce an
fp8 gradient in fp8.

The collective mock already covers the multi-rank case at fp32. Extend nothing.

## 9. Invariants

1. The simulator changes capabilities, never values.
2. One rounding function serves the CPU backend and the emulator.
3. The interposer proves the call shape and computes nothing.
4. The emulator never replaces the oracle.
5. The fp8 leg needs a real card for the final numeric proof.
6. The collective operand stays fp32.

## 10. Deliverables and owners

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
| The tolerance rows | `docs/numerics-integration.md` | Architect |
| This document | `docs/simulator-precision.md` | Architect |

## 11. Validation

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

## 12. Phases

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

## 13. Status by section

| Section | Status |
|---|---|
| 3 The profile table | Proposed |
| 4 The reference engine | Proposed |
| 5 The API interposer | Proposed |
| 6 The emulation engine | Proposed |
| 7 The numeric trace | Proposed |
| 12 Phases | Not started |

## 14. Risks and non-goals

- **The symbol gate tracks the toolkit.** A new CUDA version adds a symbol. Pin
  the toolkit, and let the gate fail on a mismatch.
- **The fp8 tolerance may hide a fault.** Prefer a narrow fp8 case with a known
  answer over a wide tolerance on a large case.
- **The emulator cannot prove the layout.** Only hardware proves a tensor-core
  layout.
- **Not a goal:** a PTX interpreter, a tensor-core model, and a second numeric
  implementation.

## 15. Open questions

1. Does the shared fp8 rounding header live under `include/nanochat/` or under
   `tools/cuda_sim/emu/`?
2. Does the interposer mock the `cublasLt` heuristic, or return a fixed
   algorithm?
3. Does the emulator map `mma` to a scalar loop, or wait for a hardware leg?
4. Is one fp8 trace case enough, or is a case per stage necessary?

## See also

- [simulator.md](simulator.md) — the S0 simulator and its parts.
- [fp8-training.md](fp8-training.md) — the fp8 design.
- [numerics-integration.md](numerics-integration.md) — the trace and tolerances.
- [performance-model.md](performance-model.md) — the time bound.
