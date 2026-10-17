# Correctness gates: the simulator and the Hopper startup check

Status: proposed. This document is Part 8 of the simulator work. Parts 1 to 6
are in [simulator.md](simulator.md). Part 7 is
[simulator-precision.md](simulator-precision.md).

This document answers three questions:

1. Which part of the fp8 and bf16 correctness can the simulator prove without a
   GPU?
2. How does the project prove the remaining part on a real Hopper card?
3. How does the project keep the simulator fast?

The reference designs are [fp8-training.md](fp8-training.md),
[precision.md](precision.md), [numerics-integration.md](numerics-integration.md),
and [attention-tensor-core.md](attention-tensor-core.md).

## 1. The split

The simulator is the S0 tier. It is CPU-only and it takes no GPU broker
([testing.md](testing.md) section "S0: the simulator tier"). The simulator
computes real values on the CPU reference backend. It cannot compute a
tensor-core result, because the CPU has no tensor core.

The plan therefore has two halves:

- **Part 1** extends the simulator, so the off-GPU proof is as complete as
  possible.
- **Part 2** adds a fast hardware check at the start of a training run.

Part 3 keeps the simulator fast.

## 2. The coverage boundary

The table states which surface proves which claim. The last column names the
only gate that can prove the claim.

| Surface | Simulator after Part 1 | Hopper startup gate |
|---|---|---|
| fp8 rounding, E4M3 and E5M2 | Exact, through a shared header | Confirm exact |
| The amax, scale, and cast kernel | Emulator against the CPU reference | Confirm |
| The scaled GEMM, forward | The CPU reference value only | The real value |
| The scaled GEMM, dgrad and wgrad | The CPU reference value only | The real value |
| The fp8 layout, three GEMMs | Named, not proven | Proven |
| The `fast_accum` flag | The call shape only | Proven |
| The fp8 layer set | Exact on the graph | Confirm |
| The graph: loss, gradients, optimizer | Exact against the CPU golden | Confirm within 1e-2 |
| The tensor-core attention tile | Not possible | Proven |
| The `cublasLt` call shape | Interposer | Confirm on the device |
| The arena and the workspace size | The size model only | Proven |
| The collective numerics | The host reference, exact | Confirm on NVLink |

Two limits are load bearing. `simulator-precision.md` section 5.3 states that
the interposer proves the call shape and computes nothing. `simulator.md`
section 14 states that only hardware proves a tensor-core layout.

## 3. Part 1: extend the simulator

This part follows phases P0 to P6 of
[simulator-precision.md](simulator-precision.md). Each subsection names the
deliverable and the gate.

### 3.1 The profile and the capability fields

The `peak_fp8_flops` field already exists at
`include/nanochat/device_profile.h:47`. Add the rest:

- `supports_fp8` to `DeviceProfile`.
- `supports_fp8` to `Caps`, and an fp8 branch in `Caps::Supports`
  (`include/nanochat/tensor.h:199`).

Extend `//tests:device_profile_test` with the assertions in
[simulator-precision.md](simulator-precision.md) section 3. Two assertions hold
today: a nonzero fp8 rate matches the flag, and an fp8 device reports compute
capability 8.9 or higher. The third assertion, the startup rejection of an fp8
build on a device without support, needs the fp8 build, so it lands in A3.

Gate: the test passes.

### 3.2 The shared rounding header

The header is `include/nanochat/fp8.h`. It sits under `include/nanochat/`, so
the CPU backend and the emulation engine both reach it below the seam. This
resolves open question 1 of [simulator-precision.md](simulator-precision.md).

The header holds the encoding, the decoding, and the round trip for both
formats:

- `Fp8E4M3Encode`, `Fp8E4M3Decode`, `Fp8E4M3Round`;
- `Fp8E5M2Encode`, `Fp8E5M2Decode`, `Fp8E5M2Round`.

Both formats round to nearest with ties to even. Both saturate at the format
maximum, 448 for E4M3 and 57344 for E5M2. The emulation engine needs the encode
and the decode for the CUDA conversion intrinsics
([simulator-precision.md](simulator-precision.md) section 6.1).

`simulator-precision.md` invariant 2 requires one implementation. Two
implementations would break the exact simulator leg.

The unit test is `//tests:fp8_rounding_test`. It pins the hand-computed
encodings, the round trip over every finite code, the tie rule against an
independent direct scan, saturation, the infinity, the NaN, and the sign bit.

Gate: the test passes. It passes today.

### 3.3 The fp8 graph in the reference engine

Add the storage tags and the scaled GEMM seam
([fp8-training.md](fp8-training.md) sections 3 and 5). Implement
`GemmScaledFp8` in `backends/cpu/kernels.cc`. The implementation quantizes,
accumulates in fp32, and dequantizes.

Status: the tree holds the forward path. The seam, both backends, the build
config, and the model plumbing are in. `--config=fp8` and
`simulate --precision fp8`
select the fp8 build. `src/ops.cc` quantizes the eligible forward matmuls, and
`src/model.cc` owns the workspace slots (one byte per operand, one fp32 per
scale, rebuilt per step). `//tests:fp8_graph_test` compares the fp8 forward
against the dense forward at an eligible shape.

The backward stays dense (the mixed choice of [fp8-training.md](fp8-training.md)
section 7.1), so the fp8 layer set is the forward only. The fp8 backward and
the operand reuse across calls are follow-ups.

Gate:

```bash
tools/nanochat simulate --device h100 --precision fp8 --suite correctness
```

The run must reach the committed fp8 golden. That golden comes from the
reference PyTorch fp8 implementation (section 3.6), so the gate cannot pass
yet.

### 3.4 The `cublasLt` coverage

The fp8 path calls `cublasLtMatmul`. Add the 15 symbols in
[simulator-precision.md](simulator-precision.md) section 5.1 to
`tools/cuda_sim/cuda_sim.map` under the `libcublasLt.so.12` node. Add the eight
validation rules in section 5.2.

Status: delivered. The map exports the family, the mock implements it in
`tools/cuda_sim/interposer.cc`, and the ABI lives in
`tools/cuda_sim/cuda_sim_abi.h`. `//tests:cuda_sim_preload_test` issues a scaled
GEMM through the real `gemm.cu` and asserts the recorded call shape.
`check_symbols.sh` resolves 57 required symbols.

Gate: `tools/cuda_sim/check_symbols.sh` resolves every symbol, and the negative
case still fails. Both hold today.

### 3.5 The emulation engine

Add `tools/cuda_sim/emu/cuda_fp8.h` with the six intrinsics in
[simulator-precision.md](simulator-precision.md) section 6.1. Each intrinsic
uses the shared rounding header.

Map `extern __shared__`. This mapping unblocks the value gate and the fused
attention forward. Map `mma` to a scalar loop for one narrow case only. Record
in [attention-tensor-core.md](attention-tensor-core.md) phase A3 that the
tile still needs hardware.

Status: complete except the `mma` scalar loop. `cuda_fp8.h` maps the conversion
intrinsics to `nanochat/fp8.h`. `cuda_emu.h` carries a per-block dynamic shared
buffer, and the BUILD file rewrites the `extern __shared__` declaration in the
emulated copy of `value_gate.cu` and `attention.cu`. The device sources do not
change; only the copies do.

`//tests:kernel_emu_test` compares the emulated cast, the value gate, and the
attention forward (the cuBLAS path and the fused windowed path) against the CPU
reference. It also asserts that the dynamic shared path ran.

The `mma` scalar loop stays open: the tensor-core tile needs hardware
([attention-tensor-core.md](attention-tensor-core.md) phase A3).

Gate: the emulated cast, the value gate, and the attention forward match the
CPU reference. All three hold today.

### 3.6 The fp8 trace leg

Add the fp8 rows to the tolerance table in
[numerics-integration.md](numerics-integration.md) section 4. Use the 1e-2
relative bound from [simulator-precision.md](simulator-precision.md) section 7.
Keep the greedy-identifier row exact.

Status: open. The fp8 trace leg needs a committed fp8 golden. That golden comes
from the reference PyTorch tree (`../nanochat`), which the fixture generator
reaches offline. The in-build fp8 forward check is
`//tests:fp8_graph_test`.

Gate: the CPU leg is exact, and the simulator leg equals the CPU leg.

## 4. Part 2: the Hopper startup gate

### 4.1 The goal

Prove, once per fresh run on the real card, the four things that the simulator
cannot. Abort before the first optimizer step on a mismatch. Do not fall back to
bf16, because `DESIGN.md` section 6 forbids a runtime precision branch.

Status: delivered. `src/hw_gate.cc` implements the cases and a host reference.
`src/train.cc` runs it before the first step when the device is real, and exits
with code 2 on a mismatch. `train_main --hw-check {auto,on,off}` selects the
behaviour. `//tests:hw_gate_cuda_test` is the standalone twin. There is no
stamp, by decision: the gate always runs on a fresh run.

### 4.2 The shape of the gate

Write one function, `RunHardwareCorrectnessGate`, in `src/hw_gate.cc` with its
header in `src/hw_gate.h`. The Harness workstream owns `src/`. The same source
also builds as a standalone test target, so the project tests the logic once.

The gate uses the real workspace and the real seam calls. Every case uses a tiny
shape: `B=2`, `T=8`, `hidden=64`, `vocab=128`.

| Step | Check | Bound |
|---|---|---|
| 1 | The device fp8 conversion against the shared rounding table | Exact |
| 2 | The scaled GEMM, forward, E4M3, `fast_accum` on | 1e-2 relative |
| 3 | The scaled GEMM, dgrad, E5M2, `fast_accum` off | 1e-2 relative |
| 4 | The scaled GEMM, wgrad, E5M2, `fast_accum` off | 1e-2 relative |
| 5 | The three layouts on a non-square matrix | 1e-2 relative |
| 6 | The `fast_accum` pair: the flag changes the result, and both stay inside the bound | 1e-2 relative |
| 7 | The attention forward and backward at one small shape | 1e-2 relative |
| 8 | A two-step training step: the loss and the gradient norm | 1e-2 relative |

Step 5 uses a non-square matrix, so a wrong stride shows. Step 6 is the
fast-accumulation contract. Step 7 is the tensor-core attention. Step 8 ties the
graph to the device.

The golden values come from the CPU reference with a fixed seed. Commit them as
one fixture, `tests/data/hw_gate.bin`. The gate then needs no torch at run time.

### 4.3 The integration into training

`train_main` gains `--hw-check {auto,on,off}`. The default is `auto`.

`auto` runs the gate when three conditions hold:

- the build has an fp8 or a bf16 path;
- the device is Hopper-class;
- the run is not a resume.

`auto` skips the gate on a resume when the checkpoint carries a passing stamp
for the same commit and the same device.

Put the call in `src/train.cc` inside `Train()`. Put it after the model and the
workspace exist, and before the first step. The analytic bound report at
`src/train.cc:973` is the place to put it beside.

On a mismatch, the gate logs the failing case with the two values. The process
then exits with code 2.

The gate writes three fields into the checkpoint: `hw_check/commit`,
`hw_check/device`, and `hw_check/result`.

The gate is in-process. It reuses the CUDA context and the `cublasLt` handle
that training needs. A separate process would pay a second context setup.

### 4.4 The cost budget

| Item | Cost |
|---|---|
| The CUDA context and the handle | Already paid by training |
| The first `cublasLt` heuristic query | 0.1 to 0.5 s |
| The conversion table | Less than 1 ms |
| The three GEMMs and the layouts | Less than 10 ms |
| The small attention forward and backward | Less than 20 ms |
| The two training steps at depth 2 | About 50 ms |
| **Total added to a fresh run** | **Less than 1 s** |

The gate runs once per fresh run. The cost is far below one training step at the
production shape.

### 4.5 The warmup variant

The learning-rate warmup is a schedule, not a correctness phase
(`src/optim.cc:638`). Keep it separate.

If the operator wants a live check, add `--hw-check=warmup`. That mode logs the
loss and the gradient norm of the first `N` steps. The operator compares them
against the golden offline. Keep it off by default, because it costs the
production step time.

### 4.6 The continuous integration twin

Add `//tests:hw_gate_cuda_test` with the tags `gpu` and `manual`. The target
links the same `src/hw_gate.cc`. Run it through
`tools/nanochat test --gpu`. It is the same check, without a training run around
it.

### 4.7 The fault injection

The gate needs a way to prove that it fails. Add a debug flag,
`--hw-check-inject=case`, that corrupts one golden value. The test asserts the
exit code 2.

## 5. Part 3: keep the simulator fast

### 5.1 The suite split

Do not put the fp8 work into the slow `correctness` suite alone. Split the work
by cost.

| Suite | Contents | Target | When |
|---|---|---|---|
| `fp8-fast` | The rounding unit, the seam GEMM at a tiny shape, the profile rows, the shared-header equality | Less than 2 s | Every edit, in the default T0 loop |
| `correctness` | The oracle, the training parity, the fp32 trace | The current cost | Per commit |
| `fp8-full` | The fp8 oracle, the fp8 training parity, the fp8 trace at a tiny shape, the emulated cast kernel | Less than 60 s | Merge gate |
| `api` | The `cublasLt` call shape, the symbol gate | Less than 30 s | Merge gate only |
| `compile` | `sm_90` under `-Werror` | The build time | Merge gate only |

### 5.2 The speed measures

1. **One process per suite.** Build the model once and run every fp8 case in it.
   The current `correctness` suite runs four separate binaries.
2. **Tiny shapes in the fast suite.** Use `B=2, T=8, hidden=64, vocab=128`. The
   trace config in [numerics-integration.md](numerics-integration.md) section
   6.2 is 10 layers, hidden 384, sequence 256, and batch 4. That config is too
   slow for the inner loop, so it stays `manual`.
3. **Committed, hashed goldens.** Compare a 64-bit hash for the exact legs, not
   a value dump. The trace already uses FNV-1a hashes.
4. **No torch at run time.** The fixtures are data.
5. **Pin the reference thread count to one.** The CPU reductions depend on the
   thread count. The pin also removes contention.
6. **Do not rebuild the CUDA backend for a CPU-only suite.** Only the `api` and
   `compile` suites need it.
7. **Run under `t0-cpu` at 200 percent with `--local_test_jobs=2`.** Two
   CPU-bound jobs run at once. The aggregate slice still bounds the host.
8. **Emulate only what the gate needs.** Map `extern __shared__` and one scalar `mma`
   case. Do not emulate the attention at the production shape.
9. **Record the time.** Add a `simulate --timing` line. The line prints the wall
   time per suite, so a regression is visible.

### 5.3 The reason this stays fast

The reference engine computes every value on the CPU. Its cost scales with the
shape, the number of steps, and the number of layers. The fast suite pays none
of that, because it checks the rounding and one tiny GEMM. The heavy graph
checks move to the merge gate.

## 6. Phases and owners

| Phase | Work | Owner | Gate |
|---|---|---|---|
| A1 | The fp8 profile and capability fields, and the two `DType` tags | Architect, Harness, Oracle | `device_profile_test` |
| A2 | The shared rounding header and its unit test | Runtime, Oracle | The exact bit table |
| A3 | `GemmScaledFp8` and the fp8 graph | Runtime | `simulate --precision fp8` |
| C1 | The `fp8-fast` suite | Architect | Less than 2 s |
| A4 | The `cublasLt` interposer and the symbol gate | Simulator | `check_symbols.sh` |
| A5 | The emulator fp8 header and `extern __shared__` | Simulator | `kernel_emu_test` |
| A6 | The fp8 trace leg and the tolerance rows | Oracle, Architect | The exact CPU and simulator legs |
| B1 | The hardware startup gate | Harness | `hw_gate_cuda_test` |
| B2 | The integration into `train_main` | Harness | A live run. There is no stamp. |

Do A1, A2, A3, and C1 first, so the fast gate exists early. Then do A4, A5, and
A6. Then do B1 and B2.

Status: A1, A2, A3 (the forward), A4, A5 (except the `mma` loop), B1, and C1 are
complete. The two `DType` tags moved into A1, because `Caps::Supports` reads
them. The open items are the fp8 backward, the external fp8 golden for A6, and
the `mma` loop. See section 9.

## 7. Validation

1. `tools/nanochat simulate --device h100 --precision fp8 --suite fp8-fast`
   passes in less than 2 seconds.
2. `tools/nanochat simulate --device h100 --precision fp8 --suite correctness`
   passes.
3. The fp8 trace leg is exact on the CPU and on the simulator.
4. `check_symbols.sh` resolves every `cublasLt` symbol, and the negative case
   fails.
5. The hardware gate passes on a fresh Hopper run in less than 1 second.
6. The fault injection makes the gate exit with code 2.
7. The gate stamp blocks a second run on a resume.

## 8. Risks and non-goals

- **The 1e-2 fp8 bound is wide.** A wide bound can hide a fault. Counter it with
  the narrow cases in steps 5 and 6 of the gate. Prefer a known answer over a
  wide tolerance on a large case.
- **The emulator can diverge from the device.** `simulator.md` section 9.4 marks
  the engine experimental. It never replaces the oracle.
- **A new CUDA version adds a `cublasLt` symbol.** Pin the toolkit. Let the
  symbol gate fail on a mismatch.
- **Non-goal: a functional PTX interpreter.** See
  [simulator.md](simulator.md) section 14.
- **Non-goal: a tensor-core accumulation model.** Only hardware has it.

## 9. Open questions

1. Does the gate stamp live in the checkpoint container, or in a side file? By
   decision, the gate always runs on a fresh run; there is no stamp.
2. Is one fp8 attention shape enough, or is one shape per window pattern needed?
3. Does the `fp8-full` suite run at every merge, or only at a Wave boundary?
4. When does the fp8 backward land? The forward-only cut is in; the backward
   needs the E5M2 operand roles and the reuse rules
   ([fp8-training.md](fp8-training.md) section 7).

## See also

- [simulator.md](simulator.md) — the S0 simulator and its engines.
- [simulator-precision.md](simulator-precision.md) — the simulator at bf16 and
  fp8.
- [fp8-training.md](fp8-training.md) — the fp8 design.
- [numerics-integration.md](numerics-integration.md) — the trace and the
  tolerances.
- [testing.md](testing.md) — the test tiers.
- [node-h100.md](node-h100.md) — the H100 run book.
