# Campaign plan: the hand tensor-core GEMM and its simulator leg

Status: campaign plan, not a reference document. Its durable sections move into
[docs/kernels.md](kernels.md) Part 3 and [docs/simulator.md](simulator.md) when
the last row lands. The retirement rule is at the end of
[docs/README.md](README.md).

This plan implements [kernels.md](kernels.md) Part 3, "The hand-written
tensor-core GEMM". It also removes the blocker that Part 3 carries. Today the
simulator cannot run a tensor-core kernel. A hand GEMM would therefore have one
correctness leg, and that leg would be a real card. The plan adds the missing
vocabulary to the emulator first. The GEMM then has an off-device gate.

## 1. Adopted decisions

| # | Decision | Value |
|---|---|---|
| D1 | The card for the rate gates | A hosted Hopper job, or a Hopper host. Nothing on this machine can gate the work |
| D2 | The end state for cuBLAS | Level 2: every closed-list shape takes the hand path, so no batched Muon call and no Linear call reaches cuBLAS. Library removal is a later decision |
| D3 | The fp8 kernel | A second campaign, after the bf16 result. The fp8 **emulation** ships here, because it is the simulator leg |
| D4 | Ownership | One Kernel agent owns `backends/cuda/gemm/**` on branch `agent/gemm`. The Simulator workstream owns `tools/cuda_sim/**`. The `backends/cuda/gemm.cu` diff is one integrator merge |
| D5 | The first promotion target | The 1280 by 1280 Muon group. The 1280 by 5120 group is bandwidth bound, so its rate would flatter the kernel |

## 2. Findings this plan must answer

| # | Finding | Where |
|---|---|---|
| F1 | The `t4` entry in `backends/cuda/caps.json` declares `supports_bf16: false`, and the host card is a GTX 1080 Ti at compute 6.1 | The bf16 and fp8 rate gates need D1 |
| F2 | `nvcuda::wmma` has no fp8 fragment type, so fp8 must use `mma.sync` inline PTX | The fp8 kernel needs a project wrapper, and the wrapper needs a host arm |
| F3 | `cp.async` does not exist on `sm_75`, the tree's floor | The copy stage needs a second path, and the emulator needs the intrinsics mapped |
| F4 | `tools/cuda_sim/emu/cuda_emu.h` maps no tensor-core instruction, and no stub exists for `<mma.h>`, `<cuda_pipeline.h>`, or `<cuda_bf16.h>` | This plan, Part A |
| F5 | `backends/cuda/gemm/**` has no row in `AGENTS.md` section 1 | The architect adds the row before work fans out (D4) |

## 3. The shape of the work

Two workstreams, one integration point, no shared file.

```
Part A  tools/cuda_sim/**            the Simulator workstream
Part B  backends/cuda/gemm/**        branch agent/gemm
Merge   backends/cuda/gemm.cu        integrator
```

Part A comes first. A tensor-core kernel written before the emulator can run it
has no gate on this machine, and the first defect would be found on a paid card.

---

# Part A. The simulator learns the tensor core

## A.0 What exists, and what is missing

The emulation engine runs the real `.cu` sources on the host. It compiles them
with the stub headers in `tools/cuda_sim/emu/`, which win over the toolkit
because `-Itools/cuda_sim/emu` is on the compile line and the emulated build
carries no toolkit include path at all. The engine turns
one block into one cooperative fiber per CUDA thread, so warp cooperation is
representable.

The mapping covers `__shared__`, `__syncthreads`, the shuffles, the atomics, the
`__fmul_rn` family, and, since Part 7 of [simulator.md](simulator.md), the fp8
conversion intrinsics. It covers no matrix instruction and no asynchronous copy.

Four headers are missing.

| Header | Why it is missing today |
|---|---|
| `<mma.h>` | `attention_bf16.cuh` includes it only under `defined(__CUDACC__)`, so the host pass never asked for it |
| `<cuda_pipeline.h>` | Same guard, same reason |
| `<cuda_bf16.h>` | Same guard. The emulator maps `__half` in `cuda_emu.h`, but not `__nv_bfloat16` |
| A `<mma.h>` form for fp8 | The vendor header has no fp8 fragment, so a stub alone cannot help |

## A.1 The three stub headers

Three new files in `tools/cuda_sim/emu/`, each following the pattern of
`cuda_runtime.h` and `cuda_fp8.h`: a short header that includes the documented
mapping.

| File | Contents |
|---|---|
| `cuda_bf16.h` | `__nv_bfloat16` as `nanochat::Bf16`, `__nv_bfloat162` as a two-element struct, and the conversions the tile uses: `__float2bfloat16`, `__bfloat162float`, `__floats2bfloat162_rn`, `__low2float`, `__high2float` |
| `cuda_pipeline.h` | Section A.3 |
| `mma.h` | Section A.2 |

`cuda_bf16.h` maps to `nanochat::Bf16` and not to a new host type. The rounding
then comes from one implementation in `include/nanochat/tensor.h`, which is the
same argument `cuda_fp8.h` makes for `nanochat/fp8.h`.

## A.2 `<mma.h>`: the fragment vocabulary

The header provides `namespace nvcuda::wmma` with the subset the tile uses.

```cpp
// tools/cuda_sim/emu/mma.h  (host only)
namespace nvcuda { namespace wmma {
enum class layout_t { mem_row_major, mem_col_major };
constexpr layout_t mem_row_major = layout_t::mem_row_major;
constexpr layout_t mem_col_major = layout_t::mem_col_major;
struct row_major_tag {};  inline constexpr row_major_tag row_major{};
struct col_major_tag {};  inline constexpr col_major_tag col_major{};

enum class use_t { kMatrixA, kMatrixB, kAccumulator };
inline constexpr use_t matrix_a = use_t::kMatrixA;
inline constexpr use_t matrix_b = use_t::kMatrixB;
inline constexpr use_t accumulator = use_t::kAccumulator;

template <use_t Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
  T storage[M * N];   // the whole 16 by 16 tile, in every fiber
};
}}  // namespace
```

Four functions, with the vendor signatures:

| Function | Host behaviour |
|---|---|
| `fill_fragment(f, value)` | Fill the whole tile |
| `load_matrix_sync(f, ptr, ld, layout)` | Read 16 by 16 elements with the given layout and leading dimension |
| `mma_sync(d, a, b, c)` | `d = a * b + c` over the 16 by 16 by 16 micro-tile, in fp32 |
| `store_matrix_sync(ptr, f, ld, layout)` | Write 16 by 16 elements with the given layout and leading dimension |

**The one emulation rule this plan introduces.** Each fiber holds the whole
micro-tile. Every fiber of a warp therefore computes the whole micro-tile, and
every fiber stores the same values. The arithmetic is correct and the result is
deterministic. The cost is one factor of the warp width, 32, over the real
arithmetic.

Consequences, which the plan accepts and records:

- The engine checks the staging layout, the tile addressing, the padding, the
  leading dimension, the epilogue, the accumulate order, and the store. Those
  are the defects a new GEMM actually has.
- The engine cannot check the fragment ownership layout, a bank conflict, the
  register count, the occupancy, or the rate. Those stay on device.
- **A kernel may only pass a fragment to `fill`, `load`, `mma`, and `store`.** A
  kernel that reads one element of a fragment depends on the vendor layout, and
  the emulator would agree with it by accident. This rule is the price of the
  whole-tile rule, so it goes into `docs/kernels.md` and the `backends/cuda`
  kernel conventions.

Rejected alternative: exact per-lane ownership, with a cross-lane exchange in
the warp shuffle window. It costs one barrier per fragment operation, it needs a
mapping for each use, and it proves no more than the whole-tile rule. It also
cannot prove the vendor layout either, because the emulator would own that
layout. Do not write it.

## A.3 `<cuda_pipeline.h>`: the asynchronous copy

| Intrinsic | Host behaviour |
|---|---|
| `__pipeline_memcpy_async(dst, src, bytes)` | `std::memcpy`, complete at the call |
| `__pipeline_commit()` | Nothing |
| `__pipeline_wait_prior(n)` | Nothing |

The reason: a real wait only delays visibility. Completing every copy at the
call is one schedule the device allows, so the emulated answer is legal. The
engine checks the staging arithmetic and the tile addressing, and it does not
check the overlap. Record that limit with the table.

**Design rule for Part B.** The tile uses the `__pipeline_*` intrinsics and not
raw `asm volatile("cp.async...")`. Inline PTX does not compile on the host, so a
raw form would remove the simulator leg in the same commit that adds the kernel.

## A.4 The fp8 matrix op, and where the wrapper lives

fp8 has no `wmma` form (F2), so the kernel needs a project wrapper. The wrapper
must hold both arms in one header, because production code cannot include
`tools/**`:

```
backends/cuda/kernels/mma_ops.cuh        backend-internal, like device_utils.cuh
  #if defined(__CUDACC__)
    // Device: asm volatile("mma.sync.aligned.m16n8k32..." ...)
    // operands e4m3 and e5m2, fp32 accumulator
  #else
    // Host (emulator): a scalar loop over 16 by 8 by 32,
    // decoding through nanochat::Fp8E4M3Decode / Fp8E5M2Decode
  #endif
```

The host arm calls `include/nanochat/fp8.h`, which the CPU backend and the
emulator already share. One rounding function then serves every consumer, which
is the invariant of [simulator.md](simulator.md) Part 7.

The wrapper ships here and is proven here, on tiny shapes, against the shared
rounding function. The production fp8 GEMM that uses it is D3, a later campaign.
This split is honest: A.4 proves the instruction mapping, and it claims nothing
about a kernel that does not exist yet.

## A.5 Build and suite wiring

| File | Change |
|---|---|
| `tools/cuda_sim/emu/BUILD.bazel` | Add `cuda_bf16.h`, `cuda_pipeline.h`, `mma.h` to `cuda_headers`. Add `//backends/cuda/kernels:mma_ops.cuh` to the same list. Add one genrule per emulated GEMM source, a plain `cp`, because the tile uses static `__shared__` |
| `tools/cuda_sim/emu/emu_prelude.h` | One rename for the backend-internal launcher, `LaunchGemmTile` to `EmuLaunchGemmTile`, beside `MuonFusedUpdate` |
| `tools/cuda_sim/emu/emu_kernels.h` | Declare the emulated launcher with the seam's parameter types |
| `tools/cuda_sim/emu/gemm_emu_test.cc` | A GEMM section: the emulated launcher against a host reference. It lives beside the emulated source and not in `tests/kernel_emu_test.cc`, because the tile is bf16 by construction and that test is fp32 |
| `python/nanochat_cpp/cli.py` | `SIM_SUITES` gains `"gemm": ["//tests:kernel_emu_test"]`, and `"all"` gains the same target once. Architect-owned: request the edit |
| `docs/simulator.md` | Section 9.2 gains the four rows above and the fp8 wrapper. Section 9.4 gains the fragment-ownership limit and the warp-width cost |

`check_symbols.py` needs no change. A tensor-core instruction is a device
instruction, and the shared-memory request above 48 KiB uses
`cudaFuncSetAttribute`, which `tools/cuda_sim/interposer.cc` already answers.

The emulated cases keep `m` and `n` at or below 128 and `k` at or below 64. The
warp-width factor then keeps the whole section under two seconds, which is the
`size = "small"` budget of `//tests:kernel_emu_test`.

## A.6 The simulator gate

| Gate | Pass condition | Command |
|---|---|---|
| SA | The emulated GEMM agrees with the host reference on tiny shapes, within the kernel tolerance | `tools/nanochat simulate --device h100 --suite gemm --precision bf16` |
| SB | The same at fp8 precision, and the wrapper's host arm agrees with the shared rounding function | `tools/nanochat simulate --device h100 --suite gemm --precision fp8` |
| SC | The default CPU loop stays green, and `kernel_emu_test` stays under its size budget | `tools/nanochat test //tests:kernel_emu_test` |

SA and SB are CPU only. They need no card, no driver, and no queue, so they ride
the merge gate in `AGENTS.md` section 5.5.

## A.7 What Part A does not do

- No rate. An emulated kernel runs one block at a time, so its wall clock says
  nothing about a card.
- No occupancy, no register count, no bank conflict, no warp divergence.
- No cooperative launch. The engine runs blocks one at a time, so a grid barrier
  stays inexpressible. A persistent GEMM tile is therefore outside this leg.
- No attention. Section A.9 holds that as an optional stage.

## A.8 Order of work

| Step | Work | Gate |
|---|---|---|
| A1 | The three stub headers, with `<mma.h>` complete for bf16 | A unit case: `load`, `mma`, `store` round trip on a 16 by 16 by 16 tile |
| A2 | The fp8 wrapper's host arm and its unit case | The wrapper against the shared rounding function |
| A3 | The emulated GEMM source in the build, with the prelude rename | `tools/nanochat build //tests:kernel_emu_test` |
| A4 | The GEMM section of the emulated test | SA, SB, SC |

A1 and A2 are small and independent. A1 unblocks Part B, W1.

**A1 landed.** `tools/cuda_sim/emu/mma.h`, `cuda_bf16.h`, and
`cuda_pipeline.h` hold the mapping. `//tools/cuda_sim/emu:mma_emu_test` is the
check, and it passes with eleven checks:

| Case | What it proves |
|---|---|
| Four operand-layout combinations, through the block runtime | The fragment mapping and the fiber scheduler agree with a row-major reference |
| The same at one thread | No part of the mapping depends on the warp width |
| A ragged leading dimension of 24 with a poisoned pad | A wrong stride reads the poison and fails |
| `mma_sync` with `d` and `c` the same object, twice | The in-place accumulation the attention tile uses |
| An accumulator read back column-major, then accumulated | The `beta` term and the rescale between two micro-tiles |
| The bf16 conversions, and the copy with a zero fill | The two neighbouring stubs |

The operands are multiples of a power of two, so every product and every 16-term
sum is exact in fp32. The check therefore uses a tolerance of 1e-6 and reports a
mapping defect rather than a rounding question. Three mutations confirm the
test has power: a swapped layout, a leading dimension that is ignored, and a
dropped accumulator term each fail the check.

Two facts moved during A1:

1. `emu_runtime.cc` is now its own `cc_library`, `:emu_runtime`. The mapping
   test needs the scheduler and the runtime without the kernel set.
   `:emulated_kernels` depends on it, so no kernel target changed.
2. The test compiles through the same two flags as an emulated kernel source,
   `-Itools/cuda_sim/emu` and `-include emu_prelude.h`. A test that compiled
   against a different header would prove nothing about a kernel.

**A2 landed.** `backends/cuda/kernels/mma_ops.cuh` holds the fp8 micro-tile of
16 by 8 by 32, in shared-memory layout, with one scalar arm that decodes
through `include/nanochat/fp8.h`. `CheckFp8Mma` in `mma_emu_test` covers the
four format pairs and the in-place accumulation, and it decodes with its own
mapping so a wrong arm of the header cannot move the expectation. A mutated
header that ignored the format fails four cases.

The `mma.sync.aligned.m16n8k32` arm is **absent on purpose**. An instruction
path written without an `sm_89` card to run it is a guess that reads like a
result. It lands in the fp8 campaign (D3) beside the scalar arm, which stays as
the fallback for a card below `sm_89`.

**A3 landed.** `backends/cuda/gemm/` holds the tile: `gemm_tile.cuh` (geometry,
the staged layout, and the two staging helpers), `gemm_launch.h` (two
launchers), `gemm_bf16.cu` (the kernel, the epilogue, and the host launcher),
and `BUILD.bazel`. `emu_prelude.h` renames both launchers, `emu_kernels.h`
declares the emulated pair, and `tools/cuda_sim/emu/BUILD.bazel` adds the
emulated source as `:emulated_gemm`.

Four deviations from sections B.3 and B.4, each with its reason:

| Deviation | Reason |
|---|---|
| One staged buffer and a barrier per slice, not two buffered stages | The arithmetic gate does not need the overlap, and the emulator cannot check it. The stage layout is what a faster copy must preserve, and `gemm_tile.cuh` says so |
| A synchronous element-wise gather, not `cp.async` | The first cut needed the arithmetic, not the overlap. Section B.4 below lands the two-stage pipeline |
| The vendor fragment type is `__nv_bfloat16`, not `ComputeType` | The real `wmma` specializes on `__nv_bfloat16` only, and the tree's `Bf16` is a different type. `attention_bf16.cuh` already views the operand memory this way. The emulator maps the two to one type |
| The kernel is compiled only when `NANOCHAT_PRECISION` is bf16 or fp8 | A bf16 `wmma` micro-tile needs a tensor core with bf16. The guard is the precision define, which is the pair `attention_bf16.cuh` uses |

The file compiles both ways. The default fp32 build compiles it to nothing. The
device build at bf16 and `sm_90` reports:

| Measure | Value |
|---|---|
| Registers per thread | 98 |
| Static shared memory per block | 32,768 B |
| Spill | 0 bytes |
| Instantiations | the bf16 store and the fp32 store report the same |

The shared budget is under the 48 KiB static limit, so the tile needs no
`cudaFuncSetAttribute` and no dynamic shared memory, and the emulator needs no
`extern __shared__` rewrite. The 32 KiB figure holds two blocks per SM on
`sm_90` and on `sm_75`, so the shared window, and not the register count, is
what bounds the occupancy. That is a sweep knob for B4, not a verdict.

**A4 landed.** `//tools/cuda_sim/emu:gemm_emu_test` runs the real kernel through
the block runtime. It covers seven cases: the four operand layouts, a ragged
shape with a k tail and an alpha and beta term, a batch of two, and the fp32
weight-gradient store. It also checks that a call the tile cannot express is
refused and writes nothing.

The case data has one rule worth naming: the operand multiplier must be
coprime with the modulus. A first version multiplied the flat index by seven
and took it modulo seven, which made one operand a constant matrix. That
constant hid a dropped sub-tile offset and the check passed. Five is coprime
with both moduli, and every mutation below then fails.

| Mutation | Caught by |
|---|---|
| A dropped warp column offset on B | Every staged case, at 155 times the bound |
| A dropped warp row offset on A | Every staged case |
| The alpha term dropped | The alpha and beta case |
| The beta term dropped | The alpha and beta case and the batch case |
| The batch base offset dropped | The batch case, at 17 times the bound |
| A swapped transpose stride pair | Every staged case |
| The fp32 store routed to the bf16 kernel | The weight-gradient case |
| The k-tail zero removed from both operands | Three cases, one at 4.5e31 |

**One limit, recorded.** A single removed guard is not observable. Three cases
mask it. The k-tail zero on A is masked by the zero on B. The k-tail zero on B
is masked by the one on A. The column bound on B is masked by the epilogue's own
bound.

The result is correct either way, so the emulator cannot see one removal. It
does see both k-tail zeros removed at once. The guards are therefore defensive,
and they stay. A masked read past an operand is a fault risk beside a wrong
number, and only an address sanitizer or a card would show it.

**Where the campaign stands.** A1 to A4 are done, and with them the off-device
half of B2: the kernel's arithmetic and its full contract are checked by
`tools/nanochat test` on a machine with no card. The rate gates (G3 and G4) and
G2 on hardware need D1. B4's schedule sweep and B5's promotion are open, as is
the fp8 kernel (D3).

## A.9 Optional stage: the tensor-core attention tile

`attention_bf16.cuh` guards its whole body on `defined(__CUDACC__)`. A simulated
bf16 windowed forward therefore takes the cuBLAS path. Remove the guard and the
tile becomes emulable with A1, because it holds its fragments opaquely. It
stores score fragments to a per-warp scratch tile. It then reloads them as
accumulator fragments.

That is a separate change with its own gate, and it touches a family with a
promotion decision behind it. Do it after A4, and only with the owner of
`backends/cuda/kernels/attention.cu`. It is not a prerequisite for Part B.

---

# Part B. The hand tensor-core GEMM

## B.1 What Part 3 asks for

| Item | Text of Part 3 |
|---|---|
| Shapes | A closed list: the Muon Polar Express groups, and the Linear forward, dgrad, and wgrad shapes of `python/nanochat_cpp/pipeline.py` `PROFILES` |
| Kernel | Generalise `attention_bf16.cuh` into a tiled GEMM under `backends/cuda/gemm/`, with double-buffered stages and 16 by 16 by 16 bf16 micro-tiles with fp32 accumulators |
| Contract | Every `GemmParams` field: transpose flags, batch, leading dimensions, strides, alpha and beta, and the fp32 output of `GemmWgrad` |
| Dispatch | A predicate in `backends/cuda/gemm.cu`. cuBLAS stays the fallback. No group slower |
| Gates | 0.70 of the cuBLAS rate to promote one row, 0.85 for a shape replacement |
| Seam | Unchanged: no new entry point, no capability bit, no version bump |

The readings of Part 3 that this plan corrects:

| Text | Correction |
|---|---|
| "`cp.async` double-buffered stages" | `cp.async` needs `sm_80`. The tree's floor is `sm_75`, so the copy stage needs two paths (F3) |
| "The fp8 path adds the E4M3 and E5M2 fragments" | `wmma` has no fp8 fragment (F2). fp8 is a second microkernel family behind the A.4 wrapper, and it is D3 |
| "Mirror `muon_fused_gemm_test.cc`" | That test is the fp32 prototype in `dev/kernels/`. It is the model for the shape list and the comparison, and it is not the model for the schedule: Part 1 measured its schedule as the loss |
| "A Hopper-native path needs WGMMA and TMA in the simulator first" | TMA is a runtime API, so the interposer can answer it. WGMMA is an instruction, so it falls under A.2. Both stay out of this plan |

## B.2 Package layout

```
backends/cuda/gemm/BUILD.bazel
backends/cuda/gemm/README.md            the schedule, the gates, the measurements
backends/cuda/gemm/gemm_tile.cuh        the block tile, the shared stage, the copy path
backends/cuda/gemm/gemm_bf16.cu         the kernel, the epilogue, the launcher
backends/cuda/gemm/dispatch.h           backend-internal, like rms_norm.h
backends/cuda/gemm/dispatch.cc          the shape table and the predicate
backends/cuda/gemm/gemm_tile_test.cc    T0: the host-side plan, bounds, epilogue expression
backends/cuda/gemm/gemm_gpu_test.cc     T1 and T2: hand against cuBLAS, full contract
backends/cuda/gemm/gemm_benchmark.cc    T3: rate against cuBLAS over the closed list
```

`backends/cuda/gemm.cu` keeps both seam entries and the `cublasLt` plan cache.
It resolves the parameters once, asks `dispatch.h` for a verdict, and calls the
hand launcher or the existing cuBLAS path. Its diff stays small, which is what
makes D4 a single merge.

## B.3 The tile

One conservative first schedule that fits a Turing shared budget, then a sweep.

| Knob | First value | Reason |
|---|---|---|
| Block tile | 128 by 64 | About 24 KiB of staged shared memory over two stages, so two blocks fit in 64 KiB |
| Threads | 128, four warps in a 2 by 2 grid | Warp tile 64 by 32 gives eight accumulator fragments, 64 fp32 registers per thread |
| Stage depth | 32, double buffered | Two 16-deep micro-steps per stage. Part 1's 8-deep stage was one of the three measured losses |
| Padding | Eight elements on the contiguous extent | Every staged segment stays 16-byte aligned, and the bank conflict goes |

The sweep is a stage, not a tuning afterthought. Part 1 section 9 names the three
numbers that lost to cuBLAS: the 8 by 4 micro-tile, the 8-deep stage, and the
80-register block. Change all three before any gate verdict. A second schedule
for `sm_90` (128 by 128, 256 threads, eight warps, a 32-deep stage) follows the
first gate.

## B.4 The copy stage

One header function, two arms, chosen by `__CUDA_ARCH__`:

- `sm_80` and newer: 16-byte asynchronous copies through the `__pipeline_*`
  intrinsics, two stages, and the swizzled shared address so the operand needs
  no second transpose.
- `sm_75`: the same shared layout, filled by synchronous 16-byte loads.

**Landed.** The two-stage pipeline is in `gemm_tile.cuh`. The two arms sit
behind one `CopyChunk`, one `CommitStage`, and one `WaitStage`, so the kernel has
no second body. A staged row moves in sixteen-byte chunks when its k index steps
by one element. It moves one element at a time otherwise: the transposed form,
the ragged edge, and the k tail. A chunk also needs both ends aligned to sixteen
bytes, so a row whose global end is not falls back to the element path. That is
the 5 by 12 Muon group and not a training step, whose hidden widths are multiples
of eight.

The sweep is three schedules, and all three fit the 48 KiB shared window, so none
needs `cudaFuncSetAttribute` and the emulator needs no dynamic shared memory:

| Schedule | Tile | Warps | k depth | Registers | Shared |
|---|---|---|---:|---:|---:|
| `128x64x32` | 128 by 64 | 2 by 2 | 32 | 96 | 32,768 B |
| `64x64x32` | 64 by 64 | 1 by 2 | 32 | 128 | 19,456 B |
| `64x64x64` | 64 by 64 | 1 by 2 | 64 | 128 | 36,864 B |

`NANOCHAT_GEMM_SCHEDULE` names one of the three, and the benchmark sweeps a row
over all three. A row of the closed list owns its schedule, so a shape keeps the
one its measurement chose.

**What the emulator proves here, and what it does not.** The emulated run covers
the staged layout, the zero fill of a ragged edge and a k tail, and the chunk and
element paths. It also covers the buffer index of the pipeline, and that the
three schedules agree with one reference. A wrong buffer index fails every case
whose k spans more than one slice. It cannot cover
the wait itself: the emulator's `__pipeline_wait_prior` is a no-op, so a missing
or too weak a wait is invisible here, and one call carries that risk. The loop
commits one group per iteration and waits for all but the newest, which is the
form that needs no branch on the last slice.

The two arms must leave the same shared contents. One emulated case compares
them, which A.3 makes cheap.

## B.5 The epilogue and the contract

The accumulator tile is fp32 in every case. One epilogue serves the whole
contract:

1. Read both operands through the four transpose combinations and the three
   leading dimensions.
2. Accumulate in fp32, then apply alpha and beta. Read `C` only when `beta` is
   not zero.
3. Store `ComputeType` for `Gemm`, and `float` for `GemmWgrad`.

The batch rides on a flat block index. The defaults for the three leading
dimensions and the three strides come from the same rules `GemmImpl` already
applies, so the two paths cannot disagree about a default.

## B.6 The dispatch

A static table in `dispatch.cc`: one row per shape of the closed list, each with
an `enabled` bit that starts false. A shape takes the hand path only when its
bit is true. Two consequences follow.

1. No run can change behaviour without a committed measurement.
2. The A/B run needs no rebuild. One environment variable, `NANOCHAT_GEMM_LIB`,
   selects `hand`, `cublas`, or `auto`.

The predicate refuses any shape it does not hold exactly. It also refuses a
small `m`: a batch-1 decode call with `m` of 1 wastes a tensor-core tile, and
`src/generate.cc` is latency bound.

## B.7 Work packages

| # | Work | Gate | Command |
|---|---|---|---|
| B0 | The host scaffolding: `GemmPlan` as pure host code, with the tile choice, the rounding up, the bounds, the epilogue expression, and the stride inference | Passes on this machine | `tools/nanochat test //backends/cuda/gemm:gemm_tile_test` |
| B1 | The `sm_80` kernel, forward only, one shape, no batch, no transpose, `beta = 0` | SA over the emulated leg, then agreement with cuBLAS at 1e-3 absolute | `tools/nanochat simulate --device h100 --suite gemm --precision bf16` |
| B2 | The full contract: four transpose forms, batch, three leading dimensions, strides, alpha and beta, and the fp32 store | cuBLAS is the reference on a card, over the same cases the emulator runs, and every schedule | `tools/nanochat test --gpu --config=hopper //backends/cuda/gemm:gemm_gpu_test` |
| B3 | The dispatch: the closed list, the verdict, and the `NANOCHAT_GEMM_LIB` switch | The T0 gate, and no shape changes hands while its bit is inert | `tools/nanochat test //backends/cuda/gemm:dispatch_test` |
| B4 | The schedule sweep: the two-stage `cp.async` copy, the three schedules, and the record of what each one measures | The code half is landed and gated off device. The record needs 0.85 on the D5 group, then on a named shape set, so it needs D1 | `tools/nanochat bench -- bazel-bin/backends/cuda/gemm/gemm_benchmark --schedule all --json --out /tmp/gemm-sweep.json` |
| B5 | The measurement and the promotion: one report row per shape, the tile against cuBLAS, then the bits that pass | 0.70 on one Muon row and one Linear shape, 0.85 for a replacement, and no enabled shape slower than cuBLAS | `tools/nanochat bench -- bazel-bin/backends/cuda/gemm/gemm_benchmark --json --out /tmp/gemm-bf16.json` |

**B3 landed.** The dispatch is one verdict between the seam and cuBLAS, and it
changes nothing until a measurement turns a bit on.

| File | What |
|---|---|
| `backends/cuda/kernels/muon_gemm_shape.h` | The three Polar Express shapes of one group. `muon.cu` issues them, and the table reads the same three, so the call sites and the closed list cannot drift |
| `backends/cuda/gemm/gemm_shape.h` | `ResolvedGemm`, `ResolveGemm`, `IsTileCallable`, `SameShape`. One resolved form serves the table, the launcher, and the kernel argument |
| `backends/cuda/gemm/dispatch.{h,cc}` | The closed list, the verdict, the switch, and the row accessors |
| `backends/cuda/gemm.cu` | One call before the cuBLAS path. A refused call, an inert row, and an unserved store type all leave cuBLAS in place |
| `backends/cuda/gemm/dispatch_test.cc` | The T0 gate |
| `backends/cuda/gemm/gemm_benchmark.cc` | The rate gate, which measures the seam under both values of the switch |

Three findings changed the design of this section.

1. **Thirteen rows, not fifteen.** Two products of the five groups are one
   shape: `A2` of a square group is also its apply product, and the `1280x5120`
   square is the `5120x1280` square. A key is one call, so a shared shape is one
   row and its name lists every product it serves. The benchmark therefore
   measures each distinct shape once.
2. **`muon.cu` owns the shape derivation, not the table.** The three products
   were spelled out twice, once at the call sites and once in a table that
   mirrors them. They now come from one header, and the test compares the table
   against that header. `muon.cu` lost about fifty lines of shape setup and
   issues three one-line calls.
3. **The gate needs no device.** `ResolvedGemm` holds extents, strides, flags,
   and two scalars, and no element type, so the table and the seam's default
   rules are checked on this machine. The test also re-checks the defaults
   `GemmImpl` applies, because only one of the two ever runs on a card.

The test covers the list, the anti-drift comparison against `muon_shape`, and a
row that selects its own key and not a near miss on any field. It also covers
the switch, both environment variables, the schedule list, and the seam's
default rules.

The benchmark calls the seam and not the launcher, so it measures the wiring a
training step takes. It has not run: it needs a card and a bf16 build (D1).

Open in B5: the Linear rows join the table from a measurement. The five Muon
groups are measured, and a bit is turned on only where the fraction gate passes.

**The device gate exists.** `backends/cuda/gemm/gemm_gpu_test.cc` is the twin
of the emulated check, and cuBLAS is its reference. Both sides accumulate bf16
products in fp32, so only the order of the sum differs. The tolerance is
therefore tight: one bf16 unit in the last place, and 1e-5 relative for an fp32
store.

It drives the launcher and not the seam. Every row of the shape table is inert,
so a call through `Gemm` would compare cuBLAS with cuBLAS.

| Case | What it covers |
|---|---|
| Four operand layouts at 4 by 3 by 5 | The stride pairs and the micro-tile addressing |
| A batch of three | The per-batch base and stride arithmetic |
| Ragged 45 by 23 by 50 | The predicated edge and the k tail |
| 128 by 64 by 32 | Exactly one base tile |
| 200 by 130 by 96 | More than one tile on both extents, and more than one slice |
| Weight gradient, 96 by 64 by 40, batch of two, fp32 out | The second store path at more than one tile |
| 640 by 640 by 96 on each of the three schedules | The sweep on a card |

Two refusals stand in front of it, and the run states which one it took. An
fp32 build prints that the tile needs bf16. A card without a bf16 tensor core
names itself and exits instead of running a tile that cannot work there. On this
machine the second is what a run prints.

The command carries `--config=hopper`, because the tile is compiled only in a
bf16 build.

## B.8 The gates

| Gate | Kind | Pass condition |
|---|---|---|
| G1 | Arithmetic, off device | SA and SB stay green, so every change is checked before it reaches a card |
| G2 | Correctness, device | `gemm_gpu_test`: every contract case matches cuBLAS, on a card, within one bf16 unit for a bf16 store and 1e-5 for an fp32 store |
| G3 | No group slower | Every enabled shape is at least as fast as cuBLAS at the same precision. One slower shape turns its bit off |
| G4 | Promotion | 0.70 of the cuBLAS rate for a first row. 0.85 for a shape replacement |
| G5 | Seam | `include/nanochat/kernels.h` has no diff, `caps.json` gains no bit, and `tools/nanochat simulate --device h100` stays green |
| G6 | Regression | `//backends/cuda:cuda_runtime_gpu_test`, both chain legs, and `//tests:train_parity_cuda_test` stay green |

G5 is nearly free by construction. A launch and a shared-memory request need no
new runtime name, so `tools/cuda_sim/check_symbols.py` needs no change.

## B.9 What a cuBLAS replacement means

| Level | Meaning | Entry condition |
|---|---|---|
| L1 | The closed list takes the hand path, and cuBLAS serves every other shape | G4 at 0.70, and G3 |
| L2 | Every closed-list shape takes the hand path, so no batched Muon call and no Linear call reaches cuBLAS (D2) | G4 at 0.85 on the whole list |
| L3 | `@cuda//:cublas` leaves the build | A new closed list that also covers the attention products in `attention.cu`, the scaled fp8 path, and the batch-1 decode shape |

Stop at L2. `attention.cu` and `GemmScaledFp8` both own cuBLAS calls that Part 3
does not list, so L3 would remove nothing without them.

## B.10 Risks and stop conditions

| Risk | Stop condition |
|---|---|
| The hand rate stays under 0.70 on every group, as Part 1 measured for fp32 | If no schedule reaches 0.70 after B4, stop and record a negative result in Part 3. Keep cuBLAS |
| No Hopper host is available (F1) | Run `tools/nanochat doctor` on the target host before B1. The simulator leg reduces the cost of the wait, and it does not remove it |
| The emulator agrees with a kernel that a device rejects | The fragment-ownership rule of A.2. A kernel that breaks it is a review defect, so name the rule in `backends/cuda/kernels/README.md` |
| An epilogue or stride difference that only appears at ragged tiles | B2 covers ragged tiles and a non-dense leading dimension before any benchmark runs |
| Two writers in `backends/cuda/gemm.cu` (F5) | The `AGENTS.md` row lands before work fans out |

## B.11 Documents this plan touches

| File | Change | Owner |
|---|---|---|
| `docs/kernels.md` Part 3 | The text becomes "implemented" or "measured, rejected". The durable rows move to `backends/cuda/gemm/README.md` | Architect |
| `docs/kernels.md` section Compute strategy | The GEMM rule keeps its reason and gains the measured fraction | Architect |
| `docs/kernels.md` section Shared device helpers | `mma_ops.cuh` joins the list | Architect |
| `docs/simulator.md` sections 9.2 and 9.4 | The new mapping rows and the fragment-ownership limit | Architect |
| `docs/model.md` section The cuBLAS seam | The table gains the dispatch row | Architect |
| `docs/baselines.json` | One row per measured shape | Architect |
| `docs/testing.md` | The emulated GEMM leg joins the S0 suite list | Architect |
| `backends/cuda/gemm/README.md` | New: the schedule, the gates, the numbers | `agent/gemm` |
| `backends/cuda/kernels/README.md` | The fragment-ownership rule | The kernels owner |
| `AGENTS.md` section 1 | The ownership row for `backends/cuda/gemm/**` and `backends/cuda/gemm.cu` (F5) | Architect |
| `BUILD.bazel` root | One label, `//backends/cuda/gemm:all` | Build agent |

## B.12 Order of work

```
A1 ──> A2 ──> A3 ──> A4 ──> B1 ──> B2 ──> B3 ──> B4 ──> B5
                                                    
B0 runs beside A1 and needs no device.
```

A1 unblocks B1. A4 unblocks the off-device half of B2. B3 needs D1, because it
times a card. B4 decides whether B5 ships at all.
