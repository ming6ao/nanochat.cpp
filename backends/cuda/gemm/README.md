# The tensor-core GEMM

This package holds two bf16 mainloops behind the seam's `Gemm` and `GemmWgrad`.
Each one implements the full `GemmParams` contract: four operand layouts, a
batch, three leading dimensions, three strides, an alpha and beta epilogue, and
the fp32 store of `GemmWgrad`.

| Mainloop | What | Section |
|---|---|---|
| `cutlass_bf16.cu` | The Hopper mainloop that NVIDIA CUTLASS supplies, and the default path | 16 |
| `gemm_bf16.cu` | The hand-written warp-level tile | 2 to 15 |

**Sections 2 to 15 are the record of the hand tile, and the measurement says no
shape takes it.** Every row of the closed list is inert, so the hand tile never
serves a training step, and section 8 keeps that negative result. It stays for
three reasons. It is the leg the emulation engine runs. It is the reference the
Hopper mainloop is compared against. And it serves the shapes smaller than one
tile.

## 1. The files

| File | What |
|---|---|
| `gemm_tile.cuh` | The staged operand tile: the two copies, the k slice, the pipeline |
| `gemm_plan.h` | The four schedules, the staged layout math, and the sweep's names |
| `gemm_bf16.cu` | The kernel, the epilogue, and the launcher |
| `gemm_launch.h` | The two backend-internal launchers |
| `gemm_shape.h` | `ResolvedGemm`, `ResolveGemm`, the tile's rule, and the Hopper mainloop's rule |
| `cutlass_bf16.cu` | The eight CUTLASS kernels of the Hopper mainloop and the launch behind them |
| `cutlass_launch.h` | The two backend-internal launchers of that mainloop |
| `linear_shape.h` | The three calls of one Linear layer, from a `Config` |
| `dispatch.{h,cc}` | The closed shape list, the verdict, and the switch |
| `gemm_tile_test.cc` | Not present: the host-side plan is checked by `dispatch_test` |
| `gemm_gpu_test.cc` | The device gate: each mainloop against cuBLAS |
| `gemm_benchmark.cc` | The rate gate: cuBLAS, the Hopper mainloop, and the tile over the closed list |
| `dispatch_test.cc` | The CPU gate of the table, the verdict, and the seam's defaults |

## 2. The staged tile

One block owns a `TileM` by `TileN` rectangle of C and walks k in slices of
`TileK`. A slice is staged into the dynamic shared window in the layout the
fragments read, and the pipeline holds `Stages` of them: the copy of slice
`n + Stages - 1` runs while slice `n` is used.

**A staged row walks the contiguous axis of the operand.** The element the
fragment reads as its leading dimension is the staged row. A transposed call
therefore stages from the other axis and reads a column-major fragment, so all
four operand forms take one chunked copy path:

| Operand | Stored as | Staged rows | Staged columns | Fragment |
|---|---|---|---|---|
| A plain | m by k, k contiguous | m | k | row major |
| A transposed | k by m, m contiguous | k | m | column major |
| B plain | k by n, n contiguous | k | n | row major |
| B transposed | n by k, k contiguous | n | k | column major |

The pad after a staged row is eight elements, which keeps every sixteen-byte
chunk aligned and breaks the bank the row stride would repeat. The pad is never
read.

The thread grid walks the (row, chunk) grid in row-major order, so threads next
to each other read addresses next to each other. A chunk whose ends are not
sixteen-byte aligned, and the part of a row outside the operand, takes a
per-element loop. The zero fill of a k tail and of a ragged edge is part of that
loop, so an operand edge never adds a product.

`docs/tensor-core-gemm-plan.md` section A.3 is why the copy uses the
`__pipeline_*` intrinsics and not raw `cp.async` assembly: inline assembly does
not compile on the host, so a raw form would remove the emulated leg in the same
commit that adds the kernel.

## 3. The schedules

The staged window is the dynamic one, so a schedule states its stage count and
the launcher asks for the opt-in limit with `cudaFuncSetAttribute`. A stage
count is a trade: one more stage hides one more copy, and it also costs a share
of the multiprocessor. Section 13 measures the trade, and section 14 measures the
tile.

| Schedule | Tile | Warps | k depth | Stages | Window | Registers |
|---|---|---|---:|---:|---:|---:|
| `128x64x32x2` | 128 by 64 | 2 by 2 | 32 | 2 | 30,736 B | 122 to 126 |
| `64x64x32x2` | 64 by 64 | 1 by 2 | 32 | 2 | 20,496 B | 122 to 126 |
| `128x128x32x4` | 128 by 128 | 4 by 2 | 32 | 4 | 81,936 B | 124 to 125 |
| `128x128x64x2` | 128 by 128 | 2 by 4 | 64 | 2 | 73,744 B | 120 to 126 |
| `128x128x64x2s8` | 128 by 128 | 2 by 4 | 64 | 2 | 73,744 B | 123 to 128 |
| `128x128x96x2s8` | 128 by 128 | 2 by 4 | 96 | 2 | 106,512 B | 122 to 124 |
| `128x256x32x2s8` | 128 by 256 | 4 by 4 | 32 | 2 | 61,456 B | 124 to 128 |

The `s8` suffix is the L2 group of eight m-tiles, and a name without it walks the
grid in the plain order. `128x128x64x2s8` is the default, because it won the
sweep of section 15, and `128x128x96x2s8` takes the weight-gradient rows. The
register and window figures are `cuobjdump -res-usage` of one build at bf16 and
`sm_90`, read over the four operand forms and the two stores of each schedule,
plus the count the plan states. The window is the staged buffers and nothing
else: the epilogue's exchange tiles lie over it, and the sixteen bytes above the
plan's count are the slack that rounds the base of the window to a
sixteen-byte boundary. A row of the shape table names its own schedule, so a
future promotion does not need a second sweep.

The register figure rose with section 15: a block that walks a run of tiles keeps
the state of one tile alive across the loop, and the compiler spends about thirty
registers on it. 128 registers at 256 threads still holds two blocks per
multiprocessor, so the tile keeps its residency, and the run of section 15 is
faster than the fixed grid it replaced.

## 4. The epilogue

The accumulator is fp32 in every case, and it leaves the registers through one
exchange tile per warp. A lane holds eight columns of two rows, so a store taken
straight from the lane map would put one warp instruction on eight rows of
sixteen bytes. The exchange tile puts the run of columns back together first: a
lane lays its two fragments down in shared memory, a warp barrier orders the
tile, and then one lane writes one sixteen-byte chunk of C. Section 12 measures
what each of those three choices costs.

Two properties are worth stating, because a later edit is likely to break them.
The tile is per warp, so no warp waits for another warp to finish its store. And
it lies over the staged window, which the last k slice releases, so the epilogue
spends no shared memory of its own.

`alpha` and `beta` are applied in the exchange write, which keeps one rounding
for a bf16 store whatever `beta` asks for. `beta` reads C only when it is not
zero. The bounds are per element, and a chunk the operand edge cuts takes its
elements one at a time.

## 5. The gates

| Gate | Command |
|---|---|
| Off device: the layout, the bounds, the pipeline, the epilogue | `tools/nanochat test --profile h100 --suite gemm` |
| The shape table, the verdict, the switch, the seam's defaults | `tools/nanochat test //backends/cuda/gemm:dispatch_test` |
| Device: each mainloop against cuBLAS on every contract case | `tools/nanochat test --gpu --config=hopper --config=bf16 //backends/cuda/gemm:gemm_gpu_test` |
| Rate: cuBLAS, the Hopper mainloop, and the tile over the closed list | `tools/nanochat bench -- bazel-bin/backends/cuda/gemm/gemm_benchmark` |

The device gate runs two candidates over the same comparison. The tile takes the
tiny shapes and the whole sweep of section 3. The Hopper mainloop of section 16
takes its own table, at shapes its alignment rule accepts, and it needs
`--config=bf16` because its operands are bf16. A refusal counts as a failure
there, so a case in that table is a case the mainloop claims to serve.

The emulated leg runs the real `.cu` source through the block runtime, and it now
runs every schedule over one shape. It checks the staged layout, the padding, the
leading dimension, the sub-tile addressing, the buffer index of the pipeline, the
accumulate order, the bounds, and the store. It cannot check the fragment
ownership layout, a bank conflict, the register count, the occupancy, or the
wait: the emulator's `__pipeline_wait_prior` is a no-op, so one call carries that
risk.

The device gate passes on an NVIDIA H100 80GB HBM3 at `sm_90` and bf16: every
contract case and every schedule agrees with cuBLAS. On Modal:

```bash
modal run tools/modal/app.py::card_job --name gemm_check
```

## 6. The measurement

One run of the H100 image, which builds `sm_90` at bf16:

```bash
modal run tools/modal/app.py::card_job --name gemm_bench \
    --argv '["--schedule","all","--json"]'
```

Every row below is one shape of the closed list, measured twice through the
shipped `kernels::Gemm` and `GemmWgrad`: once with the `NANOCHAT_GEMM_LIB`
switch at `cublas`, and once at `hand` on each schedule. The fraction is the
cuBLAS time over the best hand time, which is what gate G4 asks for. The hand
MFU is the tile's own rate over the bf16 peak of the card, 989 TFLOP/s.

| Row | cuBLAS ms | Hand ms | Schedule | Fraction | Hand MFU |
|---|---:|---:|---|---:|---:|
| `muon:5x12:A` | 0.0056 | 0.0074 | `64x64x32` | 0.758 | 0.0% |
| `muon:5x12:A2` | 0.0055 | 0.0074 | `64x64x32` | 0.740 | 0.0% |
| `muon:5x12:prod` | 0.0052 | 0.0074 | `64x64x32` | 0.699 | 0.0% |
| `muon:640x1280:A` | 0.0818 | 0.2962 | `128x128x32` | 0.276 | 14.3% |
| `muon:640x1280:A2` | 0.0558 | 0.1896 | `128x64x32` | 0.294 | 11.2% |
| `muon:640x1280:prod` | 0.0925 | 0.3588 | `128x64x32` | 0.258 | 11.8% |
| `muon:1280x1280:A` | 0.2243 | 1.0993 | `128x128x32` | 0.204 | 15.4% |
| `muon:1280x1280:A2+prod` | 0.2235 | 1.2181 | `128x64x32` | 0.183 | 13.9% |
| `muon:1280x5120:A` | 0.4551 | 1.9789 | `128x128x32` | 0.230 | 17.1% |
| `muon:1280x5120:A2+A2` | 0.1420 | 0.6215 | `128x64x32` | 0.228 | 13.6% |
| `muon:1280x5120:prod` | 0.5526 | 2.3844 | `128x128x32` | 0.232 | 14.2% |
| `muon:5120x1280:A` | 0.4457 | 2.4068 | `128x64x32` | 0.185 | 14.1% |
| `muon:5120x1280:prod` | 0.5210 | 2.3836 | `128x128x32` | 0.219 | 14.2% |
| `linear:fwd:q+proj` | 0.0751 | 0.3529 | `128x128x32` | 0.213 | 15.4% |
| `linear:dgrad:q+proj` | 0.0738 | 0.3937 | `128x128x32` | 0.188 | 13.8% |
| `linear:wgrad:q+proj` | 0.0838 | 0.6472 | `128x64x32` | 0.130 | 8.4% |
| `linear:fwd:kv` | 0.0485 | 0.1878 | `128x128x32` | 0.258 | 14.5% |
| `linear:dgrad:kv` | 0.0481 | 0.2310 | `128x64x32` | 0.208 | 11.8% |
| `linear:wgrad:kv` | 0.0506 | 0.6120 | `128x64x32` | 0.083 | 4.4% |
| `linear:fwd:fc` | 0.2592 | 1.3697 | `128x128x32` | 0.189 | 15.9% |
| `linear:dgrad:fc` | 0.2559 | 1.4280 | `128x128x32` | 0.179 | 15.2% |
| `linear:wgrad:fc` | 0.3030 | 1.7756 | `128x64x32` | 0.171 | 12.2% |
| `linear:fwd:cproj` | 0.2595 | 1.2413 | `128x128x32` | 0.209 | 17.5% |
| `linear:dgrad:cproj` | 0.2569 | 1.5052 | `128x128x32` | 0.171 | 14.4% |
| `linear:wgrad:cproj` | 0.3620 | 1.7913 | `128x64x32` | 0.202 | 12.1% |
| `linear:fwd:cls` | 1.5645 | 8.5476 | `128x128x32` | 0.183 | 16.3% |
| `linear:dgrad:cls` | 1.4678 | 8.7023 | `128x128x32` | 0.169 | 16.0% |
| `linear:wgrad:cls` | 1.4834 | 9.2846 | `128x128x32` | 0.160 | 15.0% |

The machine-readable rows are the `gemm_bf16` entry of
[docs/baselines.json](../../docs/baselines.json), which records the shipped
configuration: the same binary with each row on the schedule it owns.

The table lists the best of the four schedules for each shape. The shape table
keeps the wide tile on the rows where the two schedules are within one percent:
`muon:640x1280:prod` and the classifier weight gradient, whose difference is
inside the run to run spread. Section 7 gives the comparison over six of them.

A second run of the shipped configuration, on the same card, agreed with this
table within three percent on every row. It also showed the 5 by 12 group moving
from 0.67, 0.81, and 0.74 to 0.60, 0.64, and 0.61, which is the noise that group
carries and the reason section 8 does not read it as a rate.

## 7. The comparison, six shapes

The full list is section 6. These six rows are the ones the closed list exists
for, and each one states the work, the two times, the rate of each path, and the
fraction the gate reads.

| Shape | Work | cuBLAS | Hand | cuBLAS rate | Hand rate | Fraction |
|---|---:|---:|---:|---:|---:|---:|
| Muon `A = X^T X`, 1280 by 1280 over 40 matrices | 168 GFLOP | 0.2259 ms | 1.1092 ms | 743 TFLOP/s | 151 TFLOP/s | 0.204 |
| Muon `prod = X B`, 5120 by 1280 over 20 matrices | 336 GFLOP | 0.5219 ms | 2.4043 ms | 643 TFLOP/s | 140 TFLOP/s | 0.217 |
| Linear forward, `fc`, 16384 by 5120 by 1280 | 215 GFLOP | 0.2608 ms | 1.3813 ms | 823 TFLOP/s | 155 TFLOP/s | 0.189 |
| Linear dgrad, `cproj`, 16384 by 1280 by 5120 | 215 GFLOP | 0.2616 ms | 1.5418 ms | 821 TFLOP/s | 139 TFLOP/s | 0.170 |
| Linear wgrad, `fc`, 5120 by 1280 over 16384 | 215 GFLOP | 0.3074 ms | 1.7943 ms | 699 TFLOP/s | 120 TFLOP/s | 0.171 |
| Classifier forward, 16384 by 32768 by 1280 | 1374 GFLOP | 1.5897 ms | 8.7556 ms | 865 TFLOP/s | 157 TFLOP/s | 0.182 |

Every row needs a tensor core, so both paths do the same arithmetic. The times
are the shipped seam under each value of `NANOCHAT_GEMM_LIB`, one run of one
binary, on an NVIDIA H100 80GB HBM3 at bf16. The rate of each path is the work
over its own time.

What the six rows say:

- **The hand tile holds 139 to 157 TFLOP/s on the five rows that store the
  compute type**, which is 14 to 16 percent of the peak. The number barely moves
  with the shape, so the limit is inside the kernel and not in the operand
  traffic: a bandwidth bound would fall on the shapes that read more per output
  element.
- **cuBLAS holds 643 to 865 TFLOP/s**, which is 65 to 87 percent of the peak. Its
  worst row is the 5120 by 1280 product, whose k extent is the short one.
- **The gap is 4.6 to 5.9 times on every row.** Reaching the 0.70 gate needs 450
  to 605 TFLOP/s from the hand tile, which is 2.9 to 3.9 times the best row
  here. No knob in the sweep is worth that.
- **A weight gradient costs the hand tile a fifth of its rate**, 120 against the
  155 TFLOP/s of the same layer's forward, because the fp32 store writes twice the
  bytes through the same scratch tile. cuBLAS loses less, so that fraction is the
  lowest of the six.
- **The classifier is the largest shape and it does not change the picture**,
  which rules out the small-shape launch cost as the explanation.

## 8. The decision

Gate G4 asks for 0.70 of the cuBLAS rate, and G3 asks that no enabled shape be
slower than cuBLAS. **No row of the shapes that matter reaches 0.70.** The
largest fraction on a real shape is 0.294, section 7's six rows read 0.170 to
0.217, and the tile holds 14 to 16 percent of the card's bf16 peak against cuBLAS
at 65 to 87 percent. Every bit therefore stays off, and this is the negative
result that `docs/tensor-core-gemm-plan.md` section B.10 names as the stop
condition for this campaign.

The 5 by 12 group reads 0.70 to 0.76. That group is 0.005 to 0.008 ms on either
path, so its fraction is launch overhead and not a rate, and a slower run moves
it under the gate: an earlier run of the same binary read 0.60, 0.64, and 0.61
for it. It is not a promotion.

## 9. What a later attempt would need

The measured shape of the gap points at four causes, in the order a profile would
confirm them.

1. **The vendor `wmma` load path.** `wmma::load_matrix_sync` compiles to generic
   `LD.E` loads plus `MOVM.16.MT88` register permutes on `sm_90`, and not to
   `LDSM`. Assembling the fragment with `ldmatrix` and `mma.sync` directly is the
   single largest change available, and it needs the same two-arm wrapper that
   `backends/cuda/kernels/mma_ops.cuh` gives fp8.
2. **The shared window.** A tile deeper than 32 over k does not fit the 48 KiB
   static window beside the epilogue scratch. Four stages need dynamic shared
   memory, and that needs an `extern __shared__` mapping in the emulator before
   the off-device leg can see it.
3. **The epilogue.** The fp32 scratch tile and a store per warp cost the
   weight-gradient rows about 30 percent against the same rows of a forward
   store.
4. **Occupancy.** 128 registers per thread hold two blocks per SM at 256
   threads. A 128 by 256 tile with fewer registers would raise the arithmetic per
   staged byte.

## 10. The micro-tile, phase 1

[docs/kernels.md](../../docs/kernels.md) Part 3 reopens this package as a
removal: cuBLAS becomes the measurement reference and the hand library serves
every call. Phase 1 of that plan is the operand path, and its first step is
`mma_bf16.cuh`.

`wmma::load_matrix_sync` compiles to plain loads plus a register permute on
`sm_90`, and `wmma::mma_sync` is a 16 by 16 by 16 instruction. The header
replaces that pair with `ldmatrix.sync.aligned.m8n8` and
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, which is mechanism 1 of
section 4 of that part, and the one the probe showed is worth the most.

**Two arms, one lane map.** The device arm is the instruction pair in inline
assembly with the four fragments the instruction defines. The host arm holds the
whole micro-tile in every thread and computes it from shared memory. That is what
the emulation engine can run, exactly as the `<mma.h>` stub does for the `wmma`
path. The kernel that includes the header has one body for both, and the
host arm is exact and slower by the warp factor.

**A preprocessor symbol selects the arm, and it must be one.** A `constexpr`
constant is invisible to `#if`, so a constant would make every pass compile the
host arm and a kernel would then call a host function. The header states this at
the symbol, because it is the kind of defect that reads as correct.

**The lane map is checked, not assumed.** `mma_bf16_test.cu` holds one scalar
reference and two rows. The host row runs the header's host arm, so it needs no
card and it is the fast gate. The device row runs the instruction pair in one
warp. Both cover all four operand orientations, because the staged layouts of
`gemm_tile.cuh` produce all four, and both read the same maximum error of
`8.064e-05` against the reference.

```bash
tools/nanochat build //backends/cuda/gemm:mma_bf16_test
tools/nanochat run t0-cpu -- ./bazel-bin/backends/cuda/gemm/mma_bf16_test
modal run tools/modal/app.py::card_job --name mma_check
```

The gate found two defects before any rate was measured. The first was in the
test's own staging. The second was the address of the transposed B tile. The two
8 by 8 matrices of a B tile are its two k halves, so they lie along k. A
transposed operand stages n by k, so the second matrix is eight columns to the
right of the first, and not eight rows below it. The device row reported the two
failing orientations with an error of `2.6e-02` while the other two passed,
which named the defect in one run.

**What this is not.** A rate. The tile still stages, copies, and stores through
`wmma`, so phase 1 is a verified foundation and not a measurement. Three steps
remain:

1. Drive the header from `gemm_bf16.cu`, in place of the `wmma` pair.
2. Take the accumulator epilogue into registers. That removes the shared scratch
   and the eight serialized rounds of section 2.
3. Raise the stage count past the two of `gemm_plan.h`.

## 11. What the operand path bought

Section 10 describes the micro-tile. This section records its first effect on the
shipped tile, which is the same binary with one change: `gemm_bf16.cu` drives
`mma_bf16.cuh` in place of the vendor `wmma` pair. The staged window, the warp
tile, the two stages, the epilogue scratch, and the shape table are untouched, so
the difference between the two columns is the operand path alone.

One run of one binary, `--schedule all`, on an NVIDIA H100 80GB HBM3 at bf16:

| Row | cuBLAS | `wmma` | `ldmatrix` | Fraction before | Fraction after | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.2230 ms | 1.1004 ms | 0.8978 ms | 0.211 | 0.248 | 1.23 |
| Muon `prod` | 0.5209 ms | 2.3845 ms | 1.7967 ms | 0.219 | 0.290 | 1.33 |
| Linear forward `fc` | 0.2568 ms | 1.3633 ms | 1.1095 ms | 0.190 | 0.231 | 1.23 |
| Linear dgrad `cproj` | 0.2559 ms | 1.5208 ms | 1.1373 ms | 0.170 | 0.225 | 1.34 |
| Linear wgrad `fc` | 0.2998 ms | 1.9679 ms | 1.2782 ms | 0.155 | 0.235 | 1.54 |
| Classifier forward | 1.5593 ms | 8.6150 ms | 7.0370 ms | 0.184 | 0.222 | 1.22 |

The rows are `gemm_bf16_phase1` in [baselines.json](../../docs/baselines.json).
The `wmma` column is the `gemm_bf16` entry, measured in the same session on the
same card.

**What moved.** Every row gained. The two weight-gradient rows gained most at
1.34 and 1.54, and the classifier, which is the largest shape, gained least at
1.22. The `hand_mfu` column reads 0.19 to 0.20 on every row, so the tile is now
at a fifth of the card's bf16 peak and the curve is still flat.

**The evidence in the instructions.** The bf16 build reports `REG:120`,
`LOCAL:0`, and `SHARED:41984` per thread, against the 126 to 128 registers of the
`wmma` tile. The SASS holds `LDSM.16.M88.4` and `LDSM.16.MT88.4` for the A
operand, `LDSM.16.M88.2` and `LDSM.16.MT88.2` for B, and
`HMMA.16816.F32.BF16` for the arithmetic. There is no `MOVM.16.MT88`, which is
the register permute that the vendor load path generated for every fragment and
that section 9 named as the first cause of the gap.

**What is left.** The fractions now read 0.22 to 0.29. The phase 1 gate asks for
0.50, so the operand path was worth about a third of the remaining gap and not
all of it. Three causes of section 9 remain: the two stages of `gemm_plan.h`,
the shared epilogue scratch with its eight serialized rounds, and the occupancy.
The first of the four is now closed.

## 12. The epilogue in registers, phase 1 step 2

Step 2 of `docs/kernels.md` Part 3 section 10 takes the accumulator out of the
shared scratch. The table below is the first attempt, which wrote C straight
from the lane map, against section 11. It is a loss on every row, so the record
keeps it.

One run of one binary, `--schedule all`, on an NVIDIA H100 80GB HBM3 at bf16,
`128x128x32` on every row:

| Row | cuBLAS | scratch (section 11) | straight to global | Ratio |
|---|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.2226 ms | 0.8974 ms | 0.9463 ms | 0.948 |
| Muon `prod` | 0.5209 ms | 1.7693 ms | 1.8133 ms | 0.976 |
| Linear forward `fc` | 0.2563 ms | 1.1184 ms | 1.1990 ms | 0.933 |
| Linear dgrad `cproj` | 0.2560 ms | 1.1627 ms | 1.2172 ms | 0.955 |
| Linear wgrad `fc` | 0.2996 ms | 1.3039 ms | 1.3255 ms | 0.984 |
| Classifier forward | 1.5589 ms | 7.0285 ms | 7.8413 ms | 0.896 |

**Why the direct store loses.** A lane's four accumulator registers hold two
adjacent columns of two rows, eight rows apart. One warp store instruction
therefore lands in eight rows of sixteen bytes, which is eight sectors of a
cache line for each thirty-two-byte transaction, and C is written at the end of
a block where nothing hides it. The bf16 store is four bytes a lane and the
loss is the larger; the fp32 store of a weight gradient is eight bytes a lane
and the loss is the smaller. That ordering is what named the cause.

**The exchange tile.** The fix keeps the lane map for the arithmetic and adds a
per-warp shared tile of two neighbouring fragments, so the warp writes sixteen
columns and sixteen rows as sixteen-byte chunks. One run of the same binary:

| Row | cuBLAS | Section 11 | Step 2 | Gain |
|---|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.2231 ms | 0.8974 ms | 0.8417 ms | 1.066 |
| Muon `prod` | 0.5193 ms | 1.7693 ms | 1.6912 ms | 1.046 |
| Linear forward `fc` | 0.2579 ms | 1.1184 ms | 1.0527 ms | 1.062 |
| Linear dgrad `cproj` | 0.2597 ms | 1.1627 ms | 1.0881 ms | 1.069 |
| Linear wgrad `fc` | 0.3039 ms | 1.3039 ms | 1.3427 ms | 0.971 |
| Classifier forward | 1.5874 ms | 7.0285 ms | 7.0999 ms | 0.990 |

The five rows that store the compute type gain 4.6 to 6.9 percent, and the
`128x128x32` schedule keeps every row it won. The two rows whose store is fp32
or whose shape is the largest are within the three percent spread that section 6
records between two runs of one binary, so this run does not separate them from
the scratch tile. The register count falls from 120 to 92 to 96 over the four
operand forms, which is the one effect the epilogue was expected to have and
got.

**The gate.** Step 2 asks for `fraction_of_cublas` above 0.28 on all six rows.
This run reads 0.2236 to 0.3071, so one row passes and five do not. The epilogue
was not the limit: the tile still copies two stages of a 128 by 128 tile, which
is 64 flops for every byte it reads. Step 3 is the one that changes that.

## 13. The deeper pipeline, phase 1 step 3

Step 3 of `docs/kernels.md` Part 3 section 10 replaces the two staged buffers
with four to six. Two stages were the most the 48 KiB static window held, so the
step also moves the window to the dynamic one and asks the driver for the opt-in
limit. That part works: the emulation engine rewrites the `extern __shared__`
declaration into a pointer at the block's dynamic buffer, so the off-device gate
runs the same source, and `cudaFuncSetAttribute` joins its runtime.

**The stage count did not pay the way section 10 expected.** One run of one
binary on an NVIDIA H100 80GB HBM3 at bf16, `--schedule all`, with two and four
stages of the same 128 by 128 tile in one sweep:

| Row | cuBLAS | Two stages | Four stages | Gain | Fraction at four |
|---|---:|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.2240 ms | 0.9356 ms | 0.8904 ms | 1.051 | 0.2516 |
| Muon `prod` | 0.5193 ms | 1.8680 ms | 1.7499 ms | 1.067 | 0.2968 |
| Linear forward `fc` | 0.2586 ms | 1.1673 ms | 1.1090 ms | 1.053 | 0.2332 |
| Linear dgrad `cproj` | 0.2599 ms | 1.2053 ms | 1.1283 ms | 1.068 | 0.2304 |
| Linear wgrad `fc` | 0.3049 ms | 1.2869 ms | 1.2879 ms | 0.999 | 0.2368 |
| Classifier forward | 1.6093 ms | 7.3750 ms | 6.9889 ms | 1.055 | 0.2303 |

Four stages gain three to seven percent on five rows and nothing on the sixth.
Six stages of the same tile were measured in the run before at 1.34 to 10.65 ms,
which is 1.5 times the two-stage time: a window of 120 KiB holds one block per
multiprocessor and the tile then has half the resident warps.

**Why the returns stop there.** The tile is not waiting on one copy. At two
stages the 128 by 128 tile already asks for 2.9 TB/s of operand traffic on the
`muon:1280x1280:A` row, which is 86 percent of the card's HBM bandwidth. A deeper
pipeline cannot move more bytes than the memory system has. What it can do is
hide a latency that the arithmetic already covers, and the measurement says the
arithmetic covers it.

**Two consequences for the plans.** The wide tile keeps four stages, because two
blocks of 80 KiB still fit a Hopper multiprocessor and the gain is there. The
small tiles keep two, because a block of 128 threads is limited by how many
blocks a multiprocessor holds and every extra stage takes one away: the base
schedule at four stages read 1.18 ms where two stages read 1.02 ms.

**The gate.** Step 3 asks for `fraction_of_cublas` above 0.40 on all six rows.
The run reads 0.2303 to 0.2968, so the step misses it by a wide margin and by
its own rule the part takes the stage count of the pipeline as measured and not
as estimated. The reason is arithmetic intensity: 64 flops for every staged byte
is 15.5 TB/s of operand traffic at the card's peak, which no memory system here
can serve. That is what steps 4 and 5 are for.

## 14. The wider tile, phase 1 step 4

Step 4 of `docs/kernels.md` Part 3 section 10 asks for a wider tile, so a block
holds more arithmetic for each staged byte, and for fewer registers a thread. The
sweep that chose `128x128x32` predated the new operand path, so this is a second
sweep and not a guess. One run of one binary on an NVIDIA H100 80GB HBM3 at
bf16, `--schedule all`:

| Schedule | Tile | Warps | k | Stages | `muon:1280x1280:A` | Classifier |
|---|---|---|---:|---:|---:|---:|
| `128x128x32x4` | 128 by 128 | 8 | 32 | 4 | 0.8961 ms | 6.9945 ms |
| `128x128x64x2` | 128 by 128 | 8 | 64 | 2 | **0.8259 ms** | **6.5931 ms** |
| `128x256x32x2` | 128 by 256 | 16 | 32 | 2 | 0.9629 ms | 7.6676 ms |
| `128x256x32x3` | 128 by 256 | 16 | 32 | 3 | 0.9359 ms | 7.0511 ms |
| `256x128x32x2` | 256 by 128 | 16 | 32 | 2 | 0.9638 ms | 7.1963 ms |
| `128x64x32x2` | 128 by 64 | 4 | 32 | 2 | 1.0237 ms | 9.4424 ms |
| `64x64x32x2` | 64 by 64 | 2 | 32 | 2 | 1.2502 ms | 12.8164 ms |

**The wider tile lost.** `128x256x32x2` reads 87 flops for every staged byte
against the 64 of the square tile, and it is slower on every row. The reason is
residency again, and this time it is the register file: a block of 512 threads
holds one block per multiprocessor, so the extra columns buy no extra warps. The
same run says what does pay, and it is not a wider tile: `128x128x64x2` doubles
the k slice of the square tile, so a block does the same arithmetic for each
staged byte and passes half as many barriers. It gains 1.061 to 1.135 on every row
that matters and it keeps two blocks of 72 KiB per multiprocessor.

**The register side.** The wider schedules hold 102 to 108 registers a thread,
against 93 to 96 for `128x128x32x4`, so the plan did not deliver the register
saving the step asked for either. `128x128x64x2` holds 101 to 109 and loses
nothing by it, because the square tile's residency is set by its window.

**The gate.** "The best schedule per row changes, and the sweep says which" is
the whole of this step's gate, and it holds: the winner moved from
`128x128x32x4` to `128x128x64x2` on all six rows, and the shape table names it
for every model row. The fractions read 0.2358 to 0.3236, so the phase's own
0.50 gate is still out of reach, which section 5 of `docs/kernels.md` Part 3
predicts for a warp-level mainloop.

## 15. The tile order, phase 1 step 5

Step 5 of `docs/kernels.md` Part 3 section 10 asks for an L2 swizzle of the block
order and for stream-K on the ragged rows. The shipped form is both halves of
the tile-parallel scheduler and not the k-split half of stream-K: one block owns
a run of tiles, the grid is sized from the blocks a multiprocessor can hold, and
`SwizzleM` decides the order the run walks.

**The persistent grid is the larger half.** A row whose tile count is not a whole
number of waves leaves part of the card idle in the last wave, and the Muon rows
have three to sixteen waves. One run of one binary before and after, on the same
schedule and the same card:

| Row | Fixed grid (step 4) | Persistent grid | Gain |
|---|---:|---:|---:|
| Muon `A = XᵀX` | 0.8259 ms | 0.7425 ms | 1.112 |
| Muon `prod` | 1.6093 ms | 1.4731 ms | 1.092 |
| Linear forward `fc` | 1.0041 ms | 0.9669 ms | 1.039 |
| Linear dgrad `cproj` | 1.0373 ms | 0.9713 ms | 1.068 |
| Linear wgrad `fc` | 1.1371 ms | 1.0176 ms | 1.117 |
| Classifier forward | 6.5931 ms | 6.0635 ms | 1.087 |

**The swizzle is the smaller half, and it is where it was expected.** A group of
eight m-tiles walks n first, so the blocks resident together share eight A panels
and a run of B panels that fits the cache. In the same run, with the group on:

| Row | Plain order | Swizzled | Gain |
|---|---:|---:|---:|
| Muon `A = XᵀX` | 0.7425 ms | 0.7378 ms | 1.006 |
| Muon `prod` | 1.4731 ms | 1.4632 ms | 1.007 |
| Linear forward `fc` | 0.9669 ms | 0.9578 ms | 1.009 |
| Linear dgrad `cproj` | 0.9713 ms | 0.9615 ms | 1.010 |
| Linear wgrad `fc` | 1.0176 ms | 1.0158 ms | 1.002 |
| Classifier forward | 6.0635 ms | 5.7805 ms | 1.049 |

The classifier gains five times what the other rows gain, which is what the
argument for an L2 swizzle predicts: it is the one row whose B panel is larger
than the cache, at 84 MB against a 50 MB cache.

**The cost of the loop.** The register figure moves from 93 to 109 a thread to
120 to 128, because a block that walks a run of tiles keeps the state of one tile
alive across the loop. 128 registers at 256 threads still holds two blocks per
multiprocessor, and the run is 3.9 to 11.7 percent faster than the fixed grid, so
the loop pays for what it spends.

**What is not here.** The k-split half of stream-K is not implemented. A split
along k needs a workspace, a second accumulator pass, and a fixup stage, and it
buys the tail of a wave: the classifier row has 124 waves, so its tail is under
one percent, which is the row where section 10 says the tail matters least. It
is the right change for a row whose tile count is below the resident block
count, and every such row here is a Muon group of a few milliseconds. Recorded
as not done rather than as done.

**The gate.** Step 5 asks that no row be slower and that the classifier row
reach 0.45 of the cuBLAS rate. No row is slower in the run, and the classifier
reads 0.2753, so the step misses the second half. The fractions of all six rows
now read 0.2681 to 0.3556, against 0.2215 to 0.2950 before step 2.

**What the four steps together say.** One session measured the tree before step
2 and the tree after step 5, each with `--schedule all` and each scored on the
best schedule of its own sweep:

| Row | Before | After | Gain | Fraction before | Fraction after |
|---|---:|---:|---:|---:|---:|
| Muon `A = XᵀX` | 0.8969 ms | 0.7389 ms | 1.214 | 0.2486 | 0.3042 |
| Muon `prod` | 1.8114 ms | 1.4748 ms | 1.228 | 0.2882 | 0.3530 |
| Linear forward `fc` | 1.1211 ms | 0.9481 ms | 1.182 | 0.2318 | 0.2707 |
| Linear dgrad `cproj` | 1.1582 ms | 0.9730 ms | 1.190 | 0.2256 | 0.2671 |
| Linear wgrad `fc` | 1.3072 ms | 0.9869 ms | 1.325 | 0.2331 | 0.3079 |
| Classifier forward | 7.1397 ms | 5.7740 ms | 1.237 | 0.2229 | 0.2714 |
| Total | 13.4345 ms | 10.8957 ms | 1.233 | | |

The rate of the tile rose from 164 to 192 TFLOP/s to 218 to 238 TFLOP/s, which is
18 to 33 percent of the card's bf16 peak. A repeat run of the final tree agreed
with this table within 0.8 percent on every row.

The remaining gap is the instruction and not the schedule. A warp-level
`mma.sync` mainloop reads each operand through `ldmatrix` into the register file
and issues 16 by 8 by 16 products, and section 3 of `docs/kernels.md` Part 3 sizes
that ceiling at half the card before any of this was measured. Phase 2 is the
answer: one warpgroup instruction, the Tensor Memory Accelerator, and a producer
warp. Its gate asks for 1.05 of the cuBLAS rate, where this phase's own gate
asked for 0.50 and did not arrive.

## 16. The Hopper mainloop, phase 2

Phase 2 of `docs/kernels.md` Part 3 section 6 asks for one warpgroup instruction,
the Tensor Memory Accelerator, and a producer warp. Section 15 of this document
sized the warp-level ceiling at about half the card and closed phase 1 at 0.27 to
0.36 of the cuBLAS rate. This section is the answer, and it does not write a
mainloop: `cutlass_bf16.cu` drives the one NVIDIA CUTLASS supplies.

**What the package holds now.**

| File | What |
|---|---|
| `cutlass_bf16.cu` | Eight CUTLASS kernel instantiations and the launch behind them |
| `cutlass_launch.h` | The two launchers, readable by a host pass |
| `gemm_shape.h` | The resolved call, the tile's rule, and the Hopper mainloop's rule |

**One tile, one schedule, eight kernels.** The tile is 128 by 256 by 64 with a
2 by 1 cluster. The builder chooses three stages for it, which is 215,040 B of
the 227 KiB a block may ask for. The eight kernels are the four operand
orientations of `GemmParams` times the two store types, because a layout tag and
an epilogue element are compile-time parameters. `dev/cutlass/README.md` records
the probe that found the tile and the constraint that removed the 64 by 256
candidate.

The object holds the seven mechanisms of `docs/kernels.md` Part 3 section 4.
`UTMALDG.3D` and `UTMALDG.3D.MULTICAST` copy the operands, `HGMMA.64x256x16.F32`
`.BF16` is the arithmetic, `UTMASTG.3D` stores C, and `USETMAXREG` splits the
registers between the warpgroups. Every one of them was absent from the tile
before this section.

**The rule, and why it refuses.** The rule tests four conditions, in this order:
a non-empty extent, at least 4096 output elements, a stride in whole sixteen-byte
units, and a pointer on that boundary. The size condition is a measurement and
not a proof. A Tensor Memory Accelerator descriptor is built per call, so a shape
smaller than one tile loses to the launch it saves. The 5 by 12 Muon group is 60
outputs and stays on the hand tile.

A refused call is not an error. It takes the hand tile and then cuBLAS, and the
benchmark prints the reason.

**Three settings are not the builder's defaults.**
`StageCountAutoCarveout` makes the builder size the pipeline from what the
epilogue leaves of the window, because a hand-written stage count did not fit.
`PersistentScheduler` keeps a block over a run of tiles, which is the half of the
scheduler that section 15 measured as the larger one. `LinearCombination` is one
fused visitor for `alpha * acc + beta * C`.

**The build states `sm_90a` on itself.** `cutlass_bf16` is the only target in the
tree that names an architecture, and `rules_cuda` takes it as a replacement for
the command-line value rather than an addition. The default build of `//...`
therefore stays at `sm_75` and compiles no Hopper code. The target also states
its own warning options, because the CUTLASS headers emit thousands of warnings
and the tree builds with `-Werror`.

**What is still not here.**
- A rate. `dev/cutlass` and this file measure nothing; `gemm_benchmark` does.
- An emulated leg. The engine maps no Tensor Memory Accelerator and no warpgroup
  instruction, so `cutlass_bf16.cu` is card-only and the hand tile stays as the
  off-device gate.
- The fp8, stream-K, and grouped tiers of phase 4.

## 17. The candidate registry and the sweep

Section 11.1 of `docs/kernels.md` timed one kernel body against cuBLAS and read
0.051 to 0.154 of it. Section 11.2 found the cause in the machine code: the
accumulator leaves the register file inside the k loop, and the stack is 576
bytes. This section is the answer of section 11.4 to that record: it measures
four candidates. Section 11.9 then found the cause in the build and cleared it.
The four are clean kernels at this revision.

**What the package holds now.**

| File | What |
|---|---|
| `cutlass_candidates.h`, `cutlass_candidates.cc` | The registry of four candidates, and the lookups a report and a command line use |
| `cutlass_candidate.cu` | One kernel body, compiled once for each candidate |
| `cutlass_bf16.cu` | The dispatcher: the choice of candidate, the device test, and the two seam entries |
| `bench_shapes.h`, `bench_shapes.cc` | The 29 measurement rows, and every quantity each row derives |
| `gemm_benchmark.cc` | The sweep: two row families, three arms, one protocol |
| `cutlass_rules.h` | The decision table and its lookup |

**Four candidates.** `wide-n` is the tile of section 16, kept because its 256
wide N suits the classifier, whose N is 32768. `balanced` halves N, which halves
the accumulator and is the candidate section 11.2 expects to survive. `deep-k`
doubles the k slice, for a long loop with few tiles. `tall-m` doubles M, for the
weight gradient. Each candidate is one compiler process. One source file holds
every kernel body, and the build compiles it once for each registry index. One
process therefore holds one candidate's eight instantiations, and never two.

**The spill gate needs no card.** `tools/gemm_spill_gate.py` reads the output of
`cuobjdump -sass` for each built object, and it refuses a candidate whose `LDL`
or `STL` instructions fall between its `HGMMA` instructions. It refuses a
candidate whose k loop calls out as well, which is the defect of section 11.9.
Section 11.2 took a card run and a hand inspection to find the defect; this gate
takes three minutes. Its `--sass` mode reads a text file, so its own test needs
no toolchain.

It ran on the first build, and every candidate spilled:

| Candidate | Tile | Stages | Kernels that spill | The worst instantiation |
|---|---|---:|---|---|
| `wide-n` | 128x256x64 | 3 | 8 of 8 | 8 HGMMA, 64 LDL, 64 STL |
| `balanced` | 128x128x64 | 5 | 8 of 8 | 8 HGMMA, 32 LDL, 32 STL |
| `deep-k` | 128x128x128 | 2 | 8 of 8 | 16 HGMMA, 32 LDL, 32 STL |
| `tall-m` | 256x128x64 | 3 | 8 of 8 | 16 HGMMA, 64 LDL, 68 STL |

The `wide-n` row states the two numbers of section 11.2 exactly. The gate
therefore reproduces that finding. The `balanced` row halves the count and keeps
the symptom. The spill is therefore not a property of the tile. Section 11.9
names the property it is: the target did not define `NDEBUG`, so the `assert`
of every vendored header became a device call, and ptxas serialized the
multiplies around it.

**The second run, with `-DNDEBUG`.** One rebuild of the same four translation
units reads clean for all of them:

| Candidate | Kernels that spill | The worst instantiation | Stack |
|---|---|---|---:|
| `wide-n` | 0 of 8 | 8 HGMMA, no local access | 0 B |
| `balanced` | 0 of 8 | 8 HGMMA, no local access | 0 B |
| `deep-k` | 0 of 8 | 16 HGMMA, no local access | 0 B |
| `tall-m` | 0 of 8 | 16 HGMMA, no local access | 0 B |

The same objects hold no `CALL.ABS.NOINC` and no ptxas warning. Section 11.9 of
`docs/kernels.md` is the measurement.

**The sweep.** `gemm_benchmark` measures three arms: cuBLAS, one candidate at a
time of the registry, and one schedule at a time of the hand tile. Every arm
runs through the seam, so a measurement covers the wiring a training step takes.
Six steps bound the protocol, and each one answers a defect of the measurement
before it:

1. The operands are filled on the device. A two GiB row therefore costs no host
   copy, and the benchmark profile's host budget is 5 to 6 GiB.
2. One workspace serves every row. It is sized from the largest row that runs.
3. A row past the card's budget is reported as skipped, with its bytes and the
   reason.
4. The second-level cache is flushed before every timed call. The classifier
   operands are 63 MB and the cache is 50 MB, so a warm cache would choose the
   winner.
5. Every arm reports the minimum and the median of the rounds. The minimum keeps
   the meaning section 15 gave that column, and the median is what a decision
   reads.
6. Every candidate is checked against cuBLAS before its timing is reported. A
   small row compares every element; a big row reduces the squared norm of the
   difference on the device.

`--self-test` proves the check itself: an identical copy must match, and a
perturbed copy must be rejected by name. `--rows measurement` runs the scaling
families of section 11.5, which say how a kernel scales and never select a
candidate. `--auto-iters` takes a row's iteration count from its own work, so a
round runs at least 50 ms.

**The decision table.** `cutlass_rules.h` maps a production shape to the arm the
sweep chose, in the order candidate, then hand tile, then cuBLAS.
`tools/gemm_rules.py --write` writes the block from a `gemm_bench --json` report
of the production rows, and `--check` compares the committed block with the
report that produced it. The table is data, so a hand edit fails a gate instead
of changing a shipped decision. `dispatch_test` states that a written row keys a
shape of the closed list.

**The card runs.** Two runs on an H100 80GB HBM3 measured both families at the
protocol above, one before section 11.9 and one after. The numbers are the
`gemm_cutlass` and `gemm_scaling` entries of `docs/baselines.json`, and
`docs/kernels.md` section 11.8 is the record. The first measured spilling
kernels, so its timings state the defect and not the adapter. The second reads
0.576 to 1.176 of cuBLAS on the production rows, and it promotes nine of them.
The candidates move from 84 to 96 TFLOP/s to 576 to 705.

- The best candidate on each production row reads 0.576 to 1.176 of cuBLAS. The
  gate asks for 0.95. Twelve rows are faster than cuBLAS and six are clearly
  slower.
- The hand tile reads 0.147 to 0.740 of cuBLAS, which no flag changed.
- `wide-n` wins eight rows and `balanced` two. `deep-k` wins none at this
  revision. The 5 by 12 group refuses every candidate by the size rule, and
  `align:offset` refuses them by the pointer rule.
- cuBLAS holds 693 to 787 TFLOP/s above 4096, which is 70 to 80 percent of the
  peak. The candidates hold 576 to 705, and the hand tile 115 to 148.
- The check accepted every measured arm, so no fast arm was wrong.

`tools/gemm_rules.py --write` wrote the 28 rows from the second report and
promoted nine of them, each by more than the 0.04 margin of section 11.4 item 5.
`CutlassCandidateFor` therefore serves those nine shapes and answers -1 for the
other 19. Three rows were faster than cuBLAS and stayed on it, because their win
is inside the margin. The flush also moved the cuBLAS column by 0 to 12 percent
against the warm cache entry it replaced. The classifier forward moved most,
because its operands are 126 MB against a 50 MB cache.

**The seam arm passed on the card.** `gemm_benchmark --seam` measured the shipped
path on all 28 production rows, and no row was rejected. The nine promoted rows
read 0.87 to 0.97 of cuBLAS, which is the promotion's gain in the path a training
step takes. Its first run found a defect in the protocol and not in a kernel. A
weight-gradient row states `beta = 1`, so one call accumulates one product. The
arm ran eleven calls and the reference ran one, so the check read ten times the
reference. The reference now runs the same count.

**What is still not here.**

- The crossover of section 11.6 item 3. `kCutlassMinOutputElements` is still an
  argument, and no sweep varies it.
- The six rows the candidate loses. `linear:wgrad:kv` at 0.576 is the worst, and
  a tile with a short N is the shape they ask for.
- A candidate that wins above 8192 on a square shape. cuBLAS keeps 0.81 of the
  lead there, because the operands leave the second-level cache.

## 18. The fp8 arm, item 4a

`docs/kernels.md` section 10.4 is the record. This section states what the
package holds, why the tile is not the bf16 `wide-n` row, and what is still open.

**What it is.** `cutlass_fp8.cu` puts the seam's `GemmScaledFp8` on the same
warp-specialized mainloop with an fp32 accumulator. It carries all three forms
the seam issues: the forward at E4M3 by E4M3, and the dgrad and wgrad at E5M2 by
E4M3. Two seam entries exist, one for each store form. Only an fp8 build
compiles the target, and only an fp8 build of `gemm.cu` calls it, through
`if constexpr (kFp8Build)`.

```bash
tools/nanochat build --config=cutlass --config=cuda --config=fp8 \
  --@rules_cuda//cuda:archs=sm_90 //backends/cuda/gemm:cutlass_fp8
```

**The scale never reaches the host.** `GemmScaledFp8` states one scale for each
operand as a device pointer. One single-thread launch forms
`alpha * scale_a[0] * scale_b[0]` into a device float, and the epilogue reads
alpha from that address through the fused `LinearCombination`. Beta stays a
by-value scalar. The slot is four bytes, allocated once for the thread, and it is
not part of the CUTLASS workspace, because the epilogue reads it while the
kernel runs.

**The forward tile is `balanced`, not `wide-n`, and a host test said so.** The
fp8 accumulation path of CUTLASS keeps two fp32 accumulators, because it promotes
one into the other every `mma_promotion_interval` k blocks. At 256 wide N and two
warpgroups one accumulator is 128 registers. Both kernels of the first build read
168 registers with a 256 byte stack frame, and `tools/gemm_spill_gate.py` found
the accumulator in local memory inside their k loops.

| Form | Tile | Stages | Registers | Stack | Spill gate |
|---|---|---:|---:|---:|---|
| forward, `wide-n` | 128x256x64 | 3 | 168 | 256 B | 2 of 2 kernels spill |
| forward, `balanced` | 128x128x64 | 5 | 168 | 0 | clean |
| dgrad | 128x128x128 | 2 | 168 | 8 B | clean |
| wgrad | 128x128x128 | 2 | 168 | 0 | clean |

The fp8 operand is one byte. Five stages of the `balanced` tile stage 81920 B of
the 227 KiB window, and two stages of a 128 deep slice stage 65536 B.

**The backward forms need a deeper k slice.** The two backward calls are not
K-major in both operands. CUTLASS therefore stages the awkward operand through
its transposing collective, and that path is unreachable at a k slice of 64. A
probe instantiation reported `RS loops require more than 2 MMA k-iterations for
correctness` for the dgrad form, and `Invalid warp thread shape` for the wgrad
one. At a k slice of 128 both compile, and `kFp8BackwardTile` states that slice.
The 8 bytes of stack on the dgrad kernel is not the accumulator: the gate reads
no local access between two multiplies.

**The gate needed `QGMMA` to see the body.** `tools/gemm_spill_gate.py` looked
only for `HGMMA`, the bf16 and fp16 multiply, so an fp8 body read as "no k loop
to check" and passed unseen. Hopper spells the eight-bit multiply `QGMMA`. The
gate accepts both prefixes now, and its `cutlass_fp8` directory label names the
arm, which has no registry index.

**Every form the seam issues is covered, and no other is.** `CallSupported` reads
one rule from `gemm_shape.h`, `Fp8ScaledFormSupported`, which maps a pair and an
orientation to a form. The forward is E4M3 by E4M3 and K-major in both operands,
the dgrad is E5M2 by E4M3 with A K-major, and the wgrad is E5M2 by E4M3 with A
M-major. A call with another pair or orientation is refused, and cublasLt keeps
it. The arm is therefore four kernels and not eight: the two forward stores, the
dgrad at the compute type, and the wgrad at fp32.

The benchmark states the pair from the row's own orientation through the same
rule. Without that it stated E4M3 for every row, so every backward row was
refused and both of its fp8 columns timed cublasLt.

**Nothing is promoted.** `NANOCHAT_GEMM_FP8=cutlass` is the only way a shipped
call reaches the arm, and a training run does not set it. `gemm_benchmark --fp8`
sets it for the arm it times and clears it afterwards. A promotion needs a card
run against the tolerance and the margin of section 11.4.

**The measurement.** `gemm_benchmark --fp8` quantizes each row's own operands
with the seam's `QuantizeFp8`, then times the same call on cublasLt and on the
arm. It needs an fp8 build and a Hopper device, so the binary is built twice:

```bash
tools/nanochat build --config=cuda --config=fp8 \
  --@rules_cuda//cuda:archs=sm_90 //backends/cuda/gemm:gemm_benchmark
```

A host that builds one CUTLASS target at a time adds `--config=cutlass`, which
is the host-local cap of [build.md](../../docs/build.md).

The card route runs through the app's fp8 image. `tools/modal/app.py` holds a
third leg, `H100_FP8`, and one card job on it:

```bash
modal run tools/modal/app.py::card_job --name gemm_bench_fp8 \
  --argv '["--rows","production","--json"]'
```

The leg is not in `LEG_ORDER`, so the post-training chain never runs it, and both
chain legs keep fp32 and bf16.

**The measurement ran, and the arm is behind on four rows of five.** One run of
the fp8 image on an H100 80GB HBM3 measured the arm against cublasLt:

| Row | cublasLt | The arm | Fraction |
|---|---:|---:|---:|
| `linear:fwd:cls` | 1.2531 ms | 1.6689 ms | 0.751 |
| `linear:fwd:cproj` | 0.1795 ms | 0.2190 ms | 0.820 |
| `linear:fwd:fc` | 0.2074 ms | 0.2479 ms | 0.837 |
| `linear:fwd:q+proj` | 0.0662 ms | 0.0731 ms | 0.906 |
| `linear:fwd:kv` | 0.0424 ms | 0.0411 ms | 1.031 |

The gate asks for 0.95. One row meets it, by 3 points, which is inside the margin
of `docs/kernels.md` section 11.4, so nothing is promoted. The rows are the
`gemm_fp8` entry of [baselines.json](../../docs/baselines.json).

**Two limits bound the run, and the library states the first.** cublasLt answers
status 15 for every operand form but the K-major pair, so the Muon rows and the
two backward families carry no measurement. `src/ops.cc` calls the same seam
entry for the dgrad and the wgrad, so on Hopper the tier's own backward path ends
in that answer. The second is ours: the two backward instantiations faulted on
the card with an illegal memory access, so the arm states the K-major pair only.
The benchmark applies both limits before it calls, and prints `fp8 arms skipped`
for a row outside them.

**The host gate.** `//backends/cuda/gemm:dispatch_test` checks the arm's verdict
without a card and without the vendor header. It covers the accepted dense call,
the four refusals in their order, and the fp8 row alignment that separates that
arm from bf16. It also covers the batch refusal, the tile fields, and the staged
bytes.

## 19. The fp32 tile, phase 3

`docs/kernels.md` section 6.2 is the record. This section states what the
package holds, why the tile has no tensor core, and what the two schedules
trade.

**What it is.** `gemm_fp32.cu` is the third mainloop of `docs/kernels.md`
section 5. It serves an fp32 build and no other: the build selects its source
for the default precision, and `gemm.cu` discards the only call through
`if constexpr (!kComputeIsBf16)`, so no source holds an `#if`.

**Why no tensor core.** The T4 declares no bf16 (`backends/cuda/caps.json`), and
`sm_75` has neither a warpgroup instruction nor a Tensor Memory Accelerator. A
warp-level `mma.sync` pair exists there, but the chain's compute type is fp32 and
`sm_75` has no fp32 tensor-core input. The tile is plain fused multiply-add over
registers, so parity with cuBLAS is the target and not a multiple of it.

**Both operands stage k-major**, `As[k][m]` and `Bs[k][n]`. One thread reads its
eight by eight register tile with four sixteen-byte loads: two for A and two for
B. The sixteen threads of one block row read the same eight A values, so the read
broadcasts inside a warp and no bank repeats. That is what makes the staged read
conflict free without a pad that costs a block its second slot on the
multiprocessor. The cost is one transposing copy for each operand form whose
memory runs along the other axis: a plain A and a transposed B, which is the
forward pass's form.

**The two schedules state one trade.**

| Schedule | Tile | Threads | Window | Blocks on `sm_75` |
|---|---|---:|---:|---:|
| `wide` | 128 by 128 by 16 | 256 | 33,792 B | 1 |
| `half` | 128 by 64 by 16 | 128 | 25,600 B | 2 |

`sm_75` holds 64 KiB of shared memory for each multiprocessor. The wide schedule
therefore holds one block and the half schedule holds two, and the second stages
half the bytes for each unit of arithmetic. Which one wins is a card measurement
and not an argument.

**The window is static.** A launch asks for no shared-memory attribute and no
opt-in limit, which is also what lets the emulation engine run the kernel: its
`__shared__` mapping covers a static array and not a dynamic window.

**The build reads clean.** All eight instantiations hold between 118 and 128
registers, with `STACK 0` and no local access. `__launch_bounds__` states two
blocks for each multiprocessor, which keeps ptxas inside the 128 registers that
two blocks of 256 threads need.

**The gates, and the two that need a card.**

```bash
tools/nanochat test //backends/cuda/gemm:gemm_fp32_plan_test
tools/nanochat test //tools/cuda_sim/emu:gemm_fp32_emu_test
modal run tools/modal/app.py::card_job --name gemm_fp32_check
modal run tools/modal/app.py::card_job --name gemm_fp32_bench \
  --argv '["--rows","production","--schedule","all","--json"]'
```

The first two need no card. The emulated leg is `gemm_fp32_emu_test`, which runs
the real source through the engine against a host reference in double.

**Nothing is promoted by this change.** `gemm_fp32::kPromoted` is off, so an
unset `NANOCHAT_GEMM_LIB` leaves every fp32 call on cuBLAS. The card run above
decides whether the bit turns on.

**Both card gates are open.** Three runs of the check job stopped in the image
build, and each one stopped on an upstream fetch: the Modal builder answered
`504 Gateway Time-out` for the `rules_cuda` tarball, while the same URL answered
200 from the development host. Nothing in this package reached a card yet, so
the tile is `Correct off device, and the card tiers are pending`.

