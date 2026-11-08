# The hand tensor-core GEMM

This package is the hand-written bf16 GEMM that
[docs/kernels.md](../../docs/kernels.md) Part 3 describes. It implements the full
`GemmParams` contract: four operand layouts, a batch, three leading dimensions,
three strides, an alpha and beta epilogue, and the fp32 store of `GemmWgrad`.

**The measurement says no shape takes this path.** Every row of the closed list
is inert, so cuBLAS serves every shape of a training step. Part 6 gives the
numbers and the decision.

## 1. The files

| File | What |
|---|---|
| `gemm_tile.cuh` | The staged operand tile: the two copies, the k slice, the pipeline |
| `gemm_plan.h` | The four schedules, the staged layout math, and the sweep's names |
| `gemm_bf16.cu` | The kernel, the epilogue, and the launcher |
| `gemm_launch.h` | The two backend-internal launchers |
| `gemm_shape.h` | `ResolvedGemm`, `ResolveGemm`, and the tile's own refusals |
| `linear_shape.h` | The three calls of one Linear layer, from a `Config` |
| `dispatch.{h,cc}` | The closed shape list, the verdict, and the switch |
| `gemm_tile_test.cc` | Not present: the host-side plan is checked by `dispatch_test` |
| `gemm_gpu_test.cc` | The device gate: the tile against cuBLAS |
| `gemm_benchmark.cc` | The rate gate: the seam against cuBLAS over the closed list |
| `dispatch_test.cc` | The CPU gate of the table, the verdict, and the seam's defaults |

## 2. The staged tile

One block owns a `TileM` by `TileN` rectangle of C and walks k in slices of
`TileK`. A slice is staged into shared memory in the layout the fragments read,
and the pipeline is double buffered.

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

All four fit the 48 KiB static shared window of a Turing card, so none needs
`cudaFuncSetAttribute` and the emulator needs no dynamic shared memory.

| Schedule | Tile | Warps | k depth | Registers | Shared |
|---|---|---|---:|---:|---:|
| `128x64x32` | 128 by 64 | 2 by 2 | 32 | 94 to 96 | 31,744 B |
| `64x64x32` | 64 by 64 | 1 by 2 | 32 | 93 to 96 | 21,504 B |
| `64x64x64` | 64 by 64 | 1 by 2 | 64 | 91 to 95 | 37,888 B |
| `128x128x32` | 128 by 128 | 4 by 2 | 32 | 92 to 96 | 41,984 B |

Every schedule is two buffered. A third staged buffer was measured as
`128x128x16x3`, it won no shape, and it was removed with the general stage count
it needed. The register and shared figures are `cuobjdump -res-usage` of one
build at bf16 and `sm_90`, read over the four operand forms and the two stores of
each schedule. The shared figure is the staged window and nothing else: the four
bytes of alignment the compiler adds and the epilogue's exchange tiles, which lie
over that window. Section 12 records the register figure that the register
epilogue took from 120 to 92.

`128x128x32` is the default, because it won the sweep. A row of the shape table
names its own schedule, so a future promotion does not need a second sweep.

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
| Off device: the layout, the bounds, the pipeline, the epilogue | `tools/nanochat simulate --device h100 --suite gemm` |
| The shape table, the verdict, the switch, the seam's defaults | `tools/nanochat test //backends/cuda/gemm:dispatch_test` |
| Device: the tile against cuBLAS on every contract case and schedule | `tools/nanochat test --gpu --config=hopper //backends/cuda/gemm:gemm_gpu_test` |
| Rate: the seam against cuBLAS over the closed list | `tools/nanochat bench -- bazel-bin/backends/cuda/gemm/gemm_benchmark --schedule all --json` |

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
