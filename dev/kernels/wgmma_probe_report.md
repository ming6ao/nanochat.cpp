# WGMMA probe report — Part 3 phase 0

This report records what phase 0 of [docs/kernels.md](../../docs/kernels.md)
Part 3 measured, on which leg, and what the result changes about the plan. It
covers phase 0 only. Phases 1 to 5 of that plan are not implemented, and section
7 says why.

**The source moved, and the rows were re-recorded.** The probe that first
produced these rows was `dev/kernels/wgmma_probe.cu`, and that file is retired.
Its rows are now the `wgmma:`-prefixed rows of `dev/kernels/wgmma_bench.cu`,
which absorbed the probe when it gained the Phase H1 rows of
[docs/attention.md](../../docs/attention.md) Part 8. The rates in section 2 are a
new run of that source, over the measurement protocol of
`backends/cuda/kernels/testing/bench_protocol.h`. The one check the first run
cited, the sum of the accumulators, is withdrawn as blind to the operand form,
and the source checks one element at a time instead. Read part 3 phase 0 of the
plan with that in mind: the rate rows are reproducible, and the old check is not
quoted as evidence.

The plan's question for phase 0 is one number. How fast can one warpgroup issue
the Hopper warpgroup matrix multiply-accumulate from a resident shared-memory
tile? The rest of the plan depends on the answer. If the instruction cannot hold
the per-SM tensor-core rate, no mainloop built on it can hold the card rate, and
the campaign stops without a kernel.

## 1. The verdict

**Gate 0 passes with margin.** One warpgroup holds 99.0 percent of the per-SM
tensor-core rate on a resident tile. Two warpgroups on every SM hold 98.3 percent
of the card's declared bf16 peak. The gate asked for 85 percent of the per-SM
share.

So the instruction is not the gap. The shipped tile holds 155 to 160 TFLOP/s on
the six rows that matter. cuBLAS holds 644 to 868 TFLOP/s on those rows. That
difference now sits entirely in the data path: the operand copy, the shared
layout, the pipeline depth, the epilogue, and the occupancy. Phase 2 has roughly
98 percent of the
peak to feed from, and cuBLAS leaves 12 to 35 percent of it unused.

## 2. The rows

Source: [baselines.json](../../docs/baselines.json), entry `wgmma_probe`. One
run of one binary per row on an NVIDIA H100 80GB HBM3 at `sm_90a`, bf16.

| Row | Blocks | Passes | Time | Rate | Share of the number it is measured against |
|---|---:|---:|---:|---:|---|
| `1block` | 1 | 20000 | 2.828 ms | 7.415 TFLOP/s | 99.0 percent of the per-SM share, 7.4924 TFLOP/s |
| `grid` | 264 | 20000 | 5.696 ms | 972.1 TFLOP/s | 98.3 percent of the card peak, 989 TFLOP/s |

The per-SM share is the declared peak over 132 SMs. The card reported 132 SMs.

**The statistic.** Each row is the minimum of three windows, and each window
holds three calls, so the time is a steady state and not one launch. The
second-level cache is flushed before every call, and the measured cost of one
flush, 0.0347 ms, is removed from the window. The protocol is
`backends/cuda/kernels/testing/bench_protocol.h`, which the hand GEMM sweep of
`backends/cuda/gemm/gemm_benchmark.cc` also times through, so a row here and a
row there state the same statistic. Both rows read slightly above the single-launch
figures they replace: 98.6 and 96.4 percent. The movement is the steady state and
not a change to the kernel.

## 3. The correctness leg

The probe checks its own product before it reports a rate. The source now checks
one element at a time against a host reference over the same input, because a
sum-based check cannot see a permutation of the tile: that permutation leaves the
sum unchanged. A descriptor that reads the wrong tile or the wrong operand form
therefore fails, and the failing element is named.

The measured relative error was `1.9e-07` against a reference scale of `14.68`,
which is the bf16 rounding of the reference and not a defect of the descriptor.
The staged rows are checked against the resident product as well: both stagings
report 0 bad elements of 8192, so a wrong staging fails instead of measuring.

The shipped tile also passed its own device gate on the same card. The card job
printed `gemm gpu test: device NVIDIA H100 80GB HBM3, compute capability 9.0`
and then `gemm gpu test: ok`, over the whole `GemmParams` contract and every
schedule of the sweep.

## 4. Where the shipped tile stands, measured in this session

The same card, the same binary, `--schedule all`. The best schedule was
`128x128x32` on all six rows. Both paths do the same arithmetic, so the fraction
is the column the plan's gate reads.

| Row | cuBLAS | Hand | Fraction | Hand rate | cuBLAS rate |
|---|---:|---:|---:|---:|---:|
| Muon `A = XᵀX`, 1280 by 1280 over 40 | 0.2319 ms | 1.1004 ms | 0.211 | 152 TFLOP/s | 724 TFLOP/s |
| Muon `prod`, 5120 by 1280 over 20 | 0.5213 ms | 2.3845 ms | 0.219 | 141 TFLOP/s | 644 TFLOP/s |
| Linear forward `fc` | 0.2592 ms | 1.3633 ms | 0.190 | 158 TFLOP/s | 828 TFLOP/s |
| Linear dgrad `cproj` | 0.2584 ms | 1.5208 ms | 0.170 | 141 TFLOP/s | 831 TFLOP/s |
| Linear wgrad `fc` | 0.3050 ms | 1.9679 ms | 0.155 | 109 TFLOP/s | 704 TFLOP/s |
| Classifier forward | 1.5835 ms | 8.6150 ms | 0.184 | 160 TFLOP/s | 868 TFLOP/s |

These rows agree with the `gemm_bf16` entry within three percent. The hand rate
moves between 109 and 160 TFLOP/s across shapes that differ by six orders of
magnitude in work. That is the flat curve the package README section 7 already
records. A flat curve means the limit is inside the kernel and not in the operand
traffic.

The same run put cuBLAS at 65 to 88 percent of the peak on these rows, and at
94.6 percent on the classifier dgrad row. The instruction ceiling measured in
section 2 is 98 percent. So cuBLAS is within four points of that ceiling on its
best row, and 10 to 33 points under it on the rest.

## 5. What the probe is, and what it is not

The probe stages one query tile of 16 KiB and one key tile of 32 KiB. The ring
holds two query tiles and four key tiles, which is 160 KiB of dynamic shared
memory, and one warpgroup issues four instructions for each pass. Three pieces
are hand written, because phase 2 must not depend on a library:

- the shared-memory matrix descriptor, encoded from the sixteen-byte fields the
  instruction reads;
- the interleaved K-major staging pass, which is the descriptor form whose
  address arithmetic is short enough to check by hand;
- `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16` with its sixty-four
  fp32 accumulators.

**It is an upper bound and not a projection.** The timed loop reads no global
memory and writes none, so it measures the instruction and the shared-memory
descriptor path only. A real mainloop adds the copy, the barrier, the pipeline,
and the epilogue, and each of those is a cost this probe does not carry. The
`grid` row says what a full card of instructions can reach; it does not say what
a full card of a mainloop will reach.

## 6. Three defects the work found, and their fixes

1. **The staged window was a quarter of the size the addressing assumed.** The
   k-block stride and the whole-tile size were one constant, so the descriptors
   walked 4 times past the end of the operand region. The first device run
   faulted with `XID 13, Out Of Range Address` on the SM. The fix separates the
   two figures, and an independent check now prints the furthest byte any
   descriptor reads beside the size of the tile it reads from. That check reads
   8190 of 8192 bytes for A and 16382 of 16384 for B.
2. **The accumulators needed an operand fence.** ptxas reported that the
   instruction would be serialized against the ordinary stores that define the
   same registers. An empty read-write asm over the sixty-four accumulators
   bounds the region. The build then reports 90 registers, no local memory, and
   back-to-back `HGMMA` in the SASS. The advisory survives the fix and the SASS
   shows the loop is tight, so it is recorded here as unproven rather than
   fixed.
3. **`sm_90a` is a new architecture string for this tree.** WGMMA is
   architecture conditional, and every device configuration in `.bazelrc` emits
   `sm_90`. The probe carries a tripwire that refuses to compile below `sm_90a`,
   and the card job builds it inside the container with the tree's builder and
   that one architecture. No other target and no other job changed its
   architecture.

## 7. What was not done, and why

Phases 1 to 5 are not implemented. The plan-sized estimate for them is four to
ten months of one GPU engineer with a card in the loop. The honest statement is
that no single session produces them. What this session did produce is the phase
0 decision point, its harness, and the two legs the later gates need.

| Phase | State |
|---|---|
| 0, the ceiling probe | Done, measured, gate passed |
| 1, the `mma.sync` mainloop | Not started. It needs its own emulator mapping for `ldmatrix` and `mma.sync`, a four-to-six stage pipeline, dynamic shared memory, and the epilogue rewrite |
| 2, the WGMMA mainloop | Not started. Section 2 says the ceiling allows it. Section 6 row 3 says the architecture work is ready for it |
| 3, the fp32 tile for Turing | Not started |
| 4, fp8, stream-K, and the tiny group | Not started |
| 5, detaching the library | Not started |

The first task of phase 1 is the one the plan says it is: the operand path. The
measured gap is now the copy and the epilogue and nothing else.

## 8. Reproduction

Every command runs through the tree's entry point.

```bash
# The emulator, on the CPU of this host. No card and no queue.
tools/nanochat test //tools/cuda_sim/emu:gemm_emu_test
tools/nanochat simulate --device h100 --suite gemm --precision bf16
tools/nanochat test //backends/cuda/gemm:all

# The card, through the Modal H100 leg (tools/modal/app.py).
modal run tools/modal/app.py::card_job --name wgmma_bench \
  --argv '["--rows","phase0","--json"]'
modal run tools/modal/app.py::card_job --name gemm_check
modal run tools/modal/app.py::card_job --name gemm_bench \
  --argv '["--schedule","all","--json"]'

# The probe alone, on a host that has a Hopper card.
tools/nanochat build //dev/kernels:wgmma_bench
```

## 9. Limits of this report

- The report states one run per row. The spread is unmeasured, and the
  `gemm_bf16` entry records a three percent run-to-run spread on the same binary.
- The per-SM share divides the declared peak by the SM count. It does not read
  the clock of this card during the run, and a boost-clock change moves both rows
  together.
- The element check covers the resident 8-atom tile. It fixes the operand form
  the instruction reads, and it does not prove each element of the staged rows.
  Those rows are checked against the resident product instead, so a staging that
  permutes the tile consistently would pass.
- The two rows use different block counts, so they are two measurements and not
  a scaling curve.
