# WGMMA probe report — Part 3 phase 0 and Part 8 phase H1

This report records what phase 0 of [docs/kernels.md](../../docs/kernels.md)
Part 3 and phase H1 of [docs/attention.md](../../docs/attention.md) Part 8
measured, on which leg, and what the result changes about the plan. It covers
those two phases only. Phases 1 to 5 of the Part 3 plan are not implemented, and
section 7 says why. Section 10 carries the staged ladder that closes most of the
gap the earlier sections located.

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

**The staged rows gained a second dimension.** Section 10 reports it: a block of
two warpgroups shares one ring of staged key tiles, so one copy of a key tile
buys twice the arithmetic. Every rate in sections 1 to 8 is unchanged, and the
staged figures those sections quote are the one-warpgroup rows that the
recorded baselines entry holds.

**What this report does not cover.** The same binary also carries the
`attention-flash:` rows of a resident flash tile, which issue the score product,
the base-two softmax in registers, and the value product as one serialized
chain. That row set is another workstream's subject and this report states no
figure from it.

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
The staged rows are checked against the resident product as well, over all six
geometries of section 10 and both staging paths: twelve checks, and every one
reports 0 bad elements of up to 49,152. Each query tile holds a different
operand, so a warpgroup that reads another tile of the query region fails
there, and so does a wrong stage offset, a wrong ring, or a missing barrier.

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

The probe stages one query tile of 16 KiB for every query tile of every
warpgroup, and one key tile of 32 KiB for every stage of the ring. The deepest
geometry of section 10 is two warpgroups of three query tiles each beside a
four-stage ring, which is 96 KiB of query tiles and 128 KiB of key tiles, or
224 KiB of the 227 KiB one block may request. A warpgroup issues eight
instructions for each staged key tile. Three pieces are hand written, because
phase 2 must not depend on a library:

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
| 0b, the staged ladder of section 10 | Done, measured. Two warpgroups that share one ring of staged key tiles raise the staged row from 295.4 to 642.0 TFLOP/s, which is 64.9 percent of the card peak. The copies stop being the limit at the deepest geometry, and the shared issue path of the copies and the products becomes it |
| 1, the `mma.sync` mainloop | Not started. It needs its own emulator mapping for `ldmatrix` and `mma.sync`, a four-to-six stage pipeline, dynamic shared memory, and the epilogue rewrite |
| 2, the WGMMA mainloop | Not started. Section 2 says the ceiling allows it. Section 6 row 3 says the architecture work is ready for it |
| 3, the fp32 tile for Turing | Not started |
| 4, fp8, stream-K, and the tiny group | Not started |
| 5, detaching the library | Not started |

The first task of phase 1 is the one the plan says it is: the operand path. The
measured gap is now the copy, the issue path that the copies share with the
products, and the epilogue. Section 10 states how much of it each one holds.

## 8. Reproduction

Every command runs through the tree's entry point.

```bash
# The emulator, on the CPU of this host. No card and no queue.
tools/nanochat test //tools/cuda_sim/emu:gemm_emu_test
tools/nanochat simulate --device h100 --suite gemm --precision bf16
tools/nanochat test //backends/cuda/gemm:all

# The card, through the Modal H100 leg (tools/modal/app.py). `all` states both
# row sets, and the staged ladder of section 10 rides the h1 set.
modal run tools/modal/app.py::card_job --name wgmma_bench \
  --argv '["--rows","all","--json"]'
modal run tools/modal/app.py::card_job --name gemm_check
modal run tools/modal/app.py::card_job --name gemm_bench \
  --argv '["--schedule","all","--json"]'

# The probe alone, on a host that has a Hopper card.
tools/nanochat build //dev/kernels:wgmma_bench
```

## 9. Limits of this report

- The report states one run per row, except for the ladder of section 10, which
  states three. The `gemm_bf16` entry records a three percent run-to-run spread
  on the same binary, and the three ladder runs differ by 0.7 percent at most.
- The per-SM share divides the declared peak by the SM count. It does not read
  the clock of this card during the run, and a boost-clock change moves both rows
  together.
- The element check covers the resident 8-atom tile. It fixes the operand form
  the instruction reads, and it does not prove each element of the staged rows.
  Those rows are checked against the resident product instead, so a staging that
  permutes the tile consistently would pass. Every query tile now holds a
  different operand, so a warpgroup that reads another tile of the query region
  fails, and the check covers all six geometries of section 10.
- The two rows use different block counts, so they are two measurements and not
  a scaling curve.
- The ladder of section 10 states the best of four window and copy-order
  combinations for each geometry, and three runs of that statement. The four
  rows of one geometry span up to ten percent, and the shape of each row is in
  the document a run writes.

## 10. The staged ladder: what one staged byte buys

Sections 2 and 5 state the ceiling of the instruction and the question the
staged rows ask. This section turns that question into one product and measures
every point of it.

**The model.** A staged row can not exceed

```
rate = bandwidth of the copy path x arithmetic that one staged byte buys
```

One staged key tile is 128 keys by 128 head elements of bf16, which is 32 KiB.
The score product over it is 2.10 MFLOP for every query tile that reads it, so
one staged key byte buys 64 FLOP for each query row behind that tile. A block of
two warpgroups shares one ring of key tiles: the copies of a key tile are issued
once, and both warpgroups multiply it. So the count of query rows behind one
staged key tile is the whole lever, and it is

```
arithmetic per staged byte = 64 x warpgroups x query_tiles   FLOP
```

**The change.** `StagedKernel` carries the warpgroup count and the query-tile
count as template parameters. The ring is one region of the block, so the copies
of a key tile are shared. The query region holds one tile for every query tile
of every warpgroup, and the threads of the block stage it and copy the key tile
together. No instruction changed, and no layout changed: the descriptor, the
interleaved form, and the copy path are the ones section 6 records.

**The rows.** Three runs of one command, the card of section 2, the statistic of
section 2. The cap of a row is the copy rate of its own store order times its
arithmetic, and that copy rate is 2.36 TB/s for the `k`-block-first order and
2.44 for the row-first order, both measured in these runs. Each row below is the
best of those two orders and the two source windows, and the three runs agree to
within 0.7 percent. The third run measured the source that also carries the
`attention-flash:` rows of another workstream, so the ladder does not move when
a second row set shares the binary.

| Warpgroups | Query tiles | FLOP a staged byte | Cap | Run 1 | Run 2 | Share of the cap | Copy reached | Share of the peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 64 | 156 | 151.2 | 151.7 | 97.1 | 2.36 TB/s | 15.3 percent |
| 1 | 2 | 128 | 302 | 293.2 | 293.8 | 97.2 | 2.29 | 29.7 percent |
| 1 | 3 | 192 | 453 | 424.3 | 424.9 | 93.8 | 2.21 | 43.0 percent |
| 2 | 1 | 128 | 312 | 308.5 | 308.5 | 98.8 | 2.41 | 31.2 percent |
| 2 | 2 | 256 | 604 | 535.8 | 535.4 | 88.6 | 2.09 | 54.1 percent |
| 2 | 3 | 384 | 906 | 642.0 | 643.1 | 71.0 | 1.67 | 65.0 percent |

The last three columns are one number three times. `Copy reached` is the rate
over the arithmetic of the row, `share of the cap` is that figure over the copy
rate of the path, and `share of the peak` is the rate over the declared
989 TFLOP/s.

**The correctness leg.** All twelve checks pass in all three runs, which is six
geometries over the plain and the asynchronous staging. Every one reports 0 bad
elements, up to 49,152 of a two-warpgroup row of three query tiles. The element
check of section 3 still reads `1.9e-07` against its reference of `14.68`. The
ring, the barrier, the stage offset, and the warpgroup offset of the tile are
therefore right at every geometry, and no rate above rests on a wrong stage.

**What the rows say.**

1. **The model holds while the copies bind.** Five of the six rows are 89 to 99
   percent of the cap of their own path. The copies reach 2.09 to 2.41 TB/s,
   which is the 2.36 to 2.44 the copy-only rows of this run measure.
2. **A shared ring doubles the rate of a row.** Two warpgroups of one query tile
   read 308.5 TFLOP/s where one warpgroup of two reads 293.2: the same
   arithmetic for the same copies, and the small difference is the schedule.
   Two warpgroups of two query tiles reach 535.8, which is 1.83 times the
   293.2 of the recorded two-query-tile row.
3. **The third query tile pays less than the model.** Its row reads 642.0 of a
   906 cap, and its copies reach only 1.67 TB/s of the 2.36 available. Above
   that point the copies are not the limit: the products and the copies share
   one issue path, and the barrier at the top of every step makes both
   warpgroups wait for each other. The products hold 4.86 TFLOP/s of the 7.49
   per-SM share, which is 65 percent, so neither resource is saturated.
4. **642.0 TFLOP/s is the fastest staged row this bench has.** It is 2.17 times
   the 295.4 that the recorded two-query-tile row states, and it is above the
   604 cap of the row below it.

The rows of these runs also re-read the ceiling rows of section 2 at 971.5 to
978.6 TFLOP/s, and the one-block rows at 7.4. Those figures agree with the
committed entry, so the change did not move a row it did not touch, and the
three runs of this section differ by 0.0 to 0.7 percent on every row. The third
run reads 151.2, 293.6, 424.7, 309.0, 535.0 and 638.7 TFLOP/s over the six
geometries of the table, in the order the table states them.

**What it does not do.**

- The staged rows time the score product only. The value product stages a value
tile as well, so it doubles the staged bytes and doubles the arithmetic: the
forward tile of section 3 keeps the 256 FLOP a staged byte that this ladder
measures at one warpgroup of two query tiles. Its cap is therefore 604 TFLOP/s,
which is the row above the last. Section 3's 262 FLOP a staged byte is the same
figure for a slightly larger value tile.
- The register budget closes the second dimension at three query tiles. Three
tiles of 64 accumulators are 192 registers, and `ptxas` allocates 239 of the 255
a thread may hold, with no spill. A fourth tile does not fit, so the ladder has
no deeper row.
- The deepest geometry is also the largest block the 227 KiB shared-memory
limit allows. Four key stages beside six query tiles is 224 KiB, so the ring
cannot be deepened to decouple the barrier, and a smaller ring reads 2.12 TB/s
where four stages read 2.36.

**What this changes about the plan.** Section 7 of Part 8 puts the Tensor Memory
Accelerator copy path (H2b) before warp specialization (H3), because the copies
were the constraint at every geometry the probe held. That is now true for the
score product only while the intensity is at or below 256 FLOP a staged byte. At
two warpgroups and three query tiles the copies are no longer the limit, and the
remaining gap is the shared issue path that H3 changes and H2b does not. For the
forward tile, which also stages the value tile, the copies still bind at 604
TFLOP/s, so H2b keeps its place for that geometry, and it is now stated as the
fix for a measured cap rather than for the whole tile.
