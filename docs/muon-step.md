# The Muon optimizer step

Status: design and plan of record. No part of section 8 is implemented. Each
phase carries a gate, and each gate names the action when it fails.

This document owns one subject: the shipped multi-launch `MuonUpdate` at the
twenty-layer shape, and the cuBLAS seam it calls. It also prices that step on
the H100 and plans the work that closes the gap beside the GEMM. The fused
megakernel is a different subject, and it is closed in
[kernels.md](kernels.md) Part 1.

Read [model.md](model.md) for the optimizer and its workspace,
[precision.md](precision.md) for the precision policy,
[performance.md](performance.md) section 2.1 for the measurement protocol, and
[kernels.md](kernels.md) for the fusion boundary rule and the compute strategy.

## 1. The questions this document answers

| Question | Answer | Where |
|---|---|---|
| What is the cuBLAS path for the twenty-layer model? | One cached `cublasLt` plan per shape, replayed for 90 batched calls per step | Section 4 |
| How much of the step does the Muon stage own? | 21.6 percent of the step bound on the H100. On the project card the GEMM was 87 percent of the measured stage | Sections 5 and 6 |
| Was the fused kernel's loss a launch problem? | No. It was the GEMM rate. The launch share was 7.5 percent | Section 5 |
| Should we borrow cuBLASDx, CUDA Graphs, or CUTLASS? | CUDA Graphs only. The other two are rejected with reasons | Section 7 |
| Can the twenty-layer model be as fast as cuBLAS on the H100? | Yes, because it already is. The tree calls cuBLAS for every Muon product, and that property is kept | Sections 2 and 7 |
| Which card and precision do we measure on? | One H100 SXM card on Modal at bf16, through the existing entry points | Section 14 |
| What happens when a gate fails? | Section 8 states one action per gate, and no gate leaves a question open | Section 8 |

## 2. The answer in one page

The twenty-layer model already runs its Muon products on cuBLAS. Parity with
cuBLAS is therefore not a target to reach. It is a property to keep, and it is
the property that section 7 protects.

The gap beside the GEMM is small and it is known. One optimizer step at the
shipped `ns_steps` of 5 issues 39 device operations per shape group, and six
groups make 234 launches. It also moves the whole parameter stack through more
passes than the arithmetic needs.

Three changes close that gap, and none of them touches a product:

1. Merge the cautious update into the NorMuon scale, so the stack is read once
   instead of twice.
2. Merge each partial pass with its own finish, with an atomic ticket.
3. Capture the step in a CUDA Graph, so 234 launches become one.

The prize at the twenty-layer shape on one H100 is 0.5 to 1.1 milliseconds of a
13.8 millisecond optimizer phase. Section 6 gives the derivation. That is 1 to
2 percent of the whole step, and the plan says so before it spends a session.

The larger prize is the 4.55 milliseconds of Polar Express traffic that only a
device-side GEMM can remove. Decision D3 rejects that work, and section 6 prices
it. The price is recorded, so a later campaign starts from a number and not from
an opinion.

## 3. The step today

The twenty-layer shape is `PROFILES["big"]` of `python/nanochat_cpp/pipeline.py`:
20 layers, hidden 1280, 10 heads, 5 key/value heads, sequence 1024. The Muon
groups are the ones `muon_benchmark` builds:

| Group | Batch | Elements |
|---|---:|---:|
| 5x12, the value gate | 10 | 600 |
| 640x1280 | 40 | 32,768,000 |
| 1280x1280 | 40 | 65,536,000 |
| 1280x5120 | 20 | 131,072,000 |
| 5120x1280 | 20 | 131,072,000 |
| **Total** | | **360,448,600** |

One step runs these operations, per group:

| Stage | Operations |
|---|---:|
| Momentum and the Nesterov combination | 1 |
| Frobenius norm: partial pass, finish | 2 |
| Row equilibration, one block per row | 1 |
| Equilibration divisor: partial pass, finish, scale | 3 |
| Polar Express, five iterations | 15 GEMMs and 10 elementwise passes |
| Muon+ renormalisation: partial pass, finish, scale | 3 |
| NorMuon mean squares, one per row or column | 1 |
| Second moment and the per-index step | 1 |
| NorMuon scale | 1 |
| Cautious weight decay and the parameter write | 1 |
| **Total** | **39** |

The tensor workspace is `2·total + 2·min_total` elements, and
`EnsureMuonWorkspace` in `backends/cuda/kernels/muon.cu` sizes it once per device
thread. A steady-state step allocates nothing.

## 4. The cuBLAS seam

`kernels::Gemm` and `kernels::GemmWgrad` in `backends/cuda/gemm.cu` are the only
GEMM entry points. Both run one universal call,
`C = alpha*op(A)*op(B) + beta*C`, with `op(X) = Xᵀ` when the transpose flag is
set.

| Case | Call |
|---|---|
| Batch 1, fp32 | `cublasSgemm` |
| Batch 1, any type | `cublasGemmEx` with `CUBLAS_COMPUTE_32F` |
| Batch above 1 | `cublasLtMatmul` with a cached plan |
| No plan for the shape | `cublasGemmStridedBatchedEx` |

Every Muon product takes the third row, because a group is a batched call. The
plan is the part that matters:

- `PlanKey` holds the three extents, the batch count, three leading dimensions,
  three batch strides, two transpose flags, and the output-type flag. One plan
  serves one shape for the life of the thread.
- `LtPlan` holds one matmul descriptor, three matrix layouts, one algorithm, and
  one workspace.
- The batch rides on the layouts, through
  `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` and
  `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`.
- The workspace request is 32 MiB. The heuristic returns 8 candidates. The first
  call of a shape times all of them and keeps the fastest, because the first
  answer of the heuristic was the cost and not the query.
- Row-major becomes column-major by swapping the operands. The matmul's A is the
  stored B, so `CUBLASLT_MATMUL_DESC_TRANSA` receives `transpose_b`.

The operand type is `CUDA_R_16BF` when the build is bf16, else `CUDA_R_32F`. On
the H100 leg, which builds bf16, the Muon products therefore run on tensor cores
already. The handle and the plan cache are per device thread, so two device
threads never share one plan.

## 5. The measured record

The Muon fusion campaign measured the stage before it wrote a kernel, and its
rows are in [baselines.json](baselines.json) under `muon`. Four numbers decide
the design of this document:

| Number | Value | Source |
|---|---:|---|
| Launch and elementwise share of the step, `big` profile | 7.2 to 7.5 percent | `muon.phase0_share` |
| Polar Express GEMM share of the step, `big` profile | 87 percent | `muon.phase0_share` |
| Hand batched GEMM rate against cuBLAS, 640x1280 | 0.68 to 0.72 | `muon.phase1_gemm` |
| One fused iteration against 41.9 microseconds of launch gaps | 4.7 microseconds | `muon.sync` |

The fusion lost 45 percent at the `big` step on the project card. The loss came
from the GEMM, and not from the launch gaps, which the same rows state. The
design is [kernels.md](kernels.md) Part 1.

The lesson this document takes from that campaign is one line: keep the GEMM on
cuBLAS, and own the rest.

## 6. The cost at the twenty-layer shape on the H100

`tools/nanochat roofline` bounds one step of the twenty-layer shape on one H100
at batch 16. This is the command, and the two precision rows are its answer:

```bash
tools/nanochat roofline --devices 1 --device h100 --precision bf16 \
  --layers 20 --heads 10 --kv-heads 5 --hidden 1280 --seq 1024 \
  --vocab 32768 --window-pattern SSL --batch 16 --grad-accum 1 \
  --muon-ns-steps 5
```

| Row | bf16 | fp32 |
|---|---:|---:|
| Forward | 17.7 ms | 215.9 ms |
| Backward | 32.5 ms | 426.6 ms |
| Optimizer phase | 13.8 ms | 161.4 ms |
| — the optimizer state pass | 3.1 ms | 3.1 ms |
| — the Muon orthogonalisation | 10.7 ms | 158.3 ms |
| Step bound | 64.1 ms | 804.0 ms |

The fp32 column is the same command with `--precision fp32`. The two optimizer
rows do not move with the sequence length, because the Muon arithmetic reads the
parameter stack alone.

The optimizer phase is 21.6 percent of the bf16 step bound, and the
orthogonalisation alone is 16.7 percent. The arithmetic is 1.06e13 FLOP per step,
so the 10.7 ms row is the whole bf16 peak of the card and no headroom.

Two consequences follow, and both are decisions:

1. The campaign targets the bf16 build. The fp32 peak of the card is 14.8 times
   lower, so an fp32 product cannot compete on this card at any schedule.
2. The traffic beside the products matters more on this card, because the gain
   in arithmetic is larger than the gain in bandwidth. This table derives it
   per step at bf16, from the element counts of section 3 and the declared
   bandwidth of 3.35 TB/s (`backends/cuda/caps.json`, entry `h100`):

| Traffic | Bytes | Time |
|---|---:|---:|
| `x = ca·x + prod`, five iterations | 2.16 GB | 3.23 ms |
| `b = cb·a + cc·a2`, five iterations | 0.88 GB | 1.32 ms |
| **Polar Express elementwise traffic** | **3.05 GB** | **4.55 ms** |
| The cautious update, merged into the scale | 1.44 GB | 0.43 ms |
| 234 launches at the declared 3.0 microseconds | — | 0.70 ms |

The third row is the prize that only a device-side GEMM can reach, because the
pass sits between two products. Decision D3 rejects it and this table prices it
at 7.1 percent of the step bound. The fourth and fifth rows are the prize this
campaign takes.

### 6.1 The measurement on the H100

One run of `--profile big --muon` on one H100 SXM card at bf16, revision
`e47f71c`:

| Row | Per step |
|---|---:|
| `polar:big:step`, the five Polar Express products | 15.18 ms |
| `muon:big:step`, ns=5, the shipped stage | 29.09 ms |
| `muon:big:step`, ns=1 | 12.34 ms |
| `muon:big:step`, ns=0, every operation but the iterations | 8.12 ms |

The products run at 70.6 percent of the bf16 peak and the stage at 36.8
percent. The non-product part of the stage is 13.91 ms, or 47.8 percent.
The `ns=0` control says where it comes from: 8.12 ms is the fixed part of
the update, and each Polar Express iteration adds 4.19 ms, of which 3.04 ms
is its products and 1.16 ms is the elementwise traffic and the launches
beside them.

Section 6 forecast a prize of 0.5 to 1.1 ms, because it priced two
elementwise passes and 234 launches. The measurement puts 13.91 ms beside
the products, so the forecast is a floor and not a bound: the traffic merge
and the graph reach the 5.79 ms of Polar Express traffic and launches, and
the 8.12 ms of the fixed part is a separate item. D7's 0.5 ms criterion is
kept as the floor it was written as, and this table is the number a later
campaign starts from.

## 7. Decisions

Each decision is closed. A reason and a rejected alternative are stated for each
one, and no decision waits on a measurement.

| Number | Decision | Reason | Rejected alternative |
|---|---|---|---|
| D1 | Measure on one H100 SXM card at bf16 | The H100 leg of `tools/modal/app.py` is that build, and the bf16 step bound is 12.5 times below the fp32 one | An fp32 leg on `sm_90` |
| D2 | Keep `kernels::Gemm` as the one product engine, on every device and in every build | The seam already reaches a tuned batched plan, and the measured hand GEMM reached 0.68 to 0.72 of the cuBLAS rate | A hand batched GEMM |
| D3 | Do not port the fused path, and keep `kMuonFusedMaxElements` at 0 | The fused schedule is fp32 and loads `float4` quads, so it cannot run on the bf16 leg. A bf16 tensor-core rewrite is a new prototype with a new gate | cuBLASDx, CUTLASS, or a new hand GEMM |
| D4 | Remove the launch gaps with a CUDA Graph, one graph per `(group, ns_steps)` | It keeps cuBLAS and removes 0.70 ms of launch cost with no arithmetic change | A cooperative fused launch |
| D5 | Merge stages inside a family, and never across a whole-matrix statistic | The `ns=0` row proves that one block size for sixteen phases costs 6 to 10 percent | One kernel for all phases |
| D6 | Move bytes, not arithmetic: every merge must stay bit exact | The seam tolerance is 1e-5 against `testing/optim_ref.h`, and a bit-exact merge needs no new gate | A tolerance change |
| D7 | The acceptance criterion is 0.5 ms of the optimizer phase at `big`, and no group slower | Section 6 derives 1.13 ms of launch and traffic, and half of it is a safe floor | A percent of the step |
| D8 | The fp32 leg stays the reference, and bf16 stays the target | `precision.md` puts bf16 on Ampere and newer, and the parity work reads the fp32 leg | A single precision |
| D9 | The device-resident state stays fp32 master and fp32 momentum, with `ComputeType` parameters | That is the shipped policy, and a merge never changes a stored value | bf16 state |

D3 needs one more line, because it is the decision with a price. The work it
rejects reopens only when a device-side bf16 GEMM measures at or above 0.85 of
cuBLAS on every group of the `big` profile. That measurement is the first phase
of that campaign, not of this one. Nothing else reopens D3.

## 8. The plan

Six phases. One phase is one session. A failed gate stops the plan and the
result is recorded.

### Phase 0: fix the two blocking defects

Section 9 lists them. The H100 leg cannot measure the Muon stage until both are
closed.

Gate: `//backends/cuda/kernels:muon_benchmark` builds at bf16, `--muon` runs
without a device fault, and `//backends/cuda/kernels:muon_fused_gpu_test` stays
green at fp32.

If the gate fails: fix the defect that failed and do not start Phase 1.

### Phase 1: measure the baseline on the H100

Run the two commands of section 14 and record the answer. This phase writes no
code.

Gate: the rows exist, and the launch and elementwise share of the measured step
is at or above 2 percent. That is the floor of the fusion campaign, and it is
reused so the two campaigns compare.

If the gate fails: stop, record a negative result in this document, and close
the campaign with no code. The Muon stage is then below the measurement floor
on this card.

### Phase 2: the two traffic merges

Each merge is an independent row, and each one lands alone with its own test.

1. Fold the cautious update into the NorMuon scale, so the stack is read once.
   The fused design already proved this fold bit exact in
   [kernels.md](kernels.md) Part 1 section 4, so the arithmetic is known.
2. Merge each partial pass with its finish, with a `__threadfence` and an atomic
   ticket, so the last block sums the same partials in the same order.

Gate, and every row of it must hold:

1. The result is bit exact against the shipped path.
2. The T1 GPU test is green, and the CPU oracle test is green.
3. The `big` step on the project card is no slower.

If a gate fails for one merge: revert that merge, keep the ones that passed, and
record the failed one with its measurement.

### Phase 3: the CUDA Graph

Capture one graph per `(group, ns_steps)` and replay it. The cache is thread
local, like the plan cache.

Three constraints fix the design, and each one is a fact of the tree:

1. The first call of a shape builds the plan, queries 8 candidates, times them,
   and allocates the workspace. All of that must happen before the capture.
   `MuonUpdate` warms the group once at the first call and then captures.
2. `GemmScaledFp8` allocates inside its own call. It is not on the Muon path, so
   the capture is not widened to it. A later caller that adds an allocation
   inside a captured region fails the capture, and the fallback is the
   multi-launch path with one log line.
3. No host read sits inside the step. `ScalarArg::host()` reads a device value
   from device code, and the statistics finish on the device, so the step is
   capturable as it stands.

Gate: the graph replay is bit exact against Phase 2, one graph exists per group,
and the `big` step is at or above the D7 criterion on the project card.

If the gate fails: keep the multi-launch path, record the capture status, and
report the negative result. The Phase 2 merges stay landed, because they do not
depend on the graph.

### Phase 4: measure the result on the H100

Run Phase 1's two commands again at the same revision plus the two merges and the
graph.

Gate: the D7 criterion holds on the H100, and every `muon:` group row is no
slower than its baseline.

If the gate fails: revert the phase that the rows name, record both runs, and
close the campaign with the measurement.

### Phase 5: record and retire

Add the H100 run to `docs/baselines.json` under `muon` as a second device block,
with `validated_on` naming the H100. Update "The fused path on this card" in
`backends/cuda/kernels/README.md` with the new rows. Then retire this document
by section 15.

## 9. The defects

The identifiers `U1` and `U2` belong to this campaign.

- `U1`: **the fused benchmark rows are unsafe on a bf16 build.**
  `muon_benchmark` calls `MuonFusedUpdate` directly and sizes its workspace in
  `ComputeType` elements. On a bf16 build that buffer holds half the bytes the
  fp32 fused code writes, so the row writes past its end. The H100 leg is a bf16
  build, so this must close before any H100 run of `--muon`. The guard on the
  build precision skips the fused rows and says so once; it is landed, and it
  is what lets the H100 leg run `--muon`.
- `U2`: **`bench_h100` drops the fused rows.** The function keeps only lines
  that start with `polar:` or `muon:`, and `muon_fused:` starts with neither. The
  fix is one line, and it belongs to the Python surface.

## 10. The items this campaign closes

The fusion campaign leaves six open items. Four of them touch this
subject, and this document closes them:

| Item of the fusion campaign | Decision here |
|---|---|
| 2. Promotion is one measurement and one constant | Closed by D3: the constant stays 0, and the reopen condition is the 0.85 rate gate of section 7 |
| 3. A different GEMM schedule is a different prototype | Closed by D3: no schedule is measured in this campaign |
| 4. The bf16 build stays on the multi-launch path | Confirmed. That path is the subject of this document |
| 5. The cooperative driver keeps no emulator leg | Unaffected. The driver is not on the shipped path |

The two remaining items stay open, and neither is a question for this document.
Item 1 is the dangling citations in code comments, and each one belongs to the
directory that owns it. Item 6 is the simulator suite in an acceptance set, and
it applies to any phase that changes the simulator. This plan changes no
simulator file.

## 11. Non-goals

- A device-side GEMM in any form. That is D3.
- A change of arithmetic, a tolerance change, or a change of the stored state.
- A graph over the rest of the training step. The forward, the backward, the
  gradient sync, and the data path belong to their own workstreams.
- A change to the frozen seam, to `caps.json`, or to a capability bit.
- A new shape, batch, or `ns_steps` value.
- Multi-card behaviour. One card, one step.

## 12. Risks

| Risk | Response |
|---|---|
| The H100 launch share is below the 2 percent floor | Phase 1 stops the plan and records a negative result |
| The graph capture picks up an allocation | Warm the plan and the workspace first, and fall back to the multi-launch path with one log line |
| A merge changes a sum order and breaks the 1e-5 gate | D6 requires bit exactness, so a failed merge is reverted and not tuned |
| The graph replay hides a host-side dependency | Capture per group, and compare the replay against the uncaptured path bit for bit |
| cuBLAS is far below its bound on the 1280 shapes | That is a plan question and not a launch question. It is recorded in Phase 1, and it is a separate campaign |
| The prize is small | Section 6 states 0.5 to 1.1 ms before the first session, so the cost is known in advance |

## 13. Ownership and the Definition of Done

| Work | Owner |
|---|---|
| `backends/cuda/kernels/muon.cu`, `muon_fused.cu`, `muon_benchmark.cc`, and the T1 tests | The kernel agent of the Muon family |
| The graph facility beside `kernels::Gemm` | The kernel agent that owns `backends/cuda/gemm.cu` |
| `tools/modal/app.py` | The Python surface |
| `docs/baselines.json`, `docs/kernels.md`, and this document | The architect |

The Definition of Done for a merge is the checklist of `AGENTS.md` section 5.5.
For this subject it adds four rows:

1. `tools/nanochat lint` passes.
2. `//backends/cuda/kernels:muon_gpu_test` and `muon_fused_gpu_test` pass at
   fp32, and the bf16 leg builds and passes.
3. `tools/nanochat test` passes.
4. One H100 pair of runs at one revision is in `docs/baselines.json`, with the
   device and the precision named.

## 14. The measurement protocol

Two commands, one card, one container start each. Both already exist.

```bash
# The baseline rows and, at Phase 4, the result. `--profile big` is the
# twenty-layer shape, and `--muon` adds the step rows and the ns=0 control.
modal run tools/modal/app.py::pipeline_demo --legs h100 \
  --bench '["--profile","big","--muon","--json"]'

# The phase split of a real step: fwd_ms, bwd_ms, opt_ms, sync_ms. The split
# runs on the H100 leg by construction, so it takes no `--legs`.
modal run tools/modal/app.py::pipeline_demo \
  --split-job "$(cat big-job.json)" \
  --split-corpus "$(cat split-corpus.json)" --split-steps 6
```

The split takes a corpus document and not `--corpus-urls`, because the split
branch of `pipeline_demo` returns before the corpus block runs. The document
holds `files` and `tokenizer`, and `corpus_plan` in `tools/modal/app.py` defines
both from the URLs the chain already downloaded.

The statistic is the one the fusion campaign used: the best of three rounds per
row, device-event timing, one synchronization at the end. `profile_stage` and its
Nsight trace are for attribution after a gap is measured, and they are not part
of a gate.

The `ns=0` row is the control. It runs no Polar Express product, so the gap
between it and the baseline is the launch and traffic cost of this campaign. A
reader compares the two rows and not a number in prose.

## 15. Retirement

This document is a plan of record, and it retires when its last row lands:

1. The durable sections become [kernels.md](kernels.md) Part 2, with the section
   numbers unchanged.
2. This file is deleted, and the index row in [README.md](README.md) is
   retargeted to Part 2.
