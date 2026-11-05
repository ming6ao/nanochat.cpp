# Campaign records

A campaign is a planned change with rows, phases, or stages. This directory holds
the live reference, so a campaign's execution log does not stay here. This
document keeps the outcome, the dates, and the pointer to the evidence.

The durable content of each retired plan now lives in the reference document
named in its section. The design, the decisions, and the invariants are there.
Only the stage log, the forecast, and the review notes were cut.

The evidence of each campaign is in `.sliceme/`, which is not tracked:
`feat--<campaign>.evidence.md` holds the recorded checks, and
`feat--<campaign>.report.md` holds the summary.

## The campaigns at a glance

| Campaign | Landed | Status | Design now lives in |
|---|---|---|---|
| Dependency layering and host packaging | 2026-10-30 | Implemented, D6 open | [build.md](build.md) Part 1 |
| GPU job queue | 2026-10-18 | Implemented, Q6 optional | [sandbox.md](sandbox.md) Part 1 |
| Native collectives over NVLink and RDMA | 2026-10-18 | Phases 1 to 2, phases 3 to 6 open | [distributed-design.md](distributed-design.md) Part 3 |
| One chain on three hosts | 2026-11-01 | Executed, rows 0 to 11 | [post-training.md](post-training.md) Part 2 |
| The pipeline scale-up | 2026-11-01 | Rows 0 to 9, gates G1 to G3 pass | [post-training.md](post-training.md) Part 3 |
| Host task consolidation | 2026-11-02 | Implemented | [python.md](python.md) Part 1 |
| Supervised fine-tuning reference parity | 2026-11-03 | Phases A to F, phase G open | [parity.md](parity.md) Part 1 |
| The big-shape profile on H100 | 2026-11-03 | One iteration, kept | [performance.md](performance.md) Part 1 |
| The smear becomes a seam family | 2026-11-03 | Landed in `425546f` | [kernels.md](kernels.md) |
| The Muon fusion | 2026-11-05 | Implemented and gated, not promoted | [kernels.md](kernels.md) Part 1 |

## Dependency layering and host packaging

The goal was to keep the C++ core free of third-party packages while notebooks,
Modal training, and the PyTorch parity work still run. The rule is one zone per
dependency, and the build proves each zone boundary.

Stages S0 to S7 landed on 2026-10-30, and all eight phases passed their recorded
checks. `D1` to `D12` are decided in Part 1 section 3 of
[build.md](build.md). `D6` stays open: a per-architecture wheel is not built
yet, so a wheel does not carry the shared library today.

The campaign found four dependency leaks in the tree as it stood. Section 2 of
Part 1 of [build.md](build.md) records them.

Evidence: `.sliceme/feat--dependency-layering-plan.evidence.md`.

## GPU job queue

The goal was to remove the GPU lock. One worker process owns the device, and a
queue holds the jobs. An agent submits a job and reads the exit code and the
logs.

Stages Q1 to Q5 landed on 2026-10-18. `python/nanochat_cpp/gpu_queue.py` holds
the store, the worker, and the client, and `tools/nanochat gpuq` is the client
interface. `tools/gpu.sh` is gone. Stage Q6 stays optional, and the systemd unit
is not shipped: the on-demand detached worker works on WSL2 and on a notebook.

The campaign needed no evidence file, because it predates the campaign loop in
`.sliceme/`. Part 1 of [sandbox.md](sandbox.md) is the design of record.

## Native collectives over NVLink and RDMA

The goal was to remove NCCL as a hard dependency of the CUDA build. A native
all-reduce hides behind the existing gradient-sync seam.

Phases 1 and 2 landed on 2026-10-18 on the local two-card T4 host. The ring
schedule, the host-staged transport, the transfer-window pool, and
`NativeGradientSync` are in `backends/cuda/collective/`. The two T4 cards have no
peer access, so the ring stages through pinned host memory.

Phases 3 to 6 remain. The NVLink and RDMA transports need the H100 node, so no
row of those phases can land on this host. Part 3 of
[distributed-design.md](distributed-design.md) holds the transport interface,
the ring schedule, and the broker contract.

The campaign needed no evidence file, because it predates the campaign loop.

## One chain on three hosts

The goal was one automated chain: base training, then supervised fine-tuning,
then reinforcement learning, with one job file and one artifact contract.

Rows 0 to 11 landed on 2026-11-01. Six defects blocked the chain, `B1` to `B6`,
and all six are fixed. `B7` to `B10` landed on 2026-10-31, and `B11` is out of
scope. Two small questions remain open, and neither blocked a row.

The defect identifiers `B1` to `B11` belong to this campaign. A code comment
that names one of them cites this section.

The defects are worth naming, because each one was invisible until the chain ran
end to end:

- `B1`: the supervised stage could not set the model shape.
- `B2`: no model flag set `value_embedding`.
- `B3`: the reinforcement stage overwrote the checkpoint it warm started from.
- `B4`: every rollout in a group was identical.
- `B5`: a nested `tools/nanochat` started a nested sandbox scope.
- `B6`: the two stage executables existed only in a Bazel tree.

Decisions `D1` to `D11` are in Part 2 section 4 of
[post-training.md](post-training.md). The measured numbers of the demo shape
stay with that part; they are a record, not a forecast.

Evidence: `.sliceme/feat--pipeline-plan.evidence.md`.

## The pipeline scale-up

The goal was a bigger chain: a fourth stage, the `standard` profile, and two
Modal legs.

Rows 0 to 9 landed on 2026-11-01. Row 9 was the first two-leg run on a real
Hopper card, and it ran on 2026-10-31. Gates G1 to G3 pass. The G3 ratio band
stays `[1, 200]`, because twice the measured ratio is wider than the band.

The work rows and the gates `G1` to `G3` belong to this campaign. A reference to
"work row N" means row N of this scale-up.

The run found two device defects, and both are fixed. One was an Ada-only
intrinsic that Hopper rejects with status 7. An oracle review also corrected six
defects in an earlier draft of the plan: the eval batch, the eval count, the
Modal function names, the build flags, the roofline target, and the
hardware-check claim.

Decisions and the shape are in Part 3 of [post-training.md](post-training.md).

Evidence: `.sliceme/feat--pipeline-scaleup-plan.evidence.md`.

## Host task consolidation

The goal was one home per host task. Each host task has one command, the trust
boundary stays in shell, and the orchestration moved to Python.

The campaign is implemented. The inventory was refreshed at commit `bd6fa95`,
which landed the GPU queue. The completed GPU work moved to history, and the
language move was split into four stages.

The locked decisions and the list of commands that must not merge are in Part 1
of [python.md](python.md).

Evidence: none. The campaign changed only `tools/**` and the two Python
modules, and the test loop is its record.

## Supervised fine-tuning reference parity

The goal was to earn the name "reference parity" for the supervised
fine-tuning path, with evidence and a gate behind the claim.

Phases A to F landed by 2026-11-03, with the fixtures, the gates, and the driver
surface that each phase names. Phase G is open: the T2 driver run and the
reference comparison have not happened, so `P6` and `P7` stay `open` in
[parity.md](parity.md).

The four rungs, the pinned settings, and the fixture rules are in Part 1 of
[parity.md](parity.md).

Evidence: `.sliceme/feat--sft-parity-plan.evidence.md`.

## The big-shape profile on H100

The earlier `base`/`h100` campaign used the `h100` shape: 8 layers, hidden 512,
sequence 256. The roofline models that shape at 39.5 percent of the card, and it
measured 2.35 percent MFU, because the step was host bound. No amount of host
work could have taken it near the card.

This campaign took the largest shape in the tree, `big`: 20 layers, hidden 1280,
10 heads over 5 key/value heads, sequence 1024, vocabulary 32768, window `SSL`,
and batch 16. The roofline models it at 53.4 ms and 79.9 percent for 42.2 TFLOP
of arithmetic per step.

One iteration, cap one, decision kept. The pair gained 24.76 percent of the step
time and 32.1 percent of the MFU, and the two points do not overlap.

| Point | Step time | Throughput | MFU |
|---|---:|---:|---:|
| Before | 425.94 ms | 38,459 tokens/s | 9.90 percent |
| After | 320.47 ms | 51,065 tokens/s | 13.08 percent |

The trace named the cause. One steady step was 411.4 ms, with 269.3 ms of device
busy time, 1826 kernels, and 24 device-to-host copies. The host side held 133.3
ms of stalls, and three of them followed a copy: 52.1, 38.0, and 15.5 ms, of
which 48.7, 34.7, and 11.6 ms ran as host code in `SmearForward` and
`SmearBackward`.

The pair split the smear host loops by row, and gave the two head transposes one
16-byte vector per thread in place of three 64-bit divides per element. The
ceiling forecast was 0.34 and the measured gain was 0.2476.

The reference `nanochat` trainer ran this shape at 20.00 percent. The gap fell
from 3.43 times before the earlier rounds, to 1.97 times, to 1.53 times.

## The smear becomes a seam family

The learned bigram residual was the one composition on the H100 path with no
device branch. `ops::SmearForward` and `ops::SmearBackward` staged the whole
42 MB activation to the host twice per step and walked it one element at a time,
for 113.0 ms of the 133.3 ms of stall time in the traced step.

Commit `425546f` moved the family into the seam: `SmearParams` beside
`ValueGateParams` in `include/nanochat/kernels.h`, the row loop in the CPU
backend as the definition, one kernel per direction in
`backends/cuda/kernels/smear.cu`, and a finite-difference gate against a host
reference. The tier row and the `smear` declaration in both `caps.json` files
came with it.

One iteration on one H100 at the `big` shape, batch 16, decision kept:

| Point | Step time | Throughput | MFU |
|---|---:|---:|---:|
| Before | 403.62 ms | 40,579 tokens/s | 10.50 percent |
| After | 231.24 ms | 70,849 tokens/s | 18.11 percent |

That is 42.7 percent of the step time and 72.5 percent of the MFU, against the
compiled reference trainer at 20.00 percent for the same shape.

The scheme is one launch per direction. Every row owns its output row and one
halo row of the input, so no barrier sits inside a row pass and no two blocks
write one row. The only state that crosses rows is the 25 floats of parameter
gradient: each block writes its own partial, and the last block to arrive
combines every partial in ascending block order after a `__threadfence`. No
atomic touches the data, so two runs of one job give identical bits. That
combine is a phase boundary, which is what a training megakernel needs.

The reduction order follows the reference. The gate gradient accumulates in
double in ascending column order on one lane, because the parallelism is across
rows and the dot costs microseconds. The lambda gradient reduces over rows in
ascending order in double, and the gate weight gradient accumulates in float in
row order.

The device agrees with the committed reference at rounding level: forward
3.73e-09, backward 5.96e-08, optimizer step 1 2.38e-07, trajectory loss
2.38e-07, trajectory parameter 1.67e-05. The CPU path is bit identical to the
tree it replaces, and the failing set of `//tests:numerics_trace_test` is byte
identical before and after.

## The Muon fusion

The goal was to run a whole `MuonUpdate` stage as one device-side launch: the
three Polar Express products in one batched GEMM of our own, both elementwise
passes folded into the epilogues of their neighbours, a grid barrier between the
stages, and a phase driver as the simulator path and the fallback.

The campaign measured before it wrote, and the measurement said no. Phase 0
measured the launch and elementwise share of the step at both `muon_benchmark`
profiles: 9 to 13 percent at `h100` and 7.2 to 7.5 percent at `big`, above the
2 percent floor, so the GEMM was written. The same phase measured the sync term,
which took the launch saving out of the bet: one fused iteration replaces
41.9 microseconds of launch gaps with 4.7 microseconds of one launch and three
barriers, 0.5 percent of an `h100` iteration, so the share is elementwise
traffic and not dispatch. Phase 1 then measured the hand GEMM at 0.68 to 0.72 of
the cuBLAS rate on the targeted 640x1280 group against a gate of 0.85, and it
failed.

The fused path landed anyway, because the gates required it: the phase device
function, the sixteen phases and their three epilogue folds, the workspace
reduced from `2·total + 3·min_total` to `2·total + 2·min_total`, both drivers,
and the simulator legs all match `testing/optim_ref.h`, the shipped multi-launch
`MuonUpdate`, and each other. The frozen seam keeps its one `MuonUpdate` entry,
and no capability bit and no `caps.json` field was added.

What did **not** land is the promotion. `muon_fused:` beats `muon:` for no group
at either profile: 6 to 10 percent slower at `ns=0`, 27 percent slower at the
`h100` step, 45 percent at the `big` step, and 76 percent on the `big` 1280x5120
group. `kMuonFusedMaxElements` is therefore 0, `MuonFusedSelected` keeps every
group on the multi-launch cuBLAS path, and the fused path stays in the tree as
the shape rule and the phase-bisection tool. The measured rows are in
[baselines.json](baselines.json) under `muon`, the design is
[kernels.md](kernels.md) Part 1, and the numbers under the protocol are
[performance.md](performance.md) section 2.1.

The defects are worth naming. The identifiers `M1` to `M3` belong to this
campaign.

- `M1`: **the design document the plan cites was never committed to `docs/`.**
  Roughly thirty-five comments across `backends/cuda`, `backends/cuda/kernels`,
  `dev/kernels`, `tests`, and `tools/cuda_sim` cite it by path, so those
  citations were dangling from the day they landed. Its durable sections are now
  [kernels.md](kernels.md) Part 1 with their section numbers unchanged. The
  citations still name the old path and have to be retargeted by the directories
  that own them (open item 1).
- `M2`: **one block size fits no stage.** 256 threads, `__launch_bounds__(256, 3)`
  and 15 KiB of static shared memory hold 6 blocks per SM where the elementwise
  families' own kernels hold 8, so the `ns=0` step, which runs no GEMM at all
  and has nothing to fold, is 6 to 10 percent slower. The folds cannot pay that
  back.
- `M3`: **a gate against the cuBLAS rate is not a selector.** On 1280x5120x1280
  the hand `A` stage measured 0.95 to 1.17 of cuBLAS, not because the hand
  kernel is fast but because cuBLAS measures 4181 to 5608 GFLOP/s on that
  bandwidth-bound product. Only the end-to-end `muon_fused:`-versus-`muon:` row
  decides, and that row is what the shape rule uses.

Open items:

1. Retarget the dangling citations of `M1` in `backends/cuda`,
   `backends/cuda/kernels`, `dev/kernels`, `tests`, and `tools/cuda_sim` to
   [kernels.md](kernels.md) Part 1 section K. This repository's own documents are
   already retargeted; the code comments are not, and each one belongs to the
   directory that owns it.
2. Promotion is one measurement and one constant: raise
   `kMuonFusedMaxElements` (in `backends/cuda/kernels/muon_fused.cu`) above a
   group's element count once a group wins, and nothing else changes.
3. A different GEMM schedule is a different prototype and a different
   measurement: an 8 by 8 micro-tile, a 16- to 32-deep k slice, or a 256 by 64
   tile. So is a second block size, which the cooperative driver's one fixed
   geometry forbids by construction.
4. The bf16 build stays on the multi-launch path, because the fused schedule is
   the fp32 one.
5. The cooperative driver keeps no emulator leg: the emulator runs blocks one at
   a time, so a grid barrier would deadlock there and only the phase driver is
   compared against the CPU reference. Its device-day coverage is the T1 test
   and the forced `cudaErrorCooperativeLaunchTooLarge` fallback.

[backlog.md](backlog.md) never carried a Muon line, so this campaign closes
none. It began as a plan of its own rather than as a backlog part, which is why
its design was absorbed here instead of being moved out of the backlog.

Evidence: the campaign's evidence file under `.sliceme/`, which is not tracked.
The measured rows live in [baselines.json](baselines.json) and under "The fused
path on this card" in `backends/cuda/kernels/README.md`.

## How a plan retires

1. The last row lands, and the evidence is written to `.sliceme/`.
2. The durable sections move into the reference document: the design, the
   decisions, the invariants, and the commands that still apply.
3. The outcome, the defect list, and the open items are added here.
4. The plan file is deleted, and every reference to it is retargeted to the
   reference document or to this record.

A section number is kept when a document is absorbed, so an existing reference
still resolves. This is why the absorbed parts above are cited as
"Part N, section K".
