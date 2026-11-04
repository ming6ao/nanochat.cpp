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
