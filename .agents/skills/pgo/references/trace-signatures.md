# Trace signatures, their ceilings, and the idea each one supports

The analyzer turns one steady step into a list of signatures. A signature is a
measured cost, and its ceiling is the share of the step that the cost owns. The
analyst reads this list and writes two ideas. It never computes a share by hand.

`pgo.py analyze` cuts a step at the per-row loss copy, so it needs a stage that
copies one per forward pass. A stage that copies none takes
`scripts/analyze_step.py`, which cuts at the optimizer kernel cluster instead and
prints the same budget per step.

Read `analysis.json` -> `signatures`. It sorts by ceiling, largest first.
Read `analysis.json` -> `step` for the numbers behind each one.

## The table

| Signature | What it measures | The idea it supports | Ceiling |
|---|---|---|---|
| `host_own_code` | host time inside a stall with no driver call in flight | move the work to the device, or out of the step | that time over the step |
| `driver_call_overhead` | host time inside a CUDA driver call | one call for many operations, or fewer operations | that time over the step |
| `blocking_dtoh_scalars` | stalls that end at a scalar readback | keep the scalar on the device | the sum of those stalls |
| `dtoh_bytes` | device time of the bulk device-to-host copies | reduce the buffer on the device | that device time |
| `launches_under_3us` | kernels too short to repay a launch | fuse the chain, or batch the call | their device time |
| `tiny_gemm_tile` | a GEMM whose tile is far below the card | one batched call, or a retile | their device time |
| `dtod_gather` | device-to-device copies of a gathered group | none. This is bandwidth, not a stall | their device time |
| `idle_unexplained` | device waiting that no stall accounts for | none yet: collect `--sample cpu` | that time over the step |

## How to read the two host signatures

`host_own_code` and `driver_call_overhead` are disjoint by construction. A stall
means the device waits; inside it the host sits in a driver call or in its own
code, and never both. The two ceilings may therefore add, and a pair of one
of each addresses the whole host side of the step.

The pair of the reference iteration is exactly that, and the measurement of it
is in `docs/profile-guided-optimization.md` section 8.

A signature may contain another one. `driver_call_overhead` contains the copy
calls that `dtoh_bytes` and `blocking_dtoh_scalars` measure, so the ranking drops
the contained one when it takes the container. The `dropped` list of the ranking
names what left the pair and why.

## How to read a stall

`step.stalls` is the ordered list of idle stretches over 50 microseconds. Each
one names:

- `after`: the device operation that ends before the stall. Its name and its
  size say what the host does next.
- `host_cuda_ms`: the part of the stall inside a driver call.
- `host_own_ms`: the part inside the host's own code.

A stall of 30 ms after a 67 MB device-to-host copy, with `host_own_ms` near the
whole stall, is a host loop over that buffer. The fix is a device reduction. A
stall whose `host_cuda_ms` is nearly the whole stall is a blocking copy that
drained the queue. The fix is to remove the read.

## How to read an idle stretch the signatures miss

`idle_unexplained` is the device time that no stall over 50 microseconds
accounts for. When it is large, the trace cannot see the host cost, because the
CUDA domains record only the driver calls. Collect the next trace with:

```bash
.agents/skills/pgo/scripts/trace.py \
    --stage base --device h100 --sample cpu
```

A CPU sample names the function that holds the host, so the next analyst has a
file and a line instead of a share.

## Ceilings are upper bounds

A ceiling is what the signature costs now, not what an idea will save. Removing a
buffer read also removes its copy, its stall, and its cache traffic, and none of
those is in the new step until the measurement says so. Write the forecast into
`ideas.json` and let `compare.py` record both numbers, the pair ceiling and the
measured gain.

## The reference trace

`profiles/h100-base-20step/REPORT.md` holds one measured iteration on the H100.
It is a local artifact: git ignores everything under `profiles/`.
Its step was 436 ms of which the device worked 81.8 ms, and the analyzer reports:

| Signature | Ceiling |
|---|---|
| `host_own_code` | 0.4306 |
| `driver_call_overhead` | 0.2374 |
| `dtoh_bytes` | 0.0996 |
| `blocking_dtoh_scalars` | 0.0958 |
| `idle_unexplained` | 0.0890 |

The pair ceiling was 0.668, so the loop proceeded. The first idea of that pair
landed in commit `8b71b0c`: `kernels::GradNorm` reduces the fp32 gradient spans
on the device. The 374 MB of host staging per step then left the tree.
