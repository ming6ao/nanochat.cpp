# Distributed launch

Status: delivered. `tools/nanochat dist` starts the rank group, and
`python/nanochat_cpp/distributed.py` wraps it. The two-card notebook now uses
the single-client multi-device runtime instead
(`docs/client-runtime-design.md`), so this launch serves the process-isolation
case. The superseded per-rank helper `notebooks/parallel.py` is gone (phase 4).
This document is the authority for the launch.
[distributed-design.md](distributed-design.md) section 10 holds a summary.

## 1. Problem

`notebooks/parallel.py` starts one rank per card. It calls `tools/nanochat
train` for each rank. Two faults follow.

The first fault is the layer. The file is process orchestration.
`DESIGN.md` section 2.2 puts the process entry point outside the model layers.
The notebooks directory ([../notebooks](../notebooks)) holds
presentation, not orchestration.

The second fault is the GPU queue. `tools/nanochat train` submits one job for
one rank. Two ranks submit twice, and the queue runs them one after the other.
A multi-card host then serializes the ranks, or the run blocks. Kaggle hides
the fault, because the sandbox and the queue are off there.

## 2. Goals and non-goals

Goals:

- One process per rank.
- One queue job for the rank group.
- One sandbox scope for the group.
- A rank contract in the environment, as `torchrun` uses.
- One shell entry point, `tools/nanochat dist`.
- No change to the gradient seam or to the model.

Non-goals:

- More than one node.
- Tensor parallel and pipeline parallel.
- Elastic rendezvous.
- Distributed evaluation.
- A launcher in C++.

## 3. Position in the architecture

`train_main` owns one rank. The rank values configure training, so C++
owns their defaults. The launch, the queue, and the sandbox are host tooling.
`DESIGN.md` section 2.2 and section 6 put them there.

| Task | Owner | File |
|---|---|---|
| Rank and group values | C++ at L6 | `src/train_main.cc` |
| Gradient sum | C++ | `src/distributed.cc`, `backends/cuda/nccl_sync.cu` |
| Process start | Host tooling | `tools/nanochat` |
| GPU queue job | Host tooling | `python/nanochat_cpp/gpu_queue.py` |
| Sandbox scope | Host tooling | `tools/sandbox.sh` |
| Notebook helper | Python | `python/nanochat_cpp/distributed.py` |

A launcher in C++ would move host tooling into L6. It would also run before
the sandbox, so it could not apply the sandbox itself. The project does not
add it.

## 4. The environment contract

`train_main` reads five variables. The launcher sets them. A flag overrides
the variable.

| Variable | Meaning | Flag |
|---|---|---|
| `NANOCHAT_WORLD_SIZE` | The group size | `--world-size` |
| `NANOCHAT_RANK` | The global rank | `--rank` |
| `NANOCHAT_LOCAL_RANK` | The rank on this node | none |
| `NANOCHAT_MASTER_ADDR` | The rank 0 host | `--master` |
| `NANOCHAT_MASTER_PORT` | The rank 0 port | `--port` |

The resolution order is the flag, then the `NANOCHAT_*` variable, then the
default. The prefix `NANOCHAT_` prevents a clash with another tool. The names
follow `torchrun`, so `train_main` can also run under `torchrun`, `mpirun`, or
a shell loop.

On one node, `NANOCHAT_LOCAL_RANK` equals `NANOCHAT_RANK`. The variable exists
for a later multi-node change.

The C++ side adds two small helpers to `src/cli.h`: `EnvInt` and `EnvString`.
The helpers read the variable and parse the value.

## 5. The launcher

`tools/nanochat dist` starts the group. The command line is:

```text
tools/nanochat dist [--nproc-per-node N] [--device LIST] [--profile P]
                    [--master ADDR] [--port PORT] [--log-dir DIR] [--json]
                    -- <command...>
```

The `dist` command does these steps:

1. Select the device set. The default is `0..N-1`, or the first `N` entries
   of the worker's pool when `NANOCHAT_GPU_DEVICES` is set.
2. Submit one queue job for the whole group, and wait for it.
3. Apply one sandbox profile to the group.
4. Set `CUDA_VISIBLE_DEVICES`, `NANOCHAT_RANK`, `NANOCHAT_LOCAL_RANK`, and
   `NANOCHAT_WORLD_SIZE` for each rank.
5. Start rank 0 first, so the bind wins the port race.
6. Redirect each rank to `<log-dir>/rank<i>.stdio.log`.
7. Wait for every rank. Return the first nonzero status.
8. Print a summary. With `--json`, print one JSON object.

The `dist` command skips the queue in two cases:

- `NANOCHAT_GPU_QUEUED` has a value, so a queue job already runs it.
- The sandbox backend is `none`, as on Kaggle.

The `dist` command must kill every rank on a signal. It must also wait for
every child, so no orphan rank holds a card.

The `dist` profile in `tools/sandbox/profiles.conf` covers the group. The
`train` profile covers one rank, so it is too small.

The summary gives the per-rank return code, the log path, the stdio path, and
the seconds. The format follows `tools/nanochat doctor --json`.

## 6. The Python surface

`python/nanochat_cpp/distributed.py` is an optional wrapper for a notebook.
It does not start a process. It calls `tools/nanochat dist --json` and parses
the summary.

```python
@dataclasses.dataclass(frozen=True)
class DistributedPlan:
    argv: tuple[str, ...]
    nproc_per_node: int = 2
    device: str | None = None
    master: str = "127.0.0.1"
    port: int = 29500
    profile: str = "dist"
    log_dir: str | None = None

@dataclasses.dataclass(frozen=True)
class DistributedResult:
    ranks: tuple[RankResult, ...]

def launch(plan: DistributedPlan) -> DistributedResult: ...
def parse_log(path) -> list[dict]: ...
def loss_rows(result) -> list[dict]: ...
```

The command table `_entry.py` gains one `dist` row. The `Toolchain` class
gains one `dist` method. `parse_log` and `loss_rows` move from
`notebooks/parallel.py`.

`Logger::Log` in `src/train.cc` writes the log line. The format is a
contract. One fixture test pins it.

## 7. The notebook

The two-card notebook no longer uses this launch. It uses the single-client
multi-device runtime: one process, one `MeshRuntime`, and the in-process
collective (`docs/client-runtime-design.md`). The mesh removes the rank from the
user program, so the notebook opens no port and sets no rank variable. The
launch stays for the process-isolation case: some drivers forbid two CUDA
contexts in one process, and one bad device then stops one worker only.

## 8. Plan

### Phase 0 — Design and ownership

1. Register this document in `docs/README.md`.
2. Change [distributed-design.md](distributed-design.md) section 10 to a
   summary.
3. Assign the work: architect, harness, Python surface, and notebooks
   (`AGENTS.md` section 1).

Gate: the architect approves the variable names and the `dist` command line.

### Phase 1 — The environment contract

1. Add `EnvInt` and `EnvString` to `src/cli.h`.
2. Set the five defaults in `src/train_main.cc`.

Gate: a CPU test covers the flag-over-variable order.

### Phase 2 — The launcher

1. Add the `dist` case to `tools/nanochat`.
2. Add the `dist` profile to `tools/sandbox/profiles.conf`.
3. Add the signal handler and the wait.
4. Add the `--json` summary.

Gate: a CPU test with a fake command. The test checks the first nonzero
status, the rank logs, the signal kill, and one queue job for the group.

### Phase 3 — The Python surface

1. Add `python/nanochat_cpp/distributed.py`.
2. Add the `dist` row to `_entry.py` and the method to `toolchain.py`.

Gate: a Python test with a fake `tools/nanochat`.

### Phase 4 — The notebook and the removal

1. Change the notebook to `nanochat_cpp.distributed`.
2. Delete `notebooks/parallel.py`, `notebooks/parallel_test.py`, and the two
   `notebooks/BUILD.bazel` targets.
3. Remove `SMOKE_FLAGS` (it lived only in `notebooks/parallel.py`).

Gate: one short two-rank Kaggle run passes the checkpoint hash check.

## 9. Definition of Done

1. `tools/nanochat dist --nproc-per-node 2 -- <command>` starts two ranks.
2. One queue job covers the group. A second job on the same device set waits.
3. The `dist` command exits with the first nonzero rank status.
4. A signal kills every rank.
5. `--json` prints one object with the per-rank fields.
6. A flag overrides its `NANOCHAT_*` variable.
7. `notebooks/parallel.py` and `notebooks/parallel_test.py` are gone.
8. `tools/nanochat lint` passes. The CPU tests pass under `t0-cpu`.

## 10. Risks and controls

| Risk | Control |
|---|---|
| A generic variable name clashes with another tool | Use the `NANOCHAT_` prefix. A flag wins |
| The completed NCCL plan gains a new phase | This document holds the launch plan. The NCCL plan stays closed |
| The group exceeds the `train` profile | Add a `dist` profile sized for the group |
| A signal leaves an orphan rank | Trap the signal and kill the group. Test it |
| The log format drifts from the parser | One fixture test pins the format |
| The move crosses a workstream boundary | Phase 0 assigns the owners first |
| Kaggle has no queue | Skip the queue when the backend is `none` |

## 11. Ownership

| Phase | Owner | Files |
|---|---|---|
| 0 | Architect | `docs/**` |
| 1 | Harness | `src/train_main.cc`, `src/cli.h` |
| 2 | Architect | `tools/nanochat`, `tools/sandbox/profiles.conf` |
| 3 | Python surface | `python/nanochat_cpp/**` |
| 4 | Notebooks | `notebooks/**` |

## 12. Open questions

1. Does the `dist` profile replace the `train` profile, or sit beside it?
2. Should `dist` default the log directory, or require it?
3. Should `--device 0` allow two ranks on one card, for a smoke test?
