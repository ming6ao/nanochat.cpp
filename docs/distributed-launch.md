# Distributed launch

Status: design. The launch moves from `notebooks/parallel.py` to host tooling.
This document is the authority for the launch.
[distributed-design.md](distributed-design.md) section 10 holds a summary.

## 1. Problem

`notebooks/parallel.py` starts one rank per card. It calls `tools/nanochat
train` for each rank. Two faults follow.

The first fault is the layer. The file is process orchestration.
`DESIGN.md` section 2.2 puts the process entry point outside the model layers.
The notebooks directory holds presentation, not orchestration
([notebook-workflow.md](notebook-workflow.md)).

The second fault is the broker. `tools/nanochat train` takes the GPU broker
lock for one rank. Two ranks take the lock twice. A multi-card host then
blocks, or the run deadlocks. Kaggle hides the fault, because the sandbox and
the broker are off there.

## 2. Goals and non-goals

Goals:

- One process per rank.
- One broker lock for the rank group.
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
owns their defaults. The launch, the broker, and the sandbox are host tooling.
`DESIGN.md` section 2.2 and section 6 put them there.

| Task | Owner | File |
|---|---|---|
| Rank and group values | C++ at L6 | `src/train_main.cc` |
| Gradient sum | C++ | `src/distributed.cc`, `backends/cuda/nccl_sync.cu` |
| Process start | Host tooling | `tools/nanochat` |
| Broker lock | Host tooling | `tools/gpu.sh` |
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

1. Select the device set. The default is `0..N-1`.
2. Acquire one broker lock for the device set.
3. Apply one sandbox profile to the group.
4. Set `CUDA_VISIBLE_DEVICES`, `NANOCHAT_RANK`, `NANOCHAT_LOCAL_RANK`, and
   `NANOCHAT_WORLD_SIZE` for each rank.
5. Start rank 0 first, so the bind wins the port race.
6. Redirect each rank to `<log-dir>/rank<i>.stdio.log`.
7. Wait for every rank. Return the first nonzero status.
8. Print a summary. With `--json`, print one JSON object.

The `dist` command skips the broker in two cases:

- `NANOCHAT_SANDBOX` has a value, so another broker is active.
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

`notebooks/nanochat-cpp-on-2x-t4.ipynb` imports `nanochat_cpp.distributed`.
The notebook builds one plan and calls `launch`. The notebook keeps the loss
plot and the checkpoint hash check. Those are presentation.

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
status, the rank logs, the signal kill, and one broker acquisition.

### Phase 3 — The Python surface

1. Add `python/nanochat_cpp/distributed.py`.
2. Add the `dist` row to `_entry.py` and the method to `toolchain.py`.

Gate: a Python test with a fake `tools/nanochat`.

### Phase 4 — The notebook and the removal

1. Change the notebook to `nanochat_cpp.distributed`.
2. Delete `notebooks/parallel.py`, `notebooks/parallel_test.py`, and the two
   `notebooks/BUILD.bazel` targets.
3. Remove `SMOKE_FLAGS`.

Gate: one short two-rank Kaggle run passes the checkpoint hash check.

## 9. Definition of Done

1. `tools/nanochat dist --nproc-per-node 2 -- <command>` starts two ranks.
2. One broker lock covers the group. A second job on the same device set
   waits.
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
| Kaggle has no broker | Skip the broker when the backend is `none` |

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
