# Host script consolidation plan

Status: implemented. Owner: Architect.

This plan changed the host tooling under `tools/**`, two Python modules, and one
Python test area. It gives each host task one home. It keeps the trust boundary
in shell. It moves the orchestration to Python. It deletes the scripts that no
longer earn a place.

Land one stage per commit. Run the full test loop after each stage. Each stage
must pass on its own, so a later stage can wait.

This revision corrects the earlier draft in three ways. It refreshes the
inventory against commit `bd6fa95`, which landed the GPU queue. It cuts the
completed GPU work to history (section 5). It also splits the language move into
four stages. See section 12 for what landed.

## 1. Goal

Three results are the target.

1. One command table, not two.
2. One fixture generator, not four.
3. One worker process that owns the GPU, not a separate lock script
   ([gpu-queue-plan.md](gpu-queue-plan.md)). This result is complete; see
   section 5.

## 2. Current inventory

The counts are line counts at commit `bd6fa95`. Refresh them when a stage lands.

| File | Lines | Role |
|---|---|---|
| `tools/nanochat` | 855 | Command dispatch, argument assembly, Bazel wiring |
| `tools/sandbox.sh` | 170 | Resource launcher through `systemd-run` |
| `tools/sandbox/verify.sh` | 63 | Limit read-back inside the scope |
| `python/nanochat_cpp/gpu_queue.py` | 947 | The GPU job queue: the store, the worker, and the client |
| `tools/lint.sh` | 71 | Style gate |
| `tools/prune_bazel_output_bases` | 210 | Output base reaper |
| `tools/prune_test.sh` | 115 | Hermetic reaper test |
| `tools/cuda_sim/check_symbols.sh` | 136 | Symbol gate for the API mock |
| `tools/kaggle/bootstrap.sh` | 163 | Kaggle host setup |
| `tools/kaggle/github-setup.sh` | 62 | Kaggle git setup |
| `tools/kaggle/scripts_test.sh` | 37 | Syntax test for the Kaggle scripts |
| `tools/dump_data_fixtures.py` | 781 | Fixture generator: tokenizer, loader, chat |
| `tools/dump_model_fixtures.py` | 939 | Fixture generator: generate, eval, plan |
| `tools/dump_runtime_fixtures.py` | 666 | Fixture generator: api, numerics |
| `tools/dump_training_fixtures.py` | 1067 | Fixture generator: oracle, train, rl |
| `tools/fixture_format.py` | 104 | Record writer |
| `tools/gen_unicode_tables.py` | 267 | Unicode table generator |
| `tools/check_tokenizer_parity.py` | 129 | Tokenizer parity runner |
| `tools/convert_checkpoint.py` | 59 | Forwarder to the Python package |
| `python/nanochat_cpp/_entry.py` | 64 | Python copy of the command table |
| `python/nanochat_cpp/toolchain.py` | 457 | Python copy of the command arguments |

## 3. Duplication in the current tree

- The four fixture generators repeat one preamble and one module-level
  dispatch. The four files hold eleven subcommands in total.
- The helpers are identical. `build_config`, `_as_f32`, and `_scalar` appear in
  `tools/dump_model_fixtures.py:92-136` and
  `tools/dump_training_fixtures.py:124-145`. The helper `_as_i32` appears twice
  in `tools/dump_training_fixtures.py`.
- The command list lives in `tools/nanochat` and in `_entry.py`.
- The argument list for each command lives in `tools/nanochat` and in
  `toolchain.py`.
- The lock discipline is gone. `tools/gpu.sh` does not exist, and
  `python/nanochat_cpp/gpu_queue.py` holds the worker. See section 5.

`tools/sandbox.sh` and `tools/sandbox/verify.sh` are not duplication. They are
two halves of one mechanism, and they stay separate. `tools/sandbox.sh` starts
the scope. `tools/sandbox/verify.sh` runs inside that scope, after `taskset` and
`prlimit` (`tools/sandbox.sh:166-170`). It reads the limits back from the
process cgroup. It cannot be a function of `tools/sandbox.sh`, because the
scope starts a new program. See constraint 7.

Four documents name `tools/dump_sft_fixture.py` or
`tools/dump_sft_loop_fixture.py`. One document names
`tools/nanochat_cpp_selftest.sh`. None of the three files exists. Two of the
names record planned work, and they are not stale paths. `docs/sft-parity-plan.md`
phases B and E assign the two SFT fixtures, and `docs/post-training.md` marks
the first one as remaining. Stage S1 repoints them at the merged generator. The
`tools/nanochat_cpp_selftest.sh` row is part of the historical record in
`docs/python.md` section 8, and it stays.

## 4. Locked decisions

Lock these decisions before any code moves.

1. **Python owns the orchestration.** Python holds the command table, the
   dispatch, the argument assembly, the lock, the reaper, the fixture
   generators, `doctor`, and `simulate`.
2. **Shell owns the resource launcher.** `tools/sandbox.sh` keeps the
   `systemd-run` call, the CPU affinity, the rlimits, and the limit read-back.
3. **`tools/sandbox.sh` stays the `--run_under` program for a CPU test.** Bazel
   invokes it once per test action. A GPU test action uses a queue submission in
   `--run_under` instead, so the worker sandboxes the action and owns the device
   ([gpu-queue-plan.md](gpu-queue-plan.md)). Do not put a GPU lock in
   `tools/sandbox.sh`.
4. **Freeze the observable contract.** Keep the environment variable names, the
   two log paths and their line formats, the exit codes, the two `--run_under`
   strings, the `NANOCHAT_SANDBOX` sentinel, the `doctor --json` keys, the
   fixture subcommand names, and the fixture binary layouts.
5. **Keep `profiles.conf` as the only profile source.** Do not copy the numbers
   into Python. `tools/sandbox.sh` validates the profile name
   (`tools/sandbox.sh:65`).
6. **Land one stage per commit.** Do not mix a code move with a rewrite.
7. **Keep `tools/sandbox/verify.sh` as a separate program.** It runs inside the
   scope. Do not merge it into `tools/sandbox.sh`.
8. **The merged generator is `tools/dump_fixtures.py`.** It keeps every existing
   subcommand name: `oracle`, `train`, `rl`, `generate`, `eval`, `plan`,
   `tokenizer`, `loader`, `chat`, `api`, and `numerics`. A later fixture is one
   new row.
9. **Freeze the sandbox interface.**
   `tools/sandbox.sh [--profile NAME] [--backend auto|systemd|none]
   [--no-slice] [--] <cmd...>`. Both profile forms stay accepted, because
   `--run_under` needs `--profile=NAME` and `run_profile` uses `--profile NAME`
   (`tools/sandbox.sh:27-33`). No stage changes this interface.
10. **The Python layer keeps one rule at a time.** While the dispatch is in
    shell, the Python layer reaches the host only through
    `tools/nanochat <subcommand>`. After stage S13, the Python command-line tool
    calls Bazel, `tools/sandbox.sh`, and the queue directly. The in-process
    compute API still does not. Stage S13 updates `docs/python.md` sections 6
    and 12.

## 5. The GPU queue (history)

This work is complete. Commit `bd6fa95` landed it, and
[gpu-queue-plan.md](gpu-queue-plan.md) records the status as implemented, with
stages Q1 to Q5 in the tree. No stage in this plan depends on this section. The
section stays as the record of the decision that superseded the earlier option
to move the lock into the host command process.

- The GPU lock is gone. `tools/gpu.sh` does not exist.
  `python/nanochat_cpp/gpu_queue.py` holds the store, the worker, and the
  client.
- The worker is the only runner, so it serializes the GPU jobs by construction.
  The worker owns the pool and assigns the devices.
- The `nvidia-smi` gate stays as a pre-flight check. It is a check, not a lock.
- `NANOCHAT_DIST_BROKERED` and `NANOCHAT_CLIENT_BROKERED` are gone. The one
  reentrancy guard is `NANOCHAT_GPU_QUEUED`, and the worker sets it for every
  job (`python/nanochat_cpp/gpu_queue.py:425`).
- The build lock stays, and it keeps the required pattern:

```bash
with_build_lock() { exec 9>"$build_lock"; flock 9; "$@" 9>&-; }
```

## 6. Constraints: do not merge these

- Do not put the GPU queue in `tools/sandbox.sh`. Bazel invokes that script once
  per test action. The queue belongs to the worker.
- Do not drop the `nvidia-smi` gate while another job uses the card.
- Do not convert `tools/sandbox.sh` to Python. `systemd-run`, `taskset`,
  `prlimit`, and the `--run_under` string are shell work.
- Do not merge `tools/sandbox/verify.sh` into `tools/sandbox.sh`. A merge would
  change the `/tmp/nanochat-sandbox.log` line that decision 4 freezes.
- Do not convert the Kaggle scripts. They install packages and edit `PATH`.
- Do not change a fixture subcommand name or a fixture binary format.
- Keep the `Makefile`, which is the non-Bazel fallback.

## 7. Preparatory changes

### P1. Name the merged generator, and repoint the planned references

Decision 8 names the merged generator. Repoint the planned work in
`docs/sft-parity-plan.md:151`, `docs/sft-parity-plan.md:154`,
`docs/post-training.md:376`, `docs/parity.md:377`, and
`docs/fp8-training.md:637` at the merged path. Keep the fixture names and the
phase assignments.

Leave the historical records alone. `docs/python.md` section 8 and section 10
record the tree at the time, and they keep the old paths.

This step is documentation only. It lands first, so that no plan points at a
file that a later stage deletes.

### P2. Add a contract test suite

Pin the current behavior before any code moves. Use one hermetic `py_test` under
`//python`. The test needs no GPU and no Bazel run.

The test needs the real script. Add `tools/nanochat`, `tools/sandbox.sh`,
`tools/sandbox/profiles.conf`, and the four fixture generators to the `data`
list of the target. The current `python/tests/toolchain_test.py` writes a fake
entry script, so it cannot pin the shell.

Test these items:

- The command list from `tools/nanochat help` equals `_entry.names()`.
- The key set of `tools/nanochat doctor --json`.
- The `--run_under` string, and the exported variables, for the CPU test path
  and the GPU test path.
- The line format of `/tmp/nanochat-sandbox.log` and `/tmp/nanochat-gpu.log`.
- The sandbox interface in decision 9. Both profile forms must select the same
  profile.

Two constraints on the test:

- Use the read-only paths only: `help`, `--dry-run` (P3), and `doctor --json`.
- Point the queue state at a test directory. `open_db` creates the state
  directory and the database (`python/nanochat_cpp/gpu_queue.py:134-146`). Set
  `NANOCHAT_GPU_QUEUE_DIR` and `NANOCHAT_GPU_LOG` to paths under the test
  temporary directory.

The command list has one known difference. The shell has a
`help | -h | --help` branch (`tools/nanochat:145`), and `_entry.names()` has no
`help` row. Pin the difference now, and add the `help` row in stage S12.

### P3. Add `--dry-run`

Add one flag to `tools/nanochat`. It covers `build`, `test`, `check`, `run`, and
`simulate`.

The flag prints one JSON object per line, to stdout, in execution order, and
then exits 0. Each object holds two keys:

- `layer`: one of `argv`, `bazel`, `sandbox`, `gpuq`, or `host`.
- `argv`: the command and its arguments, as an array of strings.

The first line is the tool's own argument list, with the layer `argv`. Each
later line is one external command. The layers are:

- `bazel` — one Bazel invocation.
- `sandbox` — one `tools/sandbox.sh` invocation.
- `gpuq` — one queue submission.
- `host` — one other host command.

The flag acquires no lock, submits no job, and runs no command. It resolves the
profile names and the host-dependent flags.

Example, for `tools/nanochat build //src:train_main`:

```json
{"layer": "argv", "argv": ["build", "//src:train_main"]}
{"layer": "bazel", "argv": ["bazel", "build", "//src:train_main"]}
```

Example, for `tools/nanochat test --gpu //bindings:all`:

```json
{"layer": "argv", "argv": ["test", "--gpu", "//bindings:all"]}
{"layer": "bazel", "argv": ["bazel", "test", "//bindings:all", "--test_tag_filters=gpu", "--run_under=tools/nanochat gpuq submit --profile=t1-gpu --", "--spawn_strategy=local", "--config=cuda", "--local_test_jobs=1"]}
```

The `--run_under` field carries the queue command, so a GPU test action shows
the queue layer inside the Bazel line. The `--config=native` field appears only
on a host without NCCL, so a dry run reports the flags of its own host.

The `argv` line is the same list that `toolchain.py` builds for the same
command. The contract suite from P2 compares the two lists. Without that line
the two lists are not comparable: `toolchain.py` builds the shell argument
list, and the later lines show the Bazel argument lists.

### P4. Use extract, reroute, delete

Do not move and rewrite in one step. Use three reversible steps:

1. Extract the shared code into `tools/fixture_common.py`: the `sys.argv[1]`
   dispatch preamble, `build_config`, `_as_f32`, `_as_i32`, and `_scalar`.
2. Reroute `tools/dump_fixtures.py` to the shared helpers. Keep the four old
   scripts as thin forwarders.
3. Delete the four forwarders, their `exports_files` rows, and their four
   `filegroup` rules in `tools/BUILD.bazel`. No other target consumes those four
   filegroups.

The same pattern applies to `tools/prune_bazel_output_bases` and to
`tools/cuda_sim/check_symbols.sh`.

### P5. Keep the sandbox seam

The seam is the interface in decision 9. It is already in place. Add one case to
the contract suite (P2) for each accepted form.

After this step the orchestrator can change language and the sandbox does not
change. The reverse rule depends on the stage; see decision 10.

### P6. Pin the Python version

`tools/nanochat` already runs `python3` for `doctor --json`, for `gpuq`, and for
one inline block (`tools/nanochat:271`, `:585`, `:601`). The host runs Python
3.12.3. `MODULE.bazel:78-83` registers the local `python3` for the Bazel targets
and fails the build when it is absent.

Add one version check at the start of `tools/nanochat`. The check stops with the
corrective message when the interpreter is older than the minimum. Set the
minimum to 3.10, and verify the value in the same commit with
`python3 -m compileall` on `python/nanochat_cpp`. Record the confirmed minimum
in `docs/build.md`, next to the `local_runtime_repo` paragraph.

## 8. Stage order

| Stage | Change | Owner | Language | Risk | Depends on |
|---|---|---|---|---|---|
| S1 | Repoint the plans at the merged generator (P1) | Architect | Docs | None | — |
| S2 | Pin the Python version (P6) | Architect | Bash | None | — |
| S3 | Add the contract test suite (P2, P5) | Architect | Python test | None | S2 |
| S4 | Add `--dry-run`, and compare it in the contract suite (P3) | Architect | Bash and Python test | Low | S3 |
| S5 | Extract `tools/fixture_common.py` and reroute the four generators (P4 steps 1-2) | Oracle | Python | Low | S3 |
| S6 | Merge into `tools/dump_fixtures.py`, and delete the four forwarders and their file groups (P4 step 3) | Oracle | Python | Low | S5 |
| S7 | Rewrite the generator references across the tree | Oracle and Architect | Comments and docs | None | S6 |
| S8 | Move the reaper to Python, with a `py_test` in the same commit | Architect | Python | Low | S3 |
| S9 | Move the dispatch to Python: `build`, `lint`, `roofline` | Architect | Both | Medium | S4 |
| S10 | Move the dispatch to Python: `test`, `check`, `run`, and the aliases | Architect | Both | Medium | S9 |
| S11 | Move the dispatch to Python: `gpuq`, `gpu`, `dist`, `client` | Architect | Both | Medium | S10 |
| S12 | Move the dispatch to Python: `doctor`, `prune`, `simulate`, `shutdown`; add the `help` row | Architect | Both | Medium | S11 |
| S13 | Remove the argument copy in `toolchain.py`, and update `docs/python.md` sections 6 and 12 | Python surface | Python | Medium | S12 |

Stage S7 touches files under four owners. The Architect changes `docs/**` and
`tools/BUILD.bazel`. The Oracle changes `tests/**`. The Harness owner changes
`src/rl_step_main.cc`. The data pipeline owner changes
`src/tokenizer/bpe_trainer_test.cc`. Each owner lands one commit, and the
integrator treats the four commits as one merge group.

Stages S1 to S8 give most of the cleanup. Stage S9 to S13 is the strategic move,
and it is much safer after S3 and S4.

## 9. Acceptance criteria

Run these after each stage:

1. `tools/nanochat lint`
2. `tools/nanochat test`
3. The contract suite from stage S3
4. `tools/nanochat simulate --device h100 --suite api`, after stage S6 or S8
5. One T1 GPU smoke test, after stage S13

The default `tools/nanochat test` already runs `//tools:prune_test` and
`//tools:kaggle_scripts_test`. Its target set is `//...` with the `-gpu` tag
filter, and `tools/BUILD.bazel` places both targets in the `all` suite. Do not
run them a second time.

A stage is not done until the default loop passes, the contract suite passes,
and no stale path remains in the documents.

## 10. Risks and controls

| Risk | Control |
|---|---|
| A language move changes the output by accident | The stage S3 contract suite, and decision 4 |
| The contract test cannot reach the real script | Declare the script, the sandbox, `profiles.conf`, and the generators as test data in S3 |
| The contract test starts a worker, or writes to the real queue | Read-only paths only, and `NANOCHAT_GPU_QUEUE_DIR` points at a test directory |
| The dry-run format changes under the comparison | The format is pinned in P3, and only the contract suite reads it |
| The sandbox takes the queue per test action | Decision 3 |
| The detached Bazel server inherits the lock descriptor | Keep `with_build_lock` and the `9>&-` close |
| A large move cannot be rolled back | Four dispatch stages, one command group each; only S13 deletes code |
| The reaper move loses a case | Port `tools/prune_test.sh` to the `py_test` in the same commit, and keep every case |
| The profile numbers drift between two files | Decision 5 |
| The host lacks the needed Python | The stage S2 version check |
| The generator merge loses a subcommand or a docstring | S6 keeps all eleven names, the contract suite counts them, and each docstring stays on its command function |
| The generator merge leaves a stale path | S7 sweeps the tree, and the acceptance criteria reject a stale path |
| The tests fake the entry script and break on the move | The seam stays in S4, and S13 rewrites `toolchain_test.py` |
| A heavy import runs at startup | Keep the `torch` and `numpy` imports inside the command function |

## 11. Non-goals

- A single language for every host script. The shell trust boundary stays.
- A change to the C++ guard, the kernel seam, or any frozen header.
- A change to a fixture binary format, or to a fixture subcommand name.
- A new third-party dependency.
- A merge of `tools/sandbox/verify.sh` into `tools/sandbox.sh`.

## 12. What landed

The whole change is in one working tree. The stages landed in order, and the
project forbids an interactive commit, so they are not separate commits.

| Stage | Artifact |
|---|---|
| S1 | `docs/sft-parity-plan.md`, `docs/post-training.md`, `docs/parity.md`, and `docs/fp8-training.md` name `tools/dump_fixtures.py` and its kinds. |
| S2 | `tools/nanochat` requires Python 3.10 or later. `docs/build.md` records the floor. |
| S3 | `python/tests/contract_test.py` and `//python:contract_test`. |
| S4 | `--dry-run` in `python/nanochat_cpp/cli.py`, and the comparison in the contract suite. |
| S5 | `tools/fixture_common.py`. |
| S6 | `tools/dump_fixtures.py`, from the four generators. The four scripts and their file groups are gone. |
| S7 | The generator references across `tests/`, `docs/`, and `src/` name the merged path. |
| S8 | `python/nanochat_cpp/prune.py` and `//python:prune_test`. `tools/prune_bazel_output_bases`, `tools/prune_test.sh`, and `//tools:prune_test` are gone. |
| S9 to S12 | `python/nanochat_cpp/cli.py` holds the dispatch. `tools/nanochat` is a shim. `_entry.py` holds the command list, the help lines, and the `help` row. |
| S13 | `toolchain.py` drops the text doctor fallback, and `docs/python.md` sections 6 and 12 describe the two Python layers. |

Three small deviations from the stage text are worth a record.

1. `--dry-run` is a top-level flag. It works in any position before the
   `--` separator, and the plan reports the tool's own argument list first.
2. The contract suite also checks that the CLI handler table equals
   `_entry.names()`, and that the merged generator keeps all eleven kinds.
3. `test --config` with no value now reports `nanochat: --config needs a name`
   and exits 2. The shell reported a `2:` expansion error and exited 1.
