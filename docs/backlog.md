# Backlog: work that is designed and not implemented

This document holds the proposals that the tree does not implement. Each part
states its goal, its measured problem, its design, and its gates.

When the first phase of a part lands, move the part into a reference document
and record the outcome in [campaigns.md](campaigns.md). No code cites a part of
this document today, so nothing here is a promise about the current tree.

Read [performance.md](performance.md) for the measurement protocol,
[kernels.md](kernels.md) for the kernel seam, and [model.md](model.md) for the
workspace.

---

## Part 1. Training memory and kernel bandwidth

Three proposals, all unmeasured. The code evidence is in the tree, and the
memory figures come from [eval.md](eval.md). The savings are predictions.

### 1. Recompute of the row activations

**The problem.** The training arena is about 36 GB at `d12_s1024`, batch 32,
fp32, and the card holds 11 GB. The layer activations are about 24 GB
(`L * 20 * R * hidden`), the classifier buffers about 8.6 GB, and the scratch
about 3.3 GB.

**The cause.** `BuildTrainWorkspace` in `src/model.cc` gives every layer its own
`ops::BlockActivations` set, and `RunForward` keeps all of them because
`Backward()` reads them in reverse order. One set is 20 units of `R * hidden`:
`pre_act` and `act` take 4 units each, and the other twelve slots take 1 unit
each. `rstd1`, `rstd2`, `ve_gate`, and `attn_stats` are small and stay.

**The design.** A compile-time level in `NANOCHAT_RECOMPUTE_LEVEL`, with 0 as
the default and the current behavior. A recomputed slot becomes one buffer for
the whole model, not one per layer: `BlockForward` writes it and uses it at
once, and `BlockBackward` rebuilds it into the same buffer at the start of the
layer. The rebuild costs one extra pass over a row.

| Level | Rebuild | Rebuild from | Slots removed |
|---|---|---|---|
| 1 | `act` via `kReluSquare` | `pre_act` | 4 of 20, 20% |
| 2 | `h` and `h2` | `xr`, `x_mid`, `rstd1`, `rstd2` | 6 of 20, 30% |
| 3 | `xr` via `kScaleAdd`, and `ve_values` | `x_in`, `x0`, the blend scalars, the tokens | 8 of 20, 40% |

Nine slots stay saved per layer: `q_pre`, `k_pre`, `q_final`, `k_final`,
`v_final`, `attn_out`, `x_mid`, `pre_act`, and `x_out`. A rebuild of the first
five repeats a matrix product or loses data that `QkPrepBackward` overwrites in
place, and `x_out` is the input of the next layer.

At level 2 the layer term falls from about 24 GB to about 16.8 GB, and the total
from about 36 GB to about 28.8 GB. The arena still does not fit 11 GB at this
batch. The win is a larger batch and a smaller peak.

The frozen seam does not change and no kernel is new. Every rebuild uses an
existing op: `kernels::PointwiseForward(kReluSquare)`, `ops::RmsNormForward`,
`kernels::PointwiseForward(kScaleAdd)`, and `kernels::EmbeddingForward`. The
model already passes the input of each block to `BlockBackward`, so the API
does not change either.

**Plan and gates.** Each phase is one session.

| Phase | Work | Gate |
|---|---|---|
| 0 | The build definition, and `workspace_bytes()` reports the level | The level-0 arena matches today |
| 1 | Share `act` and rebuild it in `MlpBackward` | The layer term drops near 20%, and train parity stays green at `1e-5` |
| 2 | Share `h` and `h2` and rebuild them in `BlockBackward` | The layer term drops near 30%, and a T3 timing reports the step cost |
| 3 | Rebuild `xr` and `ve_values` (optional) | Parity green, and the timing stays inside the budget |

Stop a phase when its gate fails, and report the negative result.

**Tests.** `tests/train_parity_test.cc` is the main gate, because a rebuild
reproduces the same value from the same input, so the trajectory must not move.
The finite-difference test, the arena comparison in `src/grad_mode_test.cc`, and
the `kReluSquare` and RMSNorm kernel tests follow it.

**Reproduction.**

```bash
tools/nanochat build --copt=-DNANOCHAT_RECOMPUTE_LEVEL=2
tools/nanochat test
tools/nanochat gpu --profile t2-parity -- \
  ./bazel-bin/tests/train_parity /tmp/train_parity_d8_s512_50.bin
```

### 2. Fused classifier

**The problem.** The classifier holds `raw_logits_` and `dlogits_`, each 4.3 GB
at `R = 32768` and `padded_vocab = 32768` in fp32. The arithmetic for each byte
is small, so the traffic dominates the cost. The logits buffer moves six times
per step, and two of those passes rebuild statistics that the forward pass
already had.

| Pass | Count |
|---|---:|
| Forward logits reads | 2 |
| Backward logits reads | 3 |
| Backward `dlogits_` writes | 1 |
| Scale pass over `dlogits_` | 1 read, 1 write |

**The design.** Two independent changes, bit-identical to the current result.

1. **The statistics hand-off.** `ClassifierForward` writes `(row_max, sum_exp)`
   per row, and `ClassifierBackward` reads the pair. The pair is `2 * rows`
   floats, about 256 KB. The value is the same bit for bit, because both calls
   use the same input, the same reduction, and the same order. This removes two
   logits reads and one `RowLogSumExp` call, which is two `tanh` passes over the
   row.
2. **The fused scale.** `ClassifierBackward` takes `dlogit_scale = scale / rows`
   from `BackwardAccumulate` and writes
   `dlogits[j] = (p - onehot) * sech2 * dlogit_scale`. The separate
   `PointwiseForward(kScale)` pass disappears. The order of the arithmetic does
   not change.

Together the backward logits reads fall from 3 to 1 and the scale passes from 2
to 0, which is about 17 GB less traffic per step.

**The rejected option.** `llm.c` overwrites the logits buffer with the gradients,
which would remove `dlogits_` and 4.3 GB of arena. This tree cannot do it.
`BackwardAccumulate` may run more than once per forward pass, and
`src/harness_test.cc` calls it twice with a half scale. The second call would
read gradients where it expects logits.

**The seam change.** Additive at the parameter level, and the architect owns
`include/nanochat/kernels.h`.

```cpp
// Forward writes the per-row statistics when `stats` is non-null:
// stats[2*r] = row_max, stats[2*r+1] = sum_exp. Evaluation passes null.
void ClassifierForward(const ClassifierParams& params,
                       const ComputeType* logits, const int* targets,
                       ComputeType* losses, float* stats = nullptr);

// Backward consumes the statistics when `stats` is non-null, and rebuilds them
// otherwise. `dlogit_scale` folds the loss normalization into the write.
void ClassifierBackward(const ClassifierParams& params,
                        const ComputeType* logits, const int* targets,
                        const float* stats, float dlogit_scale,
                        ComputeType* dlogits);
```

The default argument keeps the old call sites valid at the source level, and
both backends must define the new signature. The null path keeps the current
behavior, so the kernel tests compare the two paths.

**Plan and gates.**

| Phase | Work | Gate |
|---|---|---|
| 0 | The seam, in both backends | The T1 classifier test passes on both paths, and the results agree bit for bit |
| 1 | Allocate `classifier_stats_`, wire the model, pass null on eval | Train parity green |
| 2 | Add `dlogit_scale` and remove the scale pass | `harness_test.cc` green, including the double accumulation |
| 3 | Time the classifier | The numbers go to [campaigns.md](campaigns.md) |

**Tests.** The classifier and oracle tests prove correctness and finite
differences, `harness_test.cc` is the gate for the rejected option, and
`train_parity_test.cc` plus `eval_bench` cover the trajectory and the timing.

**Reproduction.**

```bash
tools/nanochat build
tools/nanochat test
tools/nanochat test --gpu //backends/cuda/kernels:classifier_gpu_test
tools/nanochat bench -- ./bazel-bin/src/eval_bench --batch 16 --seq 1024 \
  --layers 12 --heads 6 --kv-heads 6 --hidden 768 \
  --vocab 32768 --padded-vocab 32768
```

### 3. Vectorized row kernels

**The problem.** Every row kernel in `backends/cuda` uses scalar strided access.
A search finds no `__ldcs`, `__stcs`, `__ldg`, `int4`, or `float4`, and the
loops step one element at a time. The GTX 1080 Ti moves about 484 GB/s and
computes about 11.34 TFLOP/s, so the ridge point is about 23 flop for each byte.
The row kernels sit far below it and lose their time on memory traffic.
`row_benchmark.cc` reports milliseconds only, so the first task is to add the
achieved byte rate.

**The design.** One header-only helper in
`backends/cuda/kernels/device_utils.cuh`, which inlines and does not merge a
kernel boundary. `Packed<T>` is `alignas(16)` and holds `16 / sizeof(T)`
elements, so the fp32 build packs four. `LoadPacked` and `StorePacked` cast to
`int4`, and `LoadPackedStreaming` and `StorePackedStreaming` use `__ldcs` and
`__stcs`.

The streaming choice is per kernel. `RmsNormForward` streams `x` and `out`.
`RmsNormBackward` streams `x`, `dy`, and `dx`. Both pointwise passes stream
every operand. The classifier streams `logits`. `EmbeddingForward` streams the
output and keeps the table gather scalar, because it indexes the rows.
`GradNorm` streams each span. The attention softmax rows stream `scores`,
`probs`, and `dprobs`.

A kernel takes the packed loop only when three conditions hold: the pointer is
16-byte aligned, the row width divides by the packed count, and the row count
divides by the block stride. Otherwise it takes the scalar loop. The two loops
produce the same arithmetic, and the kernel checks the conditions once at the
start. The workspace aligns the arena to 64 bytes, and `Workspace::Reserve`
gains 16-byte alignment for the slots that the packed kernels read.

**Numerics.** A packed loop changes the order of the per-thread accumulation,
so the last bits of a sum can move. The fp32 oracle tolerance is `1e-5`, and the
finite-difference tests and the oracle fixture decide the result. The classifier
softmax keeps its scalar reduction: a packed reduction can move `row_max` by one
unit in the last place, and that is the sensitive case.

| Kernel | Operand | Change | Risk |
|---|---|---|---|
| `rms_norm.cu` | `x`, `out` | packed load and store | Low |
| `pointwise.cu` | `a`, `b`, `out` | packed load and store | Low |
| `embedding.cu` | `out`, `dtable` | packed store, scalar gather | Low |
| `grad_norm.cu` | `grads` | packed load | Low |
| `attention.cu` | softmax rows | packed load and store | Medium |
| `classifier.cu` | `logits` | packed load, scalar reduction | Medium |

The frozen seam does not change, and `include/nanochat/kernels.h` keeps every
signature.

**Plan and gates.**

| Phase | Work | Gate |
|---|---|---|
| 0 | The byte-rate column in `row_benchmark.cc` (0.5 session) | The benchmark reports the byte rate and the peak fraction |
| 1 | The helper, and `rms_norm`, `pointwise`, `embedding`, `grad_norm` (1 session) | Each family gains at least 10% in byte rate, and the oracle fixture stays green |
| 2 | `classifier.cu` and the attention softmax rows (1 session) | Both benchmarks improve, and train parity stays green |

If a phase fails its gate, report the negative result and stop the phase.

**Tests.** The existing device-against-host tests need no change when the
arithmetic does not change: the T1 family tests, the row-kernel smoke test, both
precisions, the padded tail, and the end-to-end oracle and parity gates.

---

## Part 2. The self-evolution system

Status: design. No part of this system is implemented. No phase changes a frozen
header, the model graph, or `src/**`. Phase 0 and Phase 1 are the first committed
scope, and every later phase waits for explicit human approval.

The loop accepts a goal, reasons with a checkpoint, acts on the tree, measures
the result, and asks for help when it is stuck. It records each help event,
trains on the record, and publishes a new checkpoint. Six jobs follow: accept a
goal; reason and act with a checkpoint; detect a stuck state; record every
transition as training data; train and evaluate a new checkpoint; publish and
fetch through the Hugging Face Hub. Every build, test, train, and eval goes
through `tools/nanochat`, and the loop uses [python.md](python.md),
[post-training.md](post-training.md), and [sandbox.md](sandbox.md).

**The layers.** Four layers sit above the L0 to L6 stack of
[../DESIGN.md](../DESIGN.md) section 2, and they do not renumber it. E4 is the
human surface: goal intake, approvals, answers, and stop. E3 is the loop:
controller, stuck detector, and safety gate. E2 holds the services: teacher
gateway, trajectory store, dataset builder, checkpoint registry, verifier, and
code editor. E1 is the existing toolchain and API. The dependencies point down.

**The principles.** The checkpoint is the substrate and the controller is only
process. The teacher is a scaffold that the trajectory dataset removes. A human
is the top tier. Every transition is data, a failed step and a human correction
included. Each resource has one writer, and a failed build discards the edit.
Every child job runs under `tools/nanochat`. The controller is the one exception:
it runs for hours and blocks on human input, so it sets
`NANOCHAT_SANDBOX_BACKEND=none` for itself.

| Role | Responsibility | Model |
|---|---|---|
| Controller | Owns the loop and the state | Deterministic code |
| Policy | Reasons and proposes an action | `nanochat` checkpoint |
| Teacher | Resolves a stuck step | External strong model |
| Human | Sets the goal, approves, and answers | Human |
| Verifier | Scores an outcome | Tests, evals, and rewards |
| Editor | Writes code and fixes a build | Policy or teacher |

The controller never generates text. It selects an action.

**The control loop.** One iteration is one step through `IDLE`, `PLAN`, `REASON`,
`ACT`, `OBSERVE`, `VERIFY`, `REFLECT`, and then one of `STUCK`, `ESCALATE`,
`WAIT_HUMAN`, `TRAIN`, `DEPLOY`, `DONE`, or `FAILED`. `ACT` runs the action under
the sandbox, `WAIT_HUMAN` blocks durably so that a restart resumes the goal, and
`TRAIN` unloads the reason model first. The controller enforces a budget of
tokens, teacher calls, GPU seconds, and wall clock at each transition, and a
spent budget parks the goal.

**The components.** The controller selects the state, enforces the budget, calls
the safety gate, and writes one trajectory record per transition. The goal
manager persists each goal with its budget and status. The policy has two
backends: `NanochatPolicy` loads a checkpoint through `api.py` on the local GPU,
and `TeacherPolicy` calls one configurable remote model. Either returns a shell
action, a code patch, a question, or a stop, and the executor runs it through
`toolchain.py`, never through Bazel or a binary directly. The stuck detector
fires on a repeated error, a flat verifier score, a step or budget limit, no
valid action, or a repeated action with no state change. The escalation manager
routes the event to the teacher, then to the human, and blocks the goal durably.
The trajectory store is an append-only JSON Lines log with a SQLite index and a
content hash per record, and the dataset builder renders it with `chat.py` for
the template and the loss mask and `sft_data.py` for the mixture. The verifier
treats a build or lint failure as a hard failure, then scores the
`tools/nanochat check` and `tasks.py` pass rates over a pre-populated cache, so a
scored goal needs no network. The code editor is Phase 5 only: it applies a
patch, builds and tests it, restores the previous revision on failure, and
records the failed patch as a negative sample. The lifecycle manager owns the
GPU tenure and frees the model with `nanochat_model_free` before it releases the
card. The registry holds the version graph, marks one version current, and
restores the parent when a metric does not improve, and the Hub transport
publishes the container through the `hf` command line interface. The safety gate
checks every action against a short rule list and a small denylist, and the
observer writes one log line per step.

**The data model.** A trajectory record carries `id` (the content hash),
`goal_id`, `step`, `state` (the chat history and a bounded workspace digest),
`action`, `observation` (exit code, output, and duration), `score`, `stuck`, the
`intervention` or null, `policy`, `checkpoint`, `revision`, and `time`. The data
splits into a positive set, an intervention set of teacher or human answers,
which has the highest value, and a negative set for a later preference method. A
checkpoint version carries `version`, `parent`, `revision`, `dataset` (the
training-set hash), `metrics`, and `hub_path`.

**The teacher exit.** Each intervention is a labeled example, and each training
round distills the teacher into the checkpoint. The loop drops the online teacher
when a checkpoint stays within 5 percent of the teacher pass rate over 20
consecutive held-out tasks with no escalation. The teacher returns only for a new
task type.

**GPU tenure.** There is one card, so the work is serial. Reasoning and training
both hold it, an evaluation holds it briefly, and a teacher call uses only the
network. The reason model and the training job therefore cannot share the card,
so the unload step is mandatory.

| Failure | Response |
|---|---|
| A build or a test fails | Discard the patch, restore the revision, record a negative sample |
| Training diverges | Restore the parent checkpoint |
| The Hub is down | Continue with the local cache |
| The teacher returns garbage | Retry once, then escalate to the human |
| The human does not answer | Park the goal and keep the loop on other goals |
| The GPU is busy | Wait in the queue |

**Safety.** The sandbox protects the host, not the goal, so a separate gate
applies. A code edit outside the owned directory, a Hub upload, and a Hub delete
each need human approval. The caps are 200k policy tokens, 10 teacher calls, one
GPU hour, and six hours of wall clock by default. One action writes one immutable
audit record. A stop file is checked at every transition, and `SIGINT` stops a
blocked controller.

**The layout.** One package `python/nanochat_cpp/evolve/` holds the modules named
above, plus one tool, `tools/evolve`. The workstream is "Evolution", and
[../AGENTS.md](../AGENTS.md) section 1 needs one row for it. The state lives
outside the tree under `~/.cache/nanochat_cpp/evolve/`, and one goal is active at
a time. The entry point gains one row in the command table of
`python/nanochat_cpp/_entry.py` and one branch in `tools/nanochat`, both of which
need architect approval. The editor never patches `evolve/**` or `tools/evolve`.
The system adds no C++ kernel and no public C function; it reuses `toolchain.py`,
`api.py`, `chat.py`, `sft.py`, `sft_data.py`, `rl.py`, `tasks.py`,
`checkpoint.py`, the C ABI in `include/nanochat/capi.h`, the `tools/nanochat`
entry point, and `notebooks/pi_agent.py`, which is the precedent for the teacher
path.

**Delivery.**

| Phase | Work |
|---|---|
| 0 | The controller, the goals, the executor, the trajectory store, and `tools/evolve`, over a scripted policy on `doctor`, `build`, `check`, and `lint`, plus one deliberate failure that triggers an escalation |
| 1 | The stuck detector, the escalation, the verifier, the store, the human surface, and a log of every intervention |
| 2 | The dataset builder, and SFT on the intervention set, gated by `tools/nanochat check` and the eval tasks; it waits on [post-training.md](post-training.md) section 12 |
| 3 | The lifecycle manager, the registry, and the Hub, proving one cycle of load, reason, unload, train, and upload |
| 4 | `TeacherPolicy`, the distillation gap, and its closure |
| 5 | The editor and the safety gate, so a code patch runs with a rollback gate |

**Risks.** A weak verifier lets the loop drift, so keep held-out tests against
reward hacking. The teacher spend can grow, so keep the hard caps. An unsafe edit
can break the host, so keep the rollback and approval gates. SFT can degrade an
old skill, so keep a mixed dataset and a regression suite. The controller has no
per-process host limit and relies on `nanochat.slice`. An overnight round blocks
at the upload step, because every upload needs a human.

The five open questions of the first draft are resolved: the verifier is a hard
build and lint gate then a pass rate; the teacher is one configurable external
model; every upload and delete needs approval; the exit test is the 5 percent
rule above; and the registry stores a full container on the Hub with the hash and
the code revision in the local index.
