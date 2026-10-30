# Self-evolution system design

This document designs a self-evolution loop on top of `nanochat.cpp`. The loop
accepts a goal, reasons with a checkpoint, acts on the tree, measures the
result, and asks for help when it is stuck. It records every help event. It
trains on the record. It publishes a new checkpoint.

Status: design. No part of this system is implemented. This document is the
plan of record. It adds no graph, no autograd engine, and no vendor header in
`src/**`. It does not change a frozen header.

The first committed scope is Phase 0 and Phase 1. Every later phase is
deferred.

The loop sits above the existing stack. It calls `tools/nanochat` for every
build, test, train, and eval. It uses the Python surface in
[python.md](python.md), the post-training stages in
[post-training.md](post-training.md), the resource sandbox in
[sandbox.md](sandbox.md), and the optimization rules in
[../DESIGN.md](../DESIGN.md).

## 1. Purpose

The system turns the training stack into a closed loop. A human gives a goal.
The loop tries to reach the goal. When the loop fails, it escalates. The
escalation becomes training data. The next checkpoint is a little better.

The loop has six jobs:

1. Accept a goal from a human or a backlog.
2. Reason and act with a `nanochat.cpp` checkpoint.
3. Detect a stuck state and escalate.
4. Record each transition as training data.
5. Train, evaluate, and save a new checkpoint.
6. Publish and fetch checkpoints through the Hugging Face Hub.

## 2. Scope

In scope:

- One controller process on one host.
- A `nanochat.cpp` policy and an external teacher policy.
- A stuck detector and a human escalation channel.
- A trajectory store and a dataset builder.
- A training round over SFT and RL.
- A checkpoint registry and a Hub transport.
- A rollback gate for a code edit.

Out of scope:

- A change to the model architecture at run time.
- A new autograd engine or graph compiler.
- A permanent dependence on an external model.
- Unattended code changes without a rollback gate.
- Distributed training across hosts.

The fixed graph in `../DESIGN.md` section 5 forbids the first item. The
external model is a scaffold. Section 8.4 explains the exit.

## 3. Design principles

1. **The checkpoint is the substrate.** The checkpoint holds the learned
   policy. The controller holds the process.
2. **The teacher is a scaffold, not a component.** The teacher removes human
   work. The trajectory dataset removes the teacher.
3. **The human is the top tier.** A human sets the goal and resolves
   ambiguity.
4. **Every transition is data.** A failed step and a human fix are both
   training samples.
5. **One writer per resource.** The code, the checkpoint, and the GPU each
   have one owner at a time.
6. **The build is the gate.** A failed build discards the edit and stops the
   step.
7. **Every job uses the entry point.** `tools/nanochat` applies the sandbox and
   submits the job to the GPU queue. See [sandbox.md](sandbox.md) and
   [gpu-queue-plan.md](gpu-queue-plan.md). The controller is not a job. It
   runs for hours and it blocks on human input, so it sets
   `NANOCHAT_SANDBOX_BACKEND=none` for itself. Every child job stays under
   `tools/nanochat`.

## 4. System context

The new layers sit above the L0-L6 stack in `../DESIGN.md` section 2. They do
not renumber it. Call them E1 to E4.

```text
+------------------------------------------------------------------+
| E4  Human surface    goal intake | approvals | answers | stop     |
+------------------------------------------------------------------+
| E3  Evolution loop   controller | stuck detector | safety gate    |
+------------------------------------------------------------------+
| E2  Services         teacher gateway | trajectory store          |
|                      dataset builder | checkpoint registry       |
|                      verifier | code editor                       |
+------------------------------------------------------------------+
| E1  Orchestration    toolchain | api | sft | rl | chat | tasks    |
+------------------------------------------------------------------+
| L0-L6  nanochat.cpp stack (kernels, model, optim, harness, CLI)   |
+------------------------------------------------------------------+
```

The dependencies point down. E3 uses E2 and E1. E2 uses E1. E1 uses the
existing stack. No layer reaches up. This keeps the dependency invariants in
`../DESIGN.md` section 2.1.

## 5. Roles

| Role | Responsibility | Model |
|---|---|---|
| Controller | Owns the loop and the state | Deterministic code |
| Policy | Reasons and proposes an action | `nanochat` checkpoint |
| Teacher | Resolves a stuck step | External strong model |
| Human | Sets the goal, approves, and answers | Human |
| Verifier | Scores an outcome | Tests, evals, and rewards |
| Editor | Writes code and fixes a build | Policy or teacher |

The controller never generates text. It selects an action. The policy and the
teacher generate text.

## 6. The control loop

The controller is a state machine. One iteration is one step.

```text
IDLE -> PLAN -> REASON -> ACT -> OBSERVE -> VERIFY -> REFLECT
                    ^                                   |
                    |                                   v
               ESCALATE <------- STUCK            TRAIN -> DEPLOY
                    |                                   |
               WAIT_HUMAN -------------------------> IDLE
```

| State | Action | Exit condition |
|---|---|---|
| `IDLE` | Wait for a goal | A goal is active |
| `PLAN` | Split the goal into steps | A step list exists |
| `REASON` | Load the checkpoint, propose an action | An action is ready |
| `ACT` | Run the action under the sandbox | An observation exists |
| `OBSERVE` | Record the result | The result is stored |
| `VERIFY` | Score the result | A score exists |
| `REFLECT` | Update the step list | Progress, or a stuck flag |
| `STUCK` | Raise the stuck flag | The detector fires |
| `ESCALATE` | Ask the teacher, then the human | An answer exists |
| `WAIT_HUMAN` | Block and wait, durably | A human message arrives |
| `TRAIN` | Unload, train, and save | A new checkpoint exists |
| `DEPLOY` | Publish the checkpoint | The Hub upload succeeds |
| `DONE` | Close the goal | The goal is closed |
| `FAILED` | Park the goal | A human restarts it |

The controller enforces a budget at each transition. The budget covers tokens,
teacher calls, GPU seconds, and wall clock. A spent budget parks the goal.

## 7. Components

### 7.1 Goal manager

The manager holds the goal list. One goal has a description, a budget, a
status, and a step list. The status is one of `open`, `active`, `blocked`,
`done`, or `failed`. The manager persists each goal to the store. A restart
resumes an `active` goal.

### 7.2 Controller

The scheduler. It selects the next state. It enforces the budget. It calls the
safety gate before each action. It writes one trajectory record per
transition.

### 7.3 Policy

One interface, two backends.

- **`NanochatPolicy`** loads a checkpoint through
  `python/nanochat_cpp/api.py`. It generates a tool call or a patch. It runs
  on the local GPU.
- **`TeacherPolicy`** calls one configurable external model. It runs on a
  remote endpoint. It needs no GPU. A council is out of scope for now.

The interface returns a proposal. The proposal is one of: a shell action, a
code patch, a question, or a stop.

### 7.4 Tool executor

The executor runs the proposal through `python/nanochat_cpp/toolchain.py`. It
never calls Bazel or a binary directly. It returns an observation with a
return code, standard output, standard error, and a duration.

### 7.5 Stuck detector

The detector fires on any of these conditions:

- The same error appears in two consecutive steps.
- The step count passes a threshold with no goal progress.
- The verifier score stays flat over a window.
- The budget is nearly spent.
- The policy emits no valid action.
- The action repeats with no state change.

The detector does not ask the teacher for a low-confidence action. It asks
only when a rule fires. This keeps the cost down.

### 7.6 Escalation manager

The manager routes a stuck event to the teacher first, then to the human. It
writes the question and the context. It blocks the goal. The block is durable.
A restart resumes the goal. The answer becomes an intervention record.

### 7.7 Trajectory store

An append-only log. Use JSON Lines for the raw record, and SQLite for the
index. The record carries a content hash. The hash makes the dataset
reproducible.

### 7.8 Dataset builder

The builder renders trajectories into the training formats. It uses
`python/nanochat_cpp/chat.py` for the chat template and the loss mask, and
`python/nanochat_cpp/sft_data.py` for the mixture of conversations.

The supervised tokens are the assistant text, the tool call, and the human
correction. The ignored tokens are the user text and the tool output. The
renderer already masks the tool-output tokens. See [post-training.md](post-training.md)
section 3.

### 7.9 Verifier

The verifier scores an outcome. For a code goal the build and the lint gate
are hard failures. The score is then the `tools/nanochat check` pass-rate and
the `python/nanochat_cpp/tasks.py` pass-rate. The verifier reads a
pre-populated `~/.cache/nanochat/task_data` and a small offline fixture, so a
scored goal needs no network. The reward path in `python/nanochat_cpp/rl.py`
is not in the tree yet (see [post-training.md](post-training.md) section 12).

### 7.10 Code editor

The editor is Phase 5 only. It stays off until the human approves it. In
Phase 0 and Phase 1 the action surface is the toolchain plus read-only file
inspection.

The editor applies a patch. It then runs a build and a test through the
toolchain. If the build or the test fails, it discards the patch and restores
the previous revision. It records the failed patch as a negative sample. The
editor never writes inside `python/nanochat_cpp/evolve/**` or `tools/evolve`.

### 7.11 Model lifecycle manager

The manager owns the GPU tenure. The single card allows one tenant. The manager
coordinates with the GPU queue.

```text
UNLOADED --load--> REASONING --unload--> UNLOADED
UNLOADED --load--> TRAINING  --unload--> UNLOADED
```

A transition needs the card to be free. The manager frees the model with
`nanochat_model_free` before it releases the card. This frees the reason model
before the training job starts.

### 7.12 Checkpoint registry and Hub transport

The registry holds the version graph. One version has a parent, a code
revision, a dataset hash, and a metric set. The Hub is the durable store. The
local cache is the working copy. Use the `hf` command line interface for
upload and download.

The registry marks one version as current. If the metric does not improve, the
registry restores the parent. The converter in
`python/nanochat_cpp/checkpoint.py` reads a torch `state_dict` and writes the
checkpoint container. The Hub stores the released container.

### 7.13 Safety gate

The gate runs before each action. It checks these rules:

- The action stays inside the sandbox profile.
- A code edit touches no frozen header without approval.
- A Hub upload waits for approval.
- The budget is not exceeded.
- The goal is not a prohibited goal.
- A write outside the owned directory has human approval.

The prohibited list is a small hard-coded denylist. Each approval travels on
the durable answer queue of section 7.6.

### 7.14 Observer

The observer writes structured logs and metrics. One step writes one log line.
One goal writes one summary. Section 13 lists the metrics.

## 8. Data model

### 8.1 Trajectory record

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Content hash |
| `goal_id` | string | The parent goal |
| `step` | integer | The step index |
| `state` | object | The chat history and a bounded workspace digest |
| `action` | object | The proposal and its arguments |
| `observation` | object | The exit code, the output, and the duration |
| `score` | float | The verifier score |
| `stuck` | boolean | The detector output |
| `intervention` | object or null | The teacher or human answer |
| `policy` | string | `nanochat`, `teacher`, or `human` |
| `checkpoint` | string | The model version |
| `revision` | string | The code revision |
| `time` | string | The timestamp |

### 8.2 Dataset splits

- **Positive set:** steps with a good score.
- **Intervention set:** steps with a teacher or human answer. These have the
  highest value.
- **Negative set:** steps with a bad score. These support preference methods
  later.

### 8.3 Checkpoint version

| Field | Meaning |
|---|---|
| `version` | A monotonic label |
| `parent` | The previous version |
| `revision` | The code revision |
| `dataset` | The hash of the training set |
| `metrics` | The evaluation numbers |
| `hub_path` | The Hub location |

### 8.4 Teacher exit

The teacher dependence is temporary. Each intervention is a labeled example.
Each training round distills the teacher into the checkpoint. When the
checkpoint matches the teacher on the task set, the loop drops the online
teacher. It keeps the teacher only for a new task type. Section 13 defines the
number that measures the gap.

The exit test is a checkpoint pass-rate within 5 percent of the teacher over
20 consecutive held-out tasks, with no escalation.

## 9. Key sequences

### 9.1 Normal step

1. The lifecycle manager loads the current checkpoint.
2. The policy proposes a tool call.
3. The executor runs the call.
4. The verifier scores the result.
5. The controller records the transition.
6. The lifecycle manager unloads the model.

### 9.2 Stuck step and intervention

1. The stuck detector fires.
2. The escalation manager asks the teacher.
3. The teacher answers. The answer becomes an action.
4. The executor runs the action.
5. The trajectory records the teacher turn as a supervised target.
6. If the teacher fails, the manager asks the human. It blocks the goal.

### 9.3 Training round

1. The controller closes a batch of goals.
2. The dataset builder renders the batch.
3. The lifecycle manager unloads the reason model.
4. The training orchestrator runs SFT or RL through the toolchain.
5. The verifier evaluates the new checkpoint.
6. If the metric improves, the registry keeps the checkpoint.
7. If the metric does not improve, the registry restores the parent.
8. The registry uploads the checkpoint to the Hub.

The SFT stage uses `python/nanochat_cpp/sft.py`. The RL stage uses the
persistent worker in `python/nanochat_cpp/rl.py`. Both run through
`tools/nanochat train`.

## 10. GPU tenure and scheduling

There is one GPU. The loop must serialize the GPU work.

| Work | Resource | Notes |
|---|---|---|
| Reasoning | GPU held | Short and interactive |
| Teacher call | Network only | The GPU stays free |
| Training | GPU held | Long and exclusive |
| Evaluation | GPU held | Short |
| Build and test | CPU, and the GPU test tier | Through the queue |

The table has one consequence. The reason model and the training job cannot
share the card. The unload step is mandatory, not optional. The lifecycle
manager enforces this rule.

## 11. Failure handling

| Failure | Response |
|---|---|
| A build fails | Discard the patch, restore the revision, record a negative sample |
| A test fails | The same response as a build failure |
| Training diverges | Restore the parent checkpoint |
| The Hub is down | Continue with the local cache |
| The teacher returns garbage | Retry once, then escalate to the human |
| The human does not answer | Park the goal, keep the loop on other goals |
| The GPU is busy | Wait in the queue |

## 12. Safety

The sandbox protects the host. It does not protect the goal. Add a separate
safety gate.

- Require human approval for a code edit outside the owned directory.
- Require human approval for a Hub upload and a Hub delete.
- Cap the teacher spend, the token count, and the GPU seconds. The default cap
  is 200k policy tokens, 10 teacher calls, 1 GPU-hour, and 6 hours of wall
  clock.
- Keep an audit log. One action writes one immutable record. The durable
  answer queue carries each question and each approval.
- Support a hard stop. A stop file is checked at each transition, and SIGINT
  stops a blocked controller.
- Run the controller with `NANOCHAT_SANDBOX_BACKEND=none`. Every child job
  keeps its own profile.

## 13. Observability

Track these numbers per goal and per checkpoint:

- Goal success rate.
- Human interventions per goal. This number should fall.
- Teacher calls per goal. This number should fall.
- The distillation gap. This is the pass-rate difference between the
  checkpoint and the teacher.
- The evaluation score, for example bits per byte and GSM8K.
- GPU seconds per goal.
- Cost per goal.

The two falling numbers are the health signal. If they do not fall, the
distillation is not working.

## 14. Module layout and ownership

Add one Python package and one tool. The package lives under the Python
surface. The tool follows the `tools/nanochat` convention.

```text
python/nanochat_cpp/evolve/
  controller.py    the loop and the state machine
  goals.py         the goal manager
  policy.py        NanochatPolicy and TeacherPolicy
  executor.py      the toolchain wrapper
  stuck.py         the stuck detector
  escalation.py    the teacher and human routing
  trajectory.py    the record type and the store
  dataset.py       the dataset builder
  verifier.py      the scoring and the reward
  editor.py        the patch, the build, and the rollback
  lifecycle.py     the GPU tenure
  registry.py      the checkpoint versions
  hub.py           the Hugging Face Hub transport
  safety.py        the action gate
  store.py         JSON Lines and SQLite

tools/evolve       the entry point, one command table row
docs/self-evolution.md   this design
```

Ownership follows `../AGENTS.md` section 1. The new workstream is "Evolution".
It owns `python/nanochat_cpp/evolve/**`, `tools/evolve`, and
`docs/self-evolution.md`. The evolution state lives out of the tree under
`~/.cache/nanochat_cpp/evolve/`. One goal is active at a time. It depends on the toolchain, the compute layer, SFT,
RL, and the checkpoint converter. It does not edit a frozen header.

The entry point adds one row to the command table in
`python/nanochat_cpp/_entry.py` and one branch in `tools/nanochat`. Both files
need architect approval, because the table is the single source of truth. The
new workstream also needs one row in `../AGENTS.md` section 1.

The editor never patches `python/nanochat_cpp/evolve/**` or `tools/evolve`.
The controller defers all code edits to Phase 5.

## 15. Interfaces reused from the tree

The system adds no C++ kernel and no public C function. It reuses these
interfaces.

| Interface | File | Use |
|---|---|---|
| `toolchain` | `python/nanochat_cpp/toolchain.py` | Build, test, lint, train, eval |
| `api` | `python/nanochat_cpp/api.py` | Model, optimizer, generate, score |
| `chat` | `python/nanochat_cpp/chat.py` | The chat template and the loss mask |
| `sft` | `python/nanochat_cpp/sft.py` | The SFT shard and the SFT run |
| `sft_data` | `python/nanochat_cpp/sft_data.py` | The conversation mixture |
| `rl` | `python/nanochat_cpp/rl.py` | The rollout and advantage loop |
| `tasks` | `python/nanochat_cpp/tasks.py` | GSM8K, HumanEval, ARC, MMLU |
| `checkpoint` | `python/nanochat_cpp/checkpoint.py` | The checkpoint container and its run metadata |
| C ABI | `include/nanochat/capi.h` | The in-process compute calls |
| Tool entry | `tools/nanochat` | The sandbox and the GPU queue |
| Agent runner | `notebooks/pi_agent.py` | An external coding agent |

The agent runner is the precedent for the teacher path. It already streams the
event stream of the Pi coding agent.

## 16. Delivery plan

### Phase 0 — the loop

Build `controller.py`, `goals.py`, `executor.py`, `trajectory.py`, and
`tools/evolve`. Run the loop with a scripted policy. Make no learning change.
The scripted goal list is `doctor`, `build`, `check`, and `lint`, plus one
deliberate failure that triggers escalation. Phase 0 and Phase 1 are the first
committed scope.

### Phase 1 — human in the loop

Add `stuck.py`, `escalation.py`, `verifier.py`, and `store.py`. Add the human
surface. Log every intervention.

### Phase 2 — learn from intervention

Add `dataset.py`. Run SFT on the intervention set. Gate the result with
`tools/nanochat check` and the eval tasks. This phase waits on the unfinished
post-training work in [post-training.md](post-training.md) section 12.

### Phase 3 — model swap and Hub

Add `lifecycle.py`, `registry.py`, and `hub.py`. Prove one full cycle: load,
reason, unload, train, and upload.

### Phase 4 — teacher and distillation

Add `policy.py` with `TeacherPolicy`. Measure the distillation gap. Close the
gap.

### Phase 5 — code self-modification

Add `editor.py` and `safety.py`. Allow a code patch with a rollback gate. Keep
the frozen-header rule. This phase starts only after explicit human approval.

## 17. Risks

- **A weak verifier.** The loop drifts. The verifier must be strong and cheap.
- **Reward hacking.** The model games the score. Keep held-out tests.
- **Cost blowout.** The teacher spend grows. Keep hard budget caps.
- **An unsafe edit.** A code patch breaks the host or the data. Keep the
  rollback gate and the approval gate.
- **Catastrophic forgetting.** SFT degrades an old skill. Keep a mixed dataset
  and a regression suite.
- **Single-card contention.** Training and reasoning compete. Keep the
  lifecycle manager strict.
- **An unsandboxed controller.** The controller has no per-process host limit.
  A runaway controller relies on the aggregate `nanochat.slice` alone.
- **An unattended Hub upload.** The approve-every-upload rule needs a human at
  the wall clock, so an overnight round blocks at the upload step.
- **Unproven worktree rollback.** The worktree-per-goal rule must respect the
  one-Bazel-server-per-worktree rule and the prune workflow in
  [sandbox.md](sandbox.md).

## 18. Open questions

All five questions are resolved.

1. Verifier: a hard build and lint gate, then the `tools/nanochat check` and
   `tasks.py` pass-rate.
2. Teacher: one configurable external model.
3. Hub: human approval for every upload and every delete.
4. Teacher exit: a pass-rate within 5 percent of the teacher over 20
   consecutive held-out tasks, with no escalation.
5. Registry: a full checkpoint container snapshot on the Hub, with the hash and
   the code revision in the local index.

## 19. References

- [../DESIGN.md](../DESIGN.md) — the layer model, the autograd policy, and the
  evolution rules.
- [python.md](python.md) — the compute, planning, and orchestration layers.
- [post-training.md](post-training.md) — SFT and RL, and the weighted
  classifier.
- [training-seam.md](training-seam.md) — the Python and C++ boundary.
- [rl-notebook.md](rl-notebook.md) — the RL worker bridge.
- [sandbox.md](sandbox.md) — the resource sandbox and the GPU queue.
- [testing.md](testing.md) — the test tiers and the Definition of Done.
- [eval.md](eval.md) — the evaluation surfaces.
