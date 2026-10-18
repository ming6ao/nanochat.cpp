# The training seam between Python and C++

Status: proposed. This document specifies the boundary between the Python
surface and the C++ training driver. It also records the defects in the current
boundary. The project completed the first increment of section 8, and section 5
marks the fixed defects. Section 6.6 extends the boundary to the mesh training
driver, and phase 6 of section 8 delivers it.

`post-training.md` section 11 stays the status of record for the post-training
features. This document is the authority for the Python and C++ boundary. It
revises section 4.1 of [post-training.md](post-training.md) on the owner of the
packer.

The companion documents are:

- [python](python.md) for the Python surface.
- [post-training](post-training.md) for supervised fine-tuning and
  reinforcement learning.
- [eval](eval.md) for the evaluation seam.
- [model](model.md) for the model interface.

## 1. Purpose

`DESIGN.md` section 2 puts the training driver at level L5. Level L5 holds
`TrainLoop`, `Scheduler`, `Checkpointer`, `Logger`, and the MFU accounting. The
driver is C++ by design.

The Python surface must not rebuild that driver. Today it does.

The compute layer in `python/nanochat_cpp/api.py` exposes `forward_loss`,
`backward`, and `optimizer.step`. A caller therefore writes a training loop in
Python. The `Trainer` class covers pretraining only. A user who wants to
fine-tune writes the loop by hand, as the Kaggle notebook does. That loop
duplicates `TrainLoop`.

The mesh client repeated the same defect. `notebooks/mesh_train.py` owned the
step loop, the metric record, the MFU, the summary, and the job serializer. The
C++ mesh runtime provided one step and no loop. Section 5.8 records the defect,
and section 6.6 gives the target.

This document defines the correct split. It then proposes the smallest interface
that removes the duplication.

## 2. The seam rule

The repository already states the rule for evaluation. [eval.md](eval.md)
section 2 says "metric in Python, compute in C++." Python owns the datasets, the
prompt rendering, the answer extraction, the code execution, and the metric
reduction. C++ owns the model compute behind a few primitives.

This document proposes the same shape for training. The proposed rule is:

- C++ owns mechanism. Mechanism is the per-step arithmetic and the state that
  the arithmetic needs.
- Python owns policy and data. Policy is the dataset choice, the chat template,
  the reward, and the task metric.

Mechanism and policy meet at one interface. The interface carries data. It does
not carry the per-step control flow, with one exception for reinforcement
learning (section 6.4).

## 3. What C++ owns

The step arithmetic already lives in C++ and is correct.

| Item | Location |
|---|---|
| Forward loss with a valid-token mean | `Model::ForwardLoss` (`include/nanochat/model.h`) |
| Hand-written backward | `Model::Backward` |
| Gradient accumulation | `Model::BackwardAccumulate(scale)` |
| Weighted policy-gradient backward | `Model::BackwardWeighted(row_weights, scale)` |
| Forward plus backward plus one optimizer step | `Model::TrainStep` |
| Valid-token normalization | `CountValidTargets` (`src/model.cc`) |
| Batched, tokenizer-agnostic sampling | `GenerateBatch` |
| AdamW and Muon | `src/optim.cc` |
| Learning-rate schedule | `Scheduler` (`include/nanochat/scheduler.h`) |
| Checkpoint container and optimizer state | `Checkpointer`, `Checkpoint` |
| Step loop, logging, evaluation, checkpoint interval | `TrainLoop` (`src/train.cc`) |

The gaps are in the driver, not in the arithmetic. `TrainLoop` reads parquet
documents through `DataLoader` and uses the unweighted path only. The mesh
driver gained a packed `NCSFT001` shard path and a weights-only warm start
(section 6.2). The `TrainLoop` packed-row source and the weighted step remain
open.

## 4. What Python owns

Python owns four items. None of them needs tensor math.

1. Data acquisition and mixture. The reference mixes SmolTalk, MMLU, and GSM8K.
2. The chat template and the loss mask. Python turns a conversation into token
   ids and a target list with `-1` at an ignored position.
3. The reward and the answer extraction. GSM8K and HumanEval need text logic.
   HumanEval also runs generated code in a guarded executor.
4. The task metrics. [eval.md](eval.md) already grants this to Python.

The planning layer in `python/nanochat_cpp/plan.py` also stays in Python. It
needs two numbers from C and no tensors.

The split of item 2 is precise. Python owns the renderer and the mask. C++ owns
the row packing, the padding, and the batching.

## 5. Defects in the current seam

The first increment fixed 5.1, 5.2, and 5.6, and partly fixed 5.3. The mesh
driver phase fixed 5.8. The other items remain.

### 5.1 The C++ interface is richer than the C interface (fixed)

`Model::TrainStep` implements forward, backward, and one optimizer step
(`src/model.cc:951`). The first increment added `nanochat_train_step`, and
`Model.train_step` calls it (`python/nanochat_cpp/api.py`). The `Trainer` keeps
the lower-level calls, because it supports gradient accumulation.

### 5.2 The C interface has no accumulation call (fixed)

`Model::BackwardAccumulate(scale)` exists and the tests cover it. The first
increment added `nanochat_backward_accumulate`, and `Trainer.__next__` calls
`Model.backward_accumulate` (`python/nanochat_cpp/api.py`). The host float array
of `batch * seq` entries is gone.

### 5.3 The schedule leaks to the caller (partly fixed)

`Optimizer::Step(int step)` takes a 1-based step from the caller
(`include/nanochat/optim.h:56`). `TrainLoop` now persists the step and resumes
from it, so a driver owns the counter. A custom Python loop still passes the
step. Recommendation 4 in section 10 keeps `Optimizer::Step` in C++.

### 5.4 The Python trainer duplicates the C++ loop

`Trainer` reimplements gradient accumulation, the epoch reset, and `zero_grad`
(`python/nanochat_cpp/api.py:800-833`). `src/train.cc` is the source of truth
for the same logic. The Python tests do gate the loss curve against a C++
fixture (`python/tests/api_test.py:143`). The gate compares per-step losses only.
The loop structure can still drift.

### 5.5 There is no supervised fine-tuning or reinforcement-learning driver
(partly fixed)

`Trainer` takes `TokenData`, which reads parquet. The mesh driver now reads a
packed `NCSFT001` shard and does a weights-only warm start, so `posttrain.ipynb`
runs through the C++ driver (section 6.2). There is still no `SftTrainer` or
`sft_main` binary, and no `RlTrainer`. The user writes mechanism for the
reinforcement-learning path. Section 11 of
[post-training.md](post-training.md) lists the missing parts.

### 5.6 The renderer is private and incomplete (fixed)

The first increment added the public `render_conversation` in
`python/nanochat_cpp/chat.py`. `ChatEvaluator` and `python/tests/chat_test.py:68`
use the private `_PromptRenderer`, which stays private. The `_specials` helper
stays internal.

### 5.7 Forward-looking risk: the reinforcement-learning divisor

Reinforcement learning does not exist yet, so no code has this defect today.
The risk is for the future.

`Model::BackwardWeighted(row_weights, scale)` takes raw per-row weights.
[post-training](post-training.md) gives the divisor as the product of the valid
count, the pass count, and the example count per rank. Python would compute
that divisor today. The normalization is therefore not the single source of
truth. The divisor must move into C++ with the reinforcement-learning step.

### 5.8 The mesh client rebuilt the driver (fixed)

`notebooks/mesh_train.py` drove the mesh runtime one batch at a time. It owned
the loop, the record, and the summary. Each item in the first four rows
duplicated a C++ source of truth. The last two rows had no C++ writer:

| Python item | C++ source of truth |
|---|---|
| `iterate()` | `TrainLoop::Run` (`src/train.cc`) |
| `mfu()`, `steady_mfu()` | `ComputeMfu` (`include/nanochat/mfu.h`) |
| `format_row()` | `Logger::Log` (`src/train.cc:70`) |
| `job()`, `from_job()` | `TrainConfig` (`src/train.h:32`) |
| the `metrics.jsonl` writer | none |
| the `summary.json` writer | none |

The two log formats differed. `Logger::Log` prints `lr`, the timing split, and
the evaluation time. `format_row` printed the gradient norm, the step time,
the compute time, the throughput, and the MFU. So one record had two shapes.

`job()` and `from_job()` were a hand-written serializer of twenty fields. No
version marked the payload, so a stale `job.json` lost a field in silence.

`src/mesh_runtime.cc` provides `TrainStep` and no loop. So the client could not
delegate the loop. [client-runtime-design](client-runtime-design.md)
section 10.1 and section 20.5 record the decision. `src/mesh_train.h` and
`src/mesh_train.cc` now own the loop, the record, and the summary. Python owns
the job file, and the record carries a version.

## 6. Target design

### 6.1 Extend the C interface

Add the calls below to `include/nanochat/capi.h`. Each call maps to one existing
C++ method or one new driver method.

| Call | Maps to |
|---|---|
| `nanochat_train_step` | `Model::TrainStep`; returns the loss |
| `nanochat_backward_accumulate` | `Model::BackwardAccumulate` |
| `nanochat_rl_step` | a new file-driven reinforcement-learning step |

No new batch type is necessary. A target of `-1` already encodes the loss mask
for supervised fine-tuning (`include/nanochat/model.h:56`). The weighted
backward already carries the per-row weight for reinforcement learning
(`include/nanochat/model.h:76`). A separate mask field would create two mask
channels with no precedence rule.

### 6.2 The packed-row data path (partly done)

`DocumentSource` yields raw text (`include/nanochat/dataloader.h:22`). The
`DataLoader` tokenizes and packs that text. A supervised fine-tuning source
cannot use `DocumentSource`, because Python already rendered the text into token
ids and targets.

The mesh driver now reads one packed `NCSFT001` shard. Python writes the shard
with `nanochat_cpp.sft.pack`. The shard is a typed-record file
(`src/record_file.h`) with two `int32` records, `tokens` and `targets`, each of
shape `[rows, seq]`. A target of `-1` marks an ignored position. The file is
self-describing and versioned, so `src/sft_shard.h` holds only the typed view,
and `sft_main` and the mesh driver share one loader. The driver loads the shard
and cycles the rows, so it needs no reader thread.

`TrainLoop` still reads parquet only. A packed-row source beside
`DocumentSource` in `include/nanochat/dataloader.h` remains open, so a
parquet-free `TrainLoop` can run. The header is harness-owned, so that change
needs no frozen-header edit.

### 6.3 Explicit rates and the warm start (partly done)

The mesh driver takes an explicit optimizer and schedule, so a fine-tuning run
passes its own rates and does not use the `track3` preset. It also takes a
weights-only warm start: the driver loads the parameter records of a base
checkpoint and keeps the fresh optimizer state
(`docs/checkpoint-format.md` section 9.1).

`TrainLoop` (`src/train.h`) still lacks the packed-row source and these items
from [post-training.md](post-training.md) section 12:

1. A progress-based learning-rate schedule.
2. A flat or ramped Muon momentum option.
3. Hyperparameter inheritance from a base checkpoint.
4. An optimizer-state warm start from a base checkpoint.

The driver owns the step counter. The caller must not pass a step number for
the schedule. `Checkpointer` must also restore the counter, or the resume will
restart the schedule.

### 6.4 The reinforcement-learning path is a message loop

Reinforcement learning differs from supervised fine-tuning. The reward lives in
Python, and Python must compute it for each rollout. Python is therefore in the
control loop for every step. This is the one exception to the data-only rule.

The design follows [post-training.md](post-training.md) section 8:

- A file-driven `rl_step` binary consumes rollouts and advantages from a
  fixture. It is the correctness gate for the objective, and it uses
  `nanochat_rl_step`.
- A persistent worker holds the model and the optimizer. It exchanges rollout
  and advantage messages with the Python bridge over pipes.

The divisor from section 5.7 lives in the C++ step, not in Python.

### 6.5 Keep the Python surface thin

Python keeps four jobs:

1. Render a batch of conversations to tokens and targets.
2. Download and mix the datasets.
3. Compute the reward and extract the answer.
4. Report the task metrics.

For supervised fine-tuning, Python then writes a packed shard and launches the
C++ driver. Python never steps the optimizer.

Proposed Python shape:

```python
# Python renders the conversations and writes a packed shard.
shard = nc.sft.pack("chatsft.nchsft", conversations, tokenizer, seq_len=512)
# The facade launches the C++ driver through tools/nanochat, so the
# sandbox and the GPU queue apply.
nc.sft.run(shard, "chatsft.nchkpt01", num_iterations=1000)
```

`nc.sft.run` calls the toolchain, which runs `tools/nanochat train -- sft_main
...`. It does not run an in-process loop.

A workstation run therefore obeys the entry-point rule in `AGENTS.md` section
8. A Kaggle notebook uses the `none` sandbox backend, as [python](python.md)
section 3.1 describes.

`render_conversation` becomes a public function in `nanochat_cpp.chat`. The
underscore helpers stay internal.

The reference mixture and the packer live in `nanochat_cpp.sft_data`. The
mixture is SmolTalk, MMLU, and GSM8K. `sft_data.pack_rows` implements the
BOS-aligned best-fit packer with BOS padding and a `-1` mask. It mirrors
`sft_data_generator_bos_bestfit` in `scripts/chat_sft.py`.

### 6.6 The mesh path uses one C++ driver

The mesh path follows the same split. The C++ `src/mesh_train.cc` driver owns
the loop, the metric record, the summary, the checkpoint interval, and the
resume. It owns the parquet loader too, and it accepts an injected document
source, so a test supplies documents without parquet.

A coarse C interface in `capi.h` carries the driver. The client fills one
caller-owned config struct and calls one run function. The run call takes the
struct, not a file. Python owns the job file for the child process, so the C
interface carries no serializer and no parser.

`notebooks/mesh_train.py` keeps four jobs. It builds the driver config from the
notebook arguments, writes the job file, launches the child through
`tools/nanochat client`, and polls `metrics.jsonl` and `worker.log`. It does not
step the optimizer, compute the MFU, or write the summary.

The notebook builds the Python driver config directly, so no second config
class exists. The phase removes `iterate()`, `mfu()`, `format_row()`, `job()`,
and `from_job()`. `read_metrics()`, `read_summary()`, and `MeshLiveRun` stay,
because they read and display a file that C++ wrote. A file reader is not
mechanism.

The C interface carries data only. No callback from C++ into Python exists, as
section 7 states.

## 7. Non-goals

- A distributed supervised fine-tuning run. The C++ workflow has no distributed
  mode ([host-portability.md](host-portability.md)).
- A preference-optimization method such as DPO.
- A native chat template in the C++ model layer. The model stays
  tokenizer-agnostic ([post-training.md](post-training.md) section 9).
- A callback from C++ into Python. The C interface carries data only. The
  reinforcement-learning worker uses a message protocol instead.
- Autograd. The graph stays fixed ([DESIGN.md](../DESIGN.md) section 5).

## 8. Migration plan

Each phase has an owner and a gate. The owners follow `AGENTS.md` section 1.
The phases extend [post-training.md](post-training.md) section 12 with the
interface work.

| Phase | Owner | Gate |
|---|---|---|
| 1. Expose the existing step. Add `nanochat_train_step` and `nanochat_backward_accumulate`, use `backward_accumulate` in `Trainer`, and persist the optimizer step. (Done.) | Architect for `capi.h`; Python surface for `bindings/` and `api.py`; Harness for `src/train.cc` | The Python tests and `//bindings:nanochat_capi_test` |
| 2. Add the packed-row source and the reader. Add `render_conversation`. (Partly done: the renderer, the Python packer, and the mesh driver shard reader exist. The `TrainLoop` packed-row source remains.) | Harness for the loader; Python surface for the renderer | A packer test and a mask-shift test |
| 3. Add the supervised fine-tuning driver and `sft_main`. (Partly done: the mesh driver runs the SFT loop from a shard. The standalone `sft_main` remains.) | Harness for `src/`; Python surface for the facade | A short run at tier T2 against the reference loss curve |
| 4. Add `nanochat_rl_step` and the parity fixture. | Harness for the step; Oracle for the fixture and the test | `//tests:rl_parity_test` at tier T2 |
| 5. Add the reinforcement-learning worker. | Harness for the worker; Python surface for the bridge | An RL smoke run at tier T2 |
| 6. Add the mesh training driver. Add `src/mesh_train.cc` and the driver calls in `capi.h`. Use the Python driver config directly. (Done.) | Harness for `src/**`; Architect for `capi.h`; Python surface for the client; Notebooks for the launcher | The CPU driver loss sequence equals the old Python loop. The resume and the interrupt save work |

## 9. Invariants

- One step implementation. The C++ driver is the only place that advances the
  optimizer for a training run. The mesh driver is the mesh form of the same
  rule.
- One metric record. The C++ driver writes `metrics.jsonl` and `summary.json`.
  Python reads the files and displays the rows. Python does not compute a
  record.
- Valid-token normalization is the single source of truth. The
  reinforcement-learning divisor moves into C++.
- The model layer stays tokenizer-agnostic. The driver layer may use the
  tokenizer, and it already does.
- Data crosses the interface. Control flow does not, except for the
  reinforcement-learning reward, which Python computes for each rollout.
- Python calls one driver function per supervised fine-tuning run. It does not
  call one function per step.
- One mask channel. A target of `-1` marks an ignored position. No second mask
  field exists in the interface.
- The model does not replace its weights between steps. A persistent worker
  holds the model and the optimizer.

## 10. Open questions and recommendations

1. **Should the step counter survive a resume?** Yes.
   `Scheduler::LrMultiplier(step)` needs the step, and `TrainLoop::Run` starts
   at step 1 today. A resume therefore replays the warmup and breaks the curve.
   Store one more record beside the optimizer state in `Checkpointer::SaveModel`.
   On load, the driver continues to the absolute `num_iterations`. The first
   increment restores the schedule but not the data position, so a resumed run
   reads the stream from the first batch.
2. **Is `capi.h` part of the public API?** No.
   Section 1 of `AGENTS.md` freezes `capi.h`, and the architect owns it. The C
   interface is an internal seam for the Python binding, not a public API.
   Consumers use the Python package. Section 7 of the design document lists the
   public C++ headers, and it stays unchanged.
3. **Should the chat template stay in Python?** Yes, for now.
   [the evaluation document](eval.md) section 2 already gives prompt rendering
   to Python. The reference `nanochat` package owns the template, and parity is
   the goal. The C++ tokenizer has no render method
   (`include/nanochat/tokenizer.h`), so a C++ copy would be a second
   implementation that can drift. Promote `render_conversation` to a public
   Python function. Move the template to C++ only when a native command needs
   it.
4. **Should `Optimizer.step(step)` stay public?** Yes, in C++.
   The mechanism needs the step number, because the scheduler computes the
   multiplier from it. Remove the call from the primary Python path. The driver
   owns the counter for a normal run, and the Python `Optimizer` stays for a
   custom loop and for tests. One owner for the counter removes the drift
   between `TrainLoop::Run` and `Model::optimizer_step_`.
5. **Does the supervised fine-tuning driver need a thread for the reader?**
   Yes. `DataLoader` already owns a producer thread and a bounded buffer
   (`src/data.cc:274`). The packed-row loader is I/O-bound and fits the same
   design. One thread is enough, because Python did the packing.

## 11. References

- [python.md](python.md): the single Python surface.
- [post-training.md](post-training.md): supervised fine-tuning and
  reinforcement learning, and the status of record.
- [eval.md](eval.md): the evaluation seam and the "metric in Python, compute in
  C++" rule.
- [model.md](model.md): the model interface and the graphs.
- [../DESIGN.md](../DESIGN.md): section 2 for the layers, section 5 for the
  autograd policy, and section 7 for the evolution rules.
