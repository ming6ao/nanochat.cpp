# Post-training: supervised fine-tuning and reinforcement learning

`nanochat.cpp` adapts a pretrained base model to the chat distribution. It then
improves the model with reward. The reference implementation is
`scripts/chat_sft.py` and `scripts/chat_rl.py` in the PyTorch `nanochat` tree.
The Python API in [python.md](python.md) drives both stages.

This is the design of record for post-training. Keep it current: when the code
changes, change this document in the same commit. Section 10 lists the parity
entries.

The companion documents are [eval.md](eval.md) for evaluation,
[python.md](python.md) for the process seam, [parity.md](parity.md) for the
known differences, [model.md](model.md) for the model and the data formats, and
[parity.md](parity.md) for the parity plan.

## 1. Scope

Post-training has two stages:

- **SFT** (supervised fine-tuning). SFT turns a base model into a chat model on
  a fixed conversation mixture.
- **RL** (reinforcement learning). RL turns a chat model into a better problem
  solver. It uses on-policy rollouts and a reward.

Post-training does not cover these items:

- Distributed training.
- Preference optimization (DPO and similar methods).
- A native chat template in C++. The bridge owns the chat template.

The tree has a native tokenizer and a best-fit packer
([data.md](data.md), [data.md](data.md)). The
pretraining path uses them. The SFT path needs a chat renderer and a loss mask,
so the bridge owns the SFT packer.

## 2. The shared training step

SFT and RL share one mechanism. Both use a masked, weighted, mean-normalized
cross-entropy over token sequences. Three items differ: the sequence source,
the per-token weight, and the normalizer.

For `rows = batch * seq` positions, let `l_i` be the per-token loss. The loss is
zero at an ignored position. Let `w_i` be the per-token weight, and let `Z` be
the normalizer.

```
L = (1 / Z) * sum_i  w_i * l_i
```

| Stage | Sequence source | Weight `w_i` | Normalizer `Z` |
|---|---|---|---|
| Pretrain | parquet documents | 1 | valid tokens |
| SFT | conversation mixture | 1 on supervised tokens, 0 elsewhere | valid tokens |
| RL | on-policy rollouts | advantage of the sequence on sampled tokens, 0 elsewhere | valid tokens x passes x examples per rank |

The model graph stays the same in all three stages. RL does not need a second
backward pass. RL needs a per-row multiplier on the existing classifier
gradient.

### 2.1 Valid-token normalization

The reference calls `F.cross_entropy(..., ignore_index=-1)` with the default
mean reduction. That reduction divides by the non-ignored targets. The C++
runtime uses the same divisor.

`ForwardLoss` returns the mean over the valid targets. `BackwardAccumulate(scale)`
divides by the valid count. `BackwardWeighted(row_weights, scale)` applies
`row_weights[i] * scale` to row `i`, and it divides by the valid count.
Pretraining has no ignored target, so the divisor equals `rows` there and the
numbers do not change.

`ClassifierBackward` zeroes every ignored row and the padded vocabulary tail. A
row weight of 0 therefore means "do not train this position".

Implementation: `src/model.cc` (`CountValidTargets`, `ForwardLoss`,
`BackwardAccumulate`, `BackwardWeighted`); `src/masked_loss_test.cc`;
`tests/post_training_core_test.cc`.

### 2.2 Interface

```cpp
// include/nanochat/model.h (additive)
class Model {
  // Weighted policy-gradient backward. `row_weights` has batch*seq entries;
  // the classifier already zeroes ignored rows. The call accumulates into the
  // parameter gradients and does not zero them.
  virtual void BackwardWeighted(const float* row_weights, float scale) = 0;
};
```

`ClassifierParams` has `const float* row_scale = nullptr;`. A null pointer
keeps the uniform behavior. The CPU backend and the CUDA backend apply the row
scale after the softcap chain rule.

The C ABI call `nanochat_backward_weighted` and the Python method
`Model.backward_weighted` expose the same operation.

Implementation: `include/nanochat/model.h`; `include/nanochat/kernels.h`;
`src/model.cc`; `backends/cpu/kernels.cc`; `backends/cuda/kernels/classifier.cu`;
`include/nanochat/capi.h`; `bindings/nanochat_capi.cc`;
`python/nanochat_cpp/api.py`.

## 3. The chat data model

The bridge owns the chat template and the loss mask. The C++ side receives
token ids and a `-1` mask on the targets.

- **`render_conversation`** (SFT) returns `(ids, mask)`. `mask = 1` for
  assistant text, `<|assistant_end|>`, and the python tool-call tokens
  `<|python_start|> expr <|python_end|>`. `mask = 0` for BOS, user tokens, and
  the tool-output tokens `<|output_start|> result <|output_end|>`. The model
  learns to emit tool calls, but never the results. The runtime supplies the
  results at inference time.
- **`render_for_completion`** (RL and generative evaluation) drops the reference
  assistant message. It appends `<|assistant_start|>` to prime a completion.

The bridge applies the mask on the shifted targets: `targets[mask == 0] = -1`.
The same rule covers prompt tokens, tool outputs, and SFT padding.

SFT and RL share the backward pass. In both stages, the loss is a weighted sum
over the sampled or supervised positions. The mask is the only difference in
the location of the zero weights.

Implementation: `render_for_completion` and the task logic live in
`python/nanochat_cpp/chat.py` and `python/nanochat_cpp/tasks.py`.
`render_conversation` is in `python/nanochat_cpp/chat.py`. The reference
mixture and the packer are in `python/nanochat_cpp/sft_data.py`.

## 4. SFT

### 4.1 Data

The bridge downloads and mixes the reference conversation sets in
`python/nanochat_cpp/sft_data.py`. The mixture has three parts:

- SmolTalk (`HuggingFaceTB/smol-smoltalk`), the train split, about 460K rows.
- MMLU (`cais/mmlu`), the `auxiliary_train` split, three passes by default.
- GSM8K (`openai/gsm8k`), the `main` train split, four passes by default.

Validation uses the matching test splits. The bridge caps the MMLU test split at
5200 rows and the GSM8K test split at 420 rows. The validation mixture then has
the same task ratios as the training mixture.

The loader reuses `nanochat_cpp.tasks.load_hub_dataset`. That function lists the
auto-generated parquet shards through the hub API and reads them with pyarrow.
The module imports only the standard library at import time. `numpy`,
`pyarrow`, and `filelock` load lazily. The datasets cache under
`~/.cache/nanochat/task_data`.

The bridge renders each conversation with `render_conversation`. It then packs
the rows BOS-aligned with best-fit (`sft_data.iter_packed_rows`). Unlike
pretraining, SFT pads the remainder of a row with BOS instead of cropping. The
bridge masks the padding targets and the user targets to `-1`.

The native C++ loader in `src/data.cc` reads parquet documents and tokenizes
during the run. It uses best-fit packing for pretraining. The bridge owns the
SFT renderer, the mixture, and the mask. The C++ packed-row loader owns the
padding and the batching ([distributed.md](distributed.md) Part 2, section 4).
`DataLoader` does not know about chat data.

### 4.2 Loop

The SFT loop follows the base training loop with these differences:

- Load the base checkpoint. Inherit its hyperparameters: maximum sequence
  length, batch sizes, and learning rates.
- Warm-start the optimizer moments from the base checkpoint. Reset the learning
  rates to the fresh SFT values.
- Step the optimizer after `grad_accum` micro-batches.
- Set the learning rate by dataset progress: warmup, constant, and warmdown.
- Warm the Muon momentum from 0.85 to 0.95 over 300 steps.
- Evaluate the validation bits-per-byte and ChatCORE at intervals.
- Save to `chatsft_checkpoints/<tag>` with the optimizer state.

The bridge starts SFT as one job. The loop does not depend on fresh model
output.

Implementation: `src/sft_command.cc` runs the packed-row loop, the
`LrMultiplierForProgress` schedule, the warm start and the `--resume` path, the
gradient accumulation, the bits-per-byte evaluation on `--val-rows`, and the
checkpoint write with the `Recipe` block of D7. `TrainLoop` in `src/train.cc`
still provides the step loop for pretraining, which keeps the step schedule and
`Scheduler::LrMultiplier`.

## 5. RL

### 5.1 Rollout

For each training example, per step:

1. `render_for_completion` gives the prompt tokens.
2. `GenerateBatch` samples `num_samples` completions. It uses the temperature,
   the top-k value, `max_new_tokens`, and a per-sampling-step seed. It returns
   token ids and masks.
3. The bridge decodes the samples, computes the reward, and forms
   `advantage = reward - mean(reward)` across the samples of that example.
   Phase 6 of section 12 replaces this baseline with the group-relative form.
4. The bridge masks the prompt positions and the forced tool-output positions.
   The sampled positions stay. This is the same `mask = 0` rule as SFT.

The calculator tool is the one structural difference from SFT. The reference
engine detects a python block, evaluates the expression, and forces the output
tokens. That step needs decoding inside the generation loop. A
tokenizer-agnostic C++ engine cannot do it. Two options exist:

- **MVP:** `GenerateBatch` does not force tool outputs. The model samples its own
  `<|python_start|>` blocks. The bridge reads the reward from the final `####`
  answer.
- **Parity:** a streaming `chat_engine` protocol. The bridge receives one token
  column at a time, decodes python blocks, and sends forced tokens back. An
  interactive `chat_cli` needs the same protocol.

### 5.2 Objective

For a micro-batch of one example's rollouts:

```
logp_i = -l_i
pg_obj = sum_i advantage[seq(i)] * logp_i
L = -pg_obj / (num_valid * num_passes * examples_per_rank)
```

The objective has no KL term, because there is no reference model. It has no
PPO ratio or clip, because the data is on-policy. It uses DAPO-style token-level
normalization. The code counts `num_valid` per pass. The passes accumulate with
`BackwardWeighted`. The optimizer steps once per example group.

### 5.3 Optimizer and schedule

Use the standard nanochat grouping. Apply `init_lr_frac = 0.05` to every group.
Use a linear rampdown `1 - it / num_steps`. Use weight decay 0.

Hold the Muon momentum at 0.95, because the RL script never updates it. The C++
scheduler expresses this with `warmdown_ratio = 1`, `final_lr_frac = 0`, and a
flat Muon momentum. Checkpoints go to `chatrl_checkpoints/<tag>`.

### 5.4 Evaluation

Report GSM8K pass@k on the test split every `eval_every` steps. Use the
generative path from [eval.md](eval.md).

### 5.5 A standalone run on a card

The pinned settings for one RL stage, run on its own:

| Setting | Value | Why |
|---|---|---|
| Card | `T4`, one, `sm_75` | The deployed image, and the Kaggle architecture |
| Precision | fp32 | The T4 default of [precision.md](precision.md) |
| Model | 16 layers, hidden 1024, 8 heads, 4 key/value heads, seq 4096 | The 252M shape of `pretrain.ipynb` |
| Warm start | the `chat` stage, else `sft` | RL starts from a chat model |
| Task | `gsm8k` | The one task with a `reward` method |
| Smoke shape | 8 prompts, 4 samples, 3 steps, 256 tokens | A few seconds of card time |
| First real run | 8 prompts, 4 samples, 100 steps | One card, one container |
| Checkpoint | `chatrl.nchkpt01` every 25 steps | A pull after a crash keeps the work |

`nc.rl.run` runs one stage on any host under `tools/nanochat train`. In a Modal
container the run holds one container, and `fetch_stage` in `tools/modal/app.py`
reads the warm start from the `nanochat-weights` Volume. So a warm start is one
download, and it never needs an image build.

The smoke shape holds 32 rollouts per step. At 256 completion tokens each, the
rollout reply is one line of about 15000 integers, which is about 75 KiB. The
pipe carries that. A larger batch needs the reply split across messages, which
is part of Phase 8 of section 12.

The container commits the `nanochat-runs` Volume every ten seconds, and a
client sees a record only after a commit. A crash between two commits loses the
last seconds of the log, so keep the checkpoint interval small.

### 5.6 The warm start and the two checkpoint paths

Two facts hold about a warm start. The first is the design decision D11. The
second landed with B3: the load path and the save path are separate arguments.

- **A warm start carries the optimizer state** (design decision D11). The
  weights, the AdamW first and second moments, and the Muon buffers travel
  together, as section 7 describes, and the driver reads the recipe from the
  checkpoint it loads. Clearing the moments would change the stage for every
  caller to simplify one test, and it would hide the one thing a warm start is
  for.
- **`warm_start` names the load and `checkpoint` names the save**
  (`nc.rl.run(warm_start=..., checkpoint=...)`). `warm_start` reaches the
  worker as `--checkpoint`, and `checkpoint` is the save target and nothing
  else. The legacy rule that derived a load path from `checkpoint` is removed,
  because a save path must never become a load path: it made an RL stage
  overwrite the checkpoint it warm started from (blocker B3 of
  `docs/post-training.md`). A stage therefore needs both arguments whenever it
  starts from its parent, and the chain of `docs/post-training.md` Part 1, section 9
  always passes them.

A graded reward and the per-row seed mix of B4 are what make the advantage
vector non-zero beside that: the four samples of a group draw from four streams,
so the group is not four copies of one completion.

## 6. Generation for rollouts

RL rollouts and generative evaluation share `GenerateBatch`. The primitive
prefills the prompt once, clones the KV cache per sample, and decodes the rows
in lockstep. Each row stops on its own terminal token. The primitive returns
ids and masks.

The graph, the cache clone, and the sampling parameters live in
[model.md](model.md). The C++ engine does not depend on the tokenizer. The
caller passes the terminal token ids.

Implementation: `include/nanochat/model.h` (`GenerateParams`,
`GeneratedSequence`, `GenerateBatch`); `src/generate.cc`;
`src/generate_test.cc`; `tests/generate_parity_test.cc`.

## 7. Checkpoints and optimizer state

Two containers exist:

- `NCMDL001` is a legacy flat parameter file. The project no longer writes it.
  `Checkpointer::LoadModel` still reads it (`src/train.cc`).
- The checkpoint container is the self-describing file `Checkpointer` uses
  (`src/data.cc`). Writers write version 1, which carries the run metadata
  block; readers accept version 0 and version 1
  (`docs/checkpoint-format.md`). The reference converter in
  `python/nanochat_cpp/checkpoint.py` writes the same format.

Both containers hold parameters. The `Checkpoint` class also carries an
optimizer-state section. That section holds the AdamW first and second moments.
It also holds the Muon momentum and second-moment buffers.

The record names are stable, so a later load can match them to the optimizer
groups.
`Checkpointer::SaveModel` and `Checkpointer::LoadModel` have `Optimizer`
overloads. A parameter-only file stays loadable.

The checkpoint container stores fp32 or bf16 records. The record dtype is a
storage dtype, separate from the build compute dtype. An fp32 container is the canonical
artifact for a resume. A bf16 container is a compact artifact.

Pretrain checkpoints load directly. Reference base and SFT checkpoints are torch
`.pt` files. `tools/convert_checkpoint.py` remaps them. The converter also
reads the reference optimizer state.

Implementation: `include/nanochat/data.h`; `src/optim_state.h`; `src/optim.cc`;
`src/train.h`; `src/optimizer_state_test.cc`;
`python/nanochat_cpp/checkpoint.py`; `tools/convert_checkpoint.py`.

## 8. Orchestration and the sandbox

- **SFT** has two entries. The scripted entry is the bridge. The bridge
  materializes the packed shard, and then it launches the C++ training binary
  (`nanochat_cli sft`) under `tools/nanochat train`. The notebook entry is the
  in-process one-device `MeshRuntime` loop in `notebooks/posttrain.ipynb`.
  Both entries use the same loss, gradient, and optimizer code.
- **RL** is on-policy. Generation and training interleave, and the optimizer
  moments persist. The target is a persistent worker. The worker holds the
  model and the optimizer. It exchanges rollout and advantage messages with the
  bridge over pipes.
- For correctness, the same step logic exists as a file-driven `rl_step` binary.
  That binary consumes a fixture. It is the RL analog of the training-parity
  harness, and it is the gate for the objective.
- Every run goes through `tools/nanochat`, so the GPU queue and the resource
  sandbox apply ([sandbox.md](sandbox.md)).

Implementation: `src/sft_command.cc` with `python/nanochat_cpp/sft.py`, and
`src/rl_worker_command.cc` with `python/nanochat_cpp/rl.py` (section 11).

## 9. Invariants and non-goals

- One fixed graph; no autograd. RL reuses the classifier backward.
- Valid-token normalization is the single source of truth for the mean.
  Pretraining, SFT, and RL share it.
- The C++ model engine does not depend on the tokenizer. The bridge owns the
  chat template and the loss mask.
- Single process; no distributed data parallel ([parity.md](parity.md) D6).
- No native chat template in C++.
- No overlap of rollout and update, and no policy versioning.
- No clipped policy optimization with a reference model, and no Kullback
  Leibler term.
- No preference optimization.

## 10. Parity

`parity.md` is the canonical list. The table below lists the entries that
post-training uses. Add a new entry to `parity.md` when you record a new
difference.

| ID | Status | Difference |
|---|---|---|
| E1 | `open` | Sampled modes use the C++ RNG, not torch; greedy is bit-identical |
| E3 | `equivalent` | Reference checkpoints converted torch to the checkpoint container |
| E4 | `equivalent` | CORE and chat scoring use the C++ forward with the reference task logic |
| E5 | `out-of-scope` | No distributed evaluation |
| P1 | `open` | The calculator tool is absent in the MVP; the model samples its own python blocks |
| P2 | `open` | A file-driven `rl_step` makes the rollouts lag one optimizer step |
| P3 | `equivalent` | SFT starts from a converted reference base checkpoint |
| P4 | `closed` | SFT data parity (interim name): the renderer, the mixture, and the packer match the reference |
| P5 | `equivalent` | SFT arithmetic parity (interim name): the loss, the gradient, and the optimizer trajectory match on the CPU backend |
| P6 | `open` | SFT loop parity: the driver gate has not run; the progress schedule and the momentum it computes are pinned at T0 |
| P7 | `open` | SFT validation parity: the packed bits-per-byte is pinned; ChatCORE is not measured |

## Part 1. Reinforcement learning in a notebook

_Absorbed from the former `rl-notebook.md`. Its section numbers are unchanged,
so a reference to a section of that document resolves to the same number here._

The hand-rolled loop, the one-call facade, the persistent worker, and the
three-stage chain of section 9 exist. The multi-prompt C ABI binding and the
tool-forcing driver are not implemented.

This part shows how to run reinforcement learning (RL) in a notebook. Section 5
of this page and `docs/distributed.md` Part 1, section 6 hold the same design
from the other side. Section 9 gives the one-job chain that runs this call last,
after base training and SFT.

### 1. Purpose

A notebook user must write the data policy and the reward policy. The user must
not write the training mechanism. One call must start an RL run. The C++ driver
must hold the model and the optimizer.

### 2. Today: compose it by hand

A user can write an RL loop today with the public API. The loop is long.

```python
import nanochat_cpp as nc
from nanochat_cpp import chat

model = nc.Model(nc.Config(...))
opt = nc.Optimizer(model, num_iterations=steps)
task = chat.build_task("gsm8k")
stop = tokenizer.encode_special("<|assistant_end|>")

for step in range(1, steps + 1):
    ids, targets, weights = [], [], []
    for conv in prompts:
        prompt = render_for_completion(conv)
        rows = model.generate(prompt, num_samples=4, temperature=1.0,
                              stop_ids=[stop])
        rew = [task.reward(conv, tokenizer.decode(r.tokens)) for r in rows]
        mean = sum(rew) / len(rew)
        for row, r in zip(rows, rew):
            # Build tokens = prompt + row tokens. Pad to seq_len.
            # Set targets = -1 on the prompt. Set weight = r - mean.
            ...
    model.forward_loss(ids, targets)
    model.zero_grad()
    model.backward_weighted(weights, scale=1.0 / valid_tokens)
    opt.step(step)
```

The loop has six problems.

1. The user writes the mechanism and duplicates `TrainLoop`.
2. `render_for_completion` is private (`python/nanochat_cpp/chat.py`).
3. Only GSM8K has a `reward` method (`python/nanochat_cpp/tasks.py`).
4. `generate` takes one prompt and many samples. It does not do multi-prompt
   rollout.
5. The loop crosses the application binary interface (ABI) every step.
6. The divisor lives in Python, not C++. This breaks the single-source rule.

### 3. The target: one call

The plan gives the notebook one facade call. The call follows the proposed
supervised fine-tuning call in `docs/distributed.md` Part 2, section 6.5.

```python
nc.rl.run(
    task="gsm8k",
    prompts=conversations,
    num_samples=4,
    steps=1000,
    config=shape,                       # the model shape, as model flags
    warm_start="chatsft.nchkpt01",      # the load path
    checkpoint="chatrl.nchkpt01",       # the save path
    checkpoint_interval=100,
)
```

The user writes prompts and selects a task. The user does not write a loop.
Python still owns the reward, so Python stays in the control loop. The mechanism
moves into a persistent C++ worker.

Three arguments carry three separate facts, and the second is why the call is
safe to make at all:

- `warm_start` is the container the worker loads the weights and the optimizer
  moments from. It reaches the worker as `--checkpoint`.
- `checkpoint` is the container the worker saves to, and nothing else. A save
  path never becomes a load path, so a stage that warm starts from the SFT
  checkpoint and saves to its own file keeps its parent intact (blocker B3 of
  `docs/post-training.md`). The legacy rule that derived the load path from
  `checkpoint` is removed, and `docs/python.md` section 3.4 records that as a
  breaking change.
- `config` is the model shape. It reaches the worker through
  `nanochat_cpp.api.model_flags`, the one model flag list on the Python side, so
  the worker rebuilds the shape of the run it warm starts from and the two
  stages cannot drift (B1 and B2 of the same plan).

### 4. The worker

A long-lived process holds the model and the optimizer. It speaks a pipe
protocol with the Python bridge. One step has seven parts.

1. Python renders a batch of prompts.
2. Python sends a `rollout` message.
3. The worker generates rows and returns them.
4. Python scores the rows and computes the advantage.
5. Python sends an `advantage` message with weights.
6. The worker runs one RL step. The step is `ForwardLoss`, then
   `BackwardWeighted`, then `Optimizer::Step`.
7. The worker returns the loss and the gradient norm.

A `rl_step` binary shares the step logic. It reads a fixture. It is the
correctness gate. The project trusts the worker only after
`//tests:rl_parity_test` passes at tier T2.

### 6. Notebook concerns

- **Kaggle.** The notebook sets `NANOCHAT_SANDBOX_BACKEND=none`
  (`docs/python.md` section 3.1). In this mode the facade must start the worker
  directly. It must not call `tools/nanochat`, because Kaggle has no systemd
  user manager.
- **Workstation.** The facade calls `tools/nanochat` with a new profile and
  command, for example `tools/nanochat train -- rl_worker ...`. The GPU queue
  and the sandbox apply.
- **Checkpoints.** The `rl` path already resolves to `chatrl_checkpoints`
  (`python/nanochat_cpp/checkpoint.py`). A warm start and the optimizer state
  use the checkpoint container.
- **Sizes.** A notebook GPU is small. Keep the rollout batch and `num_samples`
  small. The plan caps the shapes at tier T1 and tier T2.

### 7. Convenience levels

| Level | What the user writes | Available |
|---|---|---|
| Hand-rolled | The full loop | Yes, today |
| Facade | `nc.rl.run(...)`, no loop | Delivered |
| Batched and fast | The same call, many prompts | Delivered in C++; the C ABI shim is not |
| Tool-forced parity | The same call | The mechanism exists; not yet wired |

### 9. The chain: one job file, three stages

`nc.rl.run` is the last of three stages, and one job file drives all three on
any host. The chain is `python/nanochat_cpp/pipeline.py`, and it holds no
training logic, no model flag list of its own, and no host name (design decision
D1 of `docs/post-training.md`). Each stage calls one facade and reads the
checkpoint of the stage before it:

| Stage | Facade |
|---|---|
| base | `runtime.MeshTrainConfig` |
| SFT | `sft.pack`, then `sft.run` |
| RL | `rl.run` |

```python
# base: the mesh driver at world_size = 1, over the fixture corpus.
runtime.MeshTrainConfig(config(job), steps=job.base_steps, ...)
# SFT: one packed shard, the base checkpoint as the load path.
sft.run(shard, ckpt, config=config(job), warm_start=base.checkpoint,
        num_iterations=job.sft_steps)
# RL: the SFT checkpoint as the load path, its own file as the save path.
rl.run(reward, prompts, num_samples=job.num_samples, steps=job.rl_steps,
       config=config(job), warm_start=sft.checkpoint, checkpoint=rl_ckpt,
       max_tokens=job.max_tokens)
```

`config(job)` is `pipeline.PROFILES[job.profile]`, so the shape has one home;
`PROFILES` holds the `demo` profile of decision D5, and every stage receives it
as an `nanochat_cpp.api.Config`, which `api.model_flags` turns into the flags
the stage binaries parse. The load path and the save path stay separate at every
stage (B3), so each checkpoint survives the stage that reads it.

A job is plain data with a JSON round trip: `run_id`, `profile`, `base_steps`,
`sft_steps`, `rl_steps`, `num_prompts`, `num_samples`, `max_tokens`, `seed`,
`out_root`, `corpus`, `deliver`, `temperature`, and `top_k`. `out_root` is empty
in a committed job, because the host that owns a run root fills it: the Colab
notebook sets `NANOCHAT_RUNS_DIR`, and the Modal app rewrites it to
`/runs/<run_id>`.

The reward is a class with a `reward` method, which `rl.run` accepts directly,
so no Hub dataset loads. It is graded: 1.0 for the right number, 0.5 for the
`####` marker, and otherwise the character overlap with the reference. A graded
reward keeps the advantages of a group from collapsing into zeros, and the seed
mix of B4 makes the four samples of a group differ in the first place, so the
advantage vector is not zero either.

`pipeline.run(job)` runs the three stages and returns a `Report`, and
`python -m nanochat_cpp.pipeline <job>` is the same call from a command line.
`deliver()` writes the two artifacts, and `//python:pipeline_test` asserts A1 to
A9 of `docs/post-training.md` Part 2, section 7 over the `demo` profile, so the
assertions and not the loss values are the contract.

#### 9.1 The delivered artifacts

`deliver()` writes the contract of decision D6. Five paths name it:

```
<run_root>/<run_id>/base/checkpoint.ckpt
<run_root>/<run_id>/sft/checkpoint.ckpt
<run_root>/<run_id>/rl/checkpoint.ckpt
<run_root>/<run_id>/deliver/chatrl.ckpt
<run_root>/<run_id>/deliver/pipeline_report.json
```

`deliver/chatrl.ckpt` is the RL checkpoint copied byte for byte, and the report
is the review artifact: the stage order and the step counts, the loss traces and
the mean rewards, the checkpoint digests and the metadata blocks, the seconds
and the seeds, the model shape and the tokenizer digest, and the host block of
`backend`, `arch`, `precision`, and `revision`. A reviewer reads the report and
the checkpoint, not the run.

The copy is verified against the digest the stage recorded. One field is empty
by construction: `nc.sft.run` returns an exit status and not a loss trace, so
the SFT stage says so in `extra` and the report carries an empty trace for it.
The RL stage reads its rewards through `on_step`, which receives the per-rollout
rewards, so it records the mean reward and the count of groups whose advantages
are all zero.

#### 9.2 The two entry forms

The entry is always the tool, never a bare module (decision D10), because both
stage executables call `RequireSandboxOrDie`, and a notebook's
`NANOCHAT_SANDBOX_BACKEND=none` satisfies the library only. Two forms exist:

```
# The host form: one `train` scope, or the `none` backend of a container.
tools/nanochat train -- python -m nanochat_cpp.pipeline <job>

# The container form: the owner routes the work through the GPU queue.
tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job>
```

The first form is the one the Colab notebook and a workstation use; the second
is the one the Modal `pipeline` function uses. `docs/host-portability.md`
section 9 gives the three hosts and what each one records.

---

## Part 2. One chain on three hosts

_Absorbed from the former `pipeline-plan.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

### 4. Decisions

| # | Decision |
|---|---|
| D1 | One orchestration module, `python/nanochat_cpp/pipeline.py`. Each stage calls an existing facade. It holds no training logic, no model flag list of its own, and no host name. |
| D2 | Every host builds the one stage binary, through the resolver of B6. A tree build is the only route until the dependency plan publishes a binary wheel. |
| D3 | The pinned triple is T4, one card, `sm_75`, fp32, and it is the first of two legs: a second leg runs the same job on one H100 card, `sm_90`, at bf16, in its own image and under its own run id. A local card uses its own architecture through `NANOCHAT_CUDA_ARCH`. |
| D4 | The corpus is the committed fixture. Base reads `src/parquet/testdata/text.parquet` with `tests/data/loader_tokenizer.nctoken`. SFT reads one shard that `nc.sft.pack` writes, and RL reads four of those conversations. No stage reads the Hub. |
| D5 | The demo shape has 2 layers, hidden 64, 4 heads, 2 key/value heads, sequence 64, and vocabulary 487 padded to 512. It costs about 4 s of CPU for the whole chain. |
| D6 | `deliver/` is the artifact contract, below. |
| D7 | No frozen header changes. B2 lands in `src/cli.h`, B4 stays inside `src/generate.cc`, and B1, B3, B5, and B6 are Python. |
| D8 | Colab builds from source, not from the wheel. A wheel path lands when a binary wheel exists, and B6 makes that a one-line change. |
| D9 | One owner per file, below. |
| D10 | The entry is always wrapped by `tools/nanochat`, in two forms. See below. |
| D11 | A6 compares two runs, and no warm start changes. The optimizer state stays part of what a warm start carries, as `docs/post-training.md` §7 says. Clearing the moments would change the stage for every caller to simplify one test. |

The artifact contract of D6:

```
<run_root>/<run_id>/base/checkpoint.ckpt
<run_root>/<run_id>/sft/checkpoint.ckpt
<run_root>/<run_id>/rl/checkpoint.ckpt
<run_root>/<run_id>/deliver/chatrl.ckpt
<run_root>/<run_id>/deliver/pipeline_report.json
```

The eval stage of section 5 writes no checkpoint and no directory: its whole
record is one `stage` entry inside `deliver/pipeline_report.json`.

The report is the review artifact. It holds these items:

- the stage order: base, SFT, RL, then eval, with the step counts;
- the loss traces and the mean rewards;
- the checkpoint digests and the metadata blocks;
- the seconds and the seeds;
- the model shape and the tokenizer digest;
- the source revision.

D10 has two entry forms. On a host:

```
tools/nanochat train -- python -m nanochat_cpp.pipeline <job>
```

In a container:

```
tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job>
```

Two child executables check the sentinel with `RequireSandboxOrDie`, and no host
may satisfy that check with the override.

The owners of D9:

| File | Owner |
|---|---|
| `python/nanochat_cpp/pipeline.py`, `python/tests/pipeline_test.py` | Python surface |
| `python/nanochat_cpp/sft.py`, `rl.py`, `api.py`, `toolchain.py`, `_build.py` | Python surface |
| `src/cli.h` | Harness |
| `src/generate.cc` | Workflow |
| `tools/modal/app.py`, `tools/modal/modal_app_test.py` | Python surface |
| `notebooks/colab_pipeline.ipynb`, `notebooks/colab_setup.py` | Notebooks |
| `docs/**`, `AGENTS.md`, and the root `BUILD.bazel` files | Architect |

A per-directory `BUILD.bazel` belongs to that directory's owner (`AGENTS.md`
§6), so `python/BUILD.bazel` is the Python surface's and `src/BUILD.bazel` is
the owning workstream's.

### 5. The pipeline module

The module as it stands after the extension, whose design is
[post-training.md](post-training.md) Part 3, sections 5 and 6:

```python
# python/nanochat_cpp/pipeline.py

PROFILES = {
    "demo": Config(num_layers=2, num_heads=4, num_kv_heads=2,
                   hidden_dim=64, seq_len=64, vocab_size=487,
                   padded_vocab_size=512, window_pattern="L"),
    "standard": Config(num_layers=8, num_heads=8, num_kv_heads=4,
                       hidden_dim=512, seq_len=64, vocab_size=487,
                       padded_vocab_size=512, window_pattern="L"),
}
DEFAULT_PROFILE = "standard"

@dataclass(frozen=True)
class Job:                 # JSON round trip; one file for all three hosts
    run_id: str = "demo"
    profile: str = DEFAULT_PROFILE
    batch: int = 1               # rows per device; the corpus decides it
    grad_accum: int = 1
    devices: int = 1
    base_steps: int = 200
    sft_steps: int = 200
    rl_steps: int = 8
    num_prompts: int = 4
    num_samples: int = 4
    max_tokens: int = 32
    eval_batches: int = 1        # one batch; the corpus allows no more
    eval_samples: int = 2
    seed: int = 42
    out_root: str = ""
    corpus: str = "fixture"      # "fixture" or "files"
    corpus_path: str = ""        # a parquet file or a directory of them
    tokenizer_path: str = ""     # the NCTOKEN1 artifact of that corpus
    deliver: bool = True
    temperature: float = 1.5
    top_k: int = 0

@dataclass(frozen=True)
class Stage:               # one stage's facts
    name: str
    steps: int
    losses: tuple[float, ...]
    checkpoint: str
    sha256: str
    metadata: dict
    seconds: float
    extra: dict = field(default_factory=dict)   # means rewards, dropped groups

@dataclass(frozen=True)
class Report:
    version: int               # the report schema, 2 today
    job: Job
    stages: tuple[Stage, ...]
    delivered: str
    report_path: str
    total_seconds: float
    host: dict                 # backend, arch, precision, source revision

def stage_base(job) -> Stage
def stage_sft(job, base: Stage) -> Stage
def stage_rl(job, sft: Stage) -> Stage
def stage_eval(job, base: Stage, rl: Stage) -> Stage
def deliver(job, stages, *, out_root) -> tuple[str, str]
def run(job, *, out_root=None) -> Report
def main(argv=None) -> int        # python -m nanochat_cpp.pipeline job.json
```

`demo` is the shape of D5 and it stays, because the CPU test loop runs it in
about five seconds. `standard` is the shape a throughput number is read at, and
`Job()` defaults to it: 8 layers, hidden 512, 8 heads, 4 key/value heads,
sequence 64, the same vocabulary 487 padded to 512, and the `L` window pattern.
The sequence length and the base batch stay where they are, because the
committed corpus fills exactly one 65-token row per epoch
([post-training.md](post-training.md) P1, P2 and section 5).

The schedule grows by about a factor of two: 200 base steps, 200 SFT steps, 8
RL steps, one evaluator batch, and two sampled completions. One step of the
`standard` shape does about ten times the device work of one `demo` step, so a
local CPU chain still costs tens of seconds.

Each stage function is short and calls one facade:

| Stage | Call |
|---|---|
| Base | `runtime.MeshTrainConfig(config(job), steps=job.base_steps, ...)` with `train_parquet`, `tokenizer`, `out_dir`, `checkpoint_path` |
| SFT | `sft.pack(...)` then `sft.run(shard, ckpt, config=config(job), warm_start=base.checkpoint, num_iterations=job.sft_steps)` |
| RL | `rl.run(reward, prompts, num_samples=..., steps=..., config=config(job), warm_start=sft.checkpoint, checkpoint=rl_ckpt, max_tokens=...)` |
| Eval | `api.Model.load(checkpoint)` then `api.evaluate(model, corpus, tokenizer, steps=job.eval_batches, batch=1, seed=job.seed)`, once for the base checkpoint and once for the RL one, then `api.Model.generate` for `job.eval_samples` completions |

`config(job)` is `PROFILES[job.profile]`, so the shape has one home.

The eval stage is the one stage with no executable. It runs in the chain's own
process through the in-process API, so it adds no third stage binary and no new
build target (P4). It writes no checkpoint: its record names the RL file, so
`deliver()` still copies `chatrl.ckpt`. The one limit worth knowing is the
batch: `eval_batches` is 1 and it is not a tunable number in this plan. The
corpus holds 119 tokens, so one epoch fills one 65-token row, and a batch of two
rows cannot form. An evaluator that is asked for two rows reports no bytes and
returns infinity, which is why the module passes `batch=1` by hand: the default
of `api.evaluate` is 8. A tenfold larger corpus would allow more batches, and
that corpus needs a producer, which is a separate work item
([post-training.md](post-training.md) Part 3, section 6).

The reward is a class with a `reward` method, which `_normalize_task` accepts, so
no Hub dataset loads. It is graded: 1.0 for the right number, 0.5 for the `####`
marker, and otherwise the character overlap with the reference. A graded reward
keeps the advantages from collapsing (Phase 6 of `post-training.md` §12).

The corpus builder copies the committed parquet and tokenizer into
`<run_root>/<run_id>/corpus/`, and writes
`<NANOCHAT_BASE_DIR>/tokenizer/tokenizer.nctoken`, because `data.nctoken1_path()`
reads that exact path. The pipeline never reads `NANOCHAT_BASE_DIR` for output,
and the report names the root it used.

### 6. The three hosts

Every host runs the same four stages of the same job file. The job's counts are
its own: the committed example job of section 5 asks for 200 base steps, 200 SFT
steps, 8 RL steps, and one evaluator batch, and a test job asks for the `demo`
shape and whatever schedule the test pins. The measured numbers quoted in this
section are the 2026-10-31 runs of the `demo` shape at 100, 100, and 5 steps,
and they stay as recorded.

| Host | Backend | Arch | Sandbox | Build | Gate |
|---|---|---|---|---|---|
| Local CPU | `cpu` | none | `train` scope | Bazel | `//python:pipeline_test`, default loop |
| Local GPU | `cuda` | card | `t1-gpu` queue job | Bazel | `//python:pipeline_gpu_test`, `gpu` tag |
| Colab | `cuda` | `sm_75` | `none` | bootstrap + Bazel | manual run; owed, and recorded as owed in `docs/host-portability.md` section 9.2 |
| Modal, T4 leg | `cuda` | `sm_75` | `none` | image build, fp32 | `modal run ...::pipeline_demo`; recorded on 2026-10-31, with the host block of `docs/host-portability.md` section 9.3 |
| Modal, H100 leg | `cuda` | `sm_90` | `none` | image build, bf16 | the same entry point; G1 to G3 pass, with the host block of `docs/host-portability.md` section 9.3 |
| Modal CPU | `cpu` | none | `none` | image build, CPU config | verification only |

The two Modal rows are one run of one job: the T4 leg first, then the H100 leg.

#### 6.1 Local CPU and local GPU

```
tools/nanochat build //src:nanochat_cli
NANOCHAT_CPP_BACKEND=cpu tools/nanochat train -- \
    python -m nanochat_cpp.pipeline /tmp/job.json
```

One `train` scope, at 6 GB and 800 % CPU, covers all four stages. B5 keeps the
stage children inside that scope: the entry point already skips its own scope
under a sentinel, so a facade runs its child directly instead of re-entering the
entry point, and nothing asks for a second profile (section 3, B5). The eval
stage starts no child at all, so the scope covers whatever it reads.

For the local GPU, run the same command with `NANOCHAT_CPP_BACKEND=cuda` under
`tools/nanochat gpu --profile t1-gpu -- ...`. `tools/nanochat doctor` reports
the name, driver, memory, use, and utilisation, and no compute capability, so the
arch comes from `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.

Acceptance: the chain finishes, `deliver/pipeline_report.json` exists, the four
stages record the counts the job asked for (the example job: `base`@200,
`sft`@200, `rl`@8, and `eval` on its one batch), and `host.backend` reports the
backend that ran.

#### 6.2 Colab with a T4

New `notebooks/colab_pipeline.ipynb`, four cells:

1. Setup: clone the tree, install Bazelisk with `tools/kaggle/bootstrap.sh`, then
   `notebooks/colab_setup.setup(backend="cuda", arch=probe(),
   prefer_installed=False)`, then `_build.build_stages()`. Two calls, not one:
   `setup` resolves the library only.
2. Environment: `NANOCHAT_BASE_DIR=/content/cache`,
   `NANOCHAT_RUNS_DIR=/content/runs`, `NANOCHAT_CPP_BACKEND=cuda`,
   `NANOCHAT_CUDA_ARCH=probe()`, `NANOCHAT_SANDBOX_BACKEND=none`.
3. Run: `tools/nanochat train -- python -m nanochat_cpp.pipeline
   /content/runs/job.json`.
4. Collect: print the report, and offer one download of `deliver/`.

Cell 3 must wrap the module. The backend value satisfies the library check in
`nanochat_init` only: every stage subcommand calls `RequireSandboxOrDie`, which
reads `NANOCHAT_SANDBOX` and not the backend value. So the entry is the tool,
which sets the sentinel to `off` on a host without systemd. Do not set the
sentinel in the notebook: `docs/sandbox.md` and `docs/host-portability.md`
forbid it, and the image has the same boundary.

`probe()` maps the output of `nvidia-smi --query-gpu=compute_cap` to
`sm_<major><minor>`. Free Colab gives a T4, so `sm_75`. A paid session can give
a V100, `sm_70`, or an A100, `sm_80`, and an `sm_75` build then fails inside
`nanochat_init` with a message about the architecture, not about the job.

`prefer_installed=False` is required, not optional. `setup` prefers an installed
distribution by default (`notebooks/kaggle_setup.py`), so a session that already
carries the wheel returns the install root as `repo`. That root has no
`tools/nanochat` and no stage binaries, so it contradicts D8 and breaks D10.

`notebooks/colab_setup.py` is additive: a thin module that calls
`kaggle_setup.setup(...)` with the `/content` defaults, then `build_stages()`.
`kaggle_setup.py` keeps its name, because `notebooks/session.py` and five
documents name it. A rename is a separate, mechanical change, and it is not part
of this plan.

Acceptance: `/content/runs/<run_id>/deliver/pipeline_report.json` exists, it
names four stages with the counts the job asked for and one evaluator batch, and
`host.backend` is `cuda` with `host.arch` equal to the probed value.

#### 6.3 Modal: two legs, a T4 at fp32 and an H100 at bf16

`tools/modal/app.py` gains four functions over two images, and the entry point
takes both legs of one job.

| Leg | Card | Arch | Precision | Build command |
|---|---|---|---|---|
| `t4` | `T4` | `sm_75` | `fp32` | `--config=cuda --define=precision=fp32 --@rules_cuda//cuda:archs=sm_75` |
| `h100` | `H100` | `sm_90` | `bf16` | `--config=cuda --define=precision=bf16 --@rules_cuda//cuda:archs=sm_90` |

A frozen `Leg` record holds the name, the card type, the card count, the
architecture, the precision, and the build configuration. `image_for(leg)`
returns one image per leg, and the stage executables, the precision, the
architecture, and the build call all read the leg. One library per image is
what keeps the architecture and the precision honest: one Bazel output base
holds one build of `libnanochat_shared.so` (section 6.4 makes the same argument
for the CPU leg).

The four functions are `train` and `pipeline` on the T4 leg, and `train_h100`
and `pipeline_h100` on the H100 leg. The first two keep their names, because
`notebooks/target.py` looks the name `train` up through
`modal.Function.from_name`. `notebooks/target.py` also gains a `leg` argument on
`ModalTarget`, so a notebook selects one leg.

Each leg mirrors what the single leg did:

- `pipeline` and `pipeline_h100` are thin faces of one `_pipeline_on(leg, ...)`,
  so the two legs cannot drift. The leg selects the image, the card, the
  precision, the architecture, and the roofline device.
- Both rewrite `out_root` to `/runs/<run_id>`, because the app owns the Volume
  path.
- Both start the work as `tools/nanochat client --profile train -- python -m
  nanochat_cpp.pipeline <job.json>`, which is the container entry of D10.
- Both keep `max_containers=1`, `_check_cards(leg, ...)`, the commit keeper, and
  the `worker.log`, and both reuse `_publish` for the Hub push of the final
  stage, so the chain gains no second uploader.
- Both set the same `cpu=` value. Modal sizes the CPU by memory when `cpu=` is
  absent, so two legs with different memory would get different host budgets,
  and gate G2 compares the two base step times: the shared `cpu=` is what makes
  that comparison meaningful.
- Both legs link the native ring, which is the one collective, and the chain
  runs `world_size=1`. The H100 leg must not use the `h100` named config, which
  is the simulator profile. The `hopper` config of
  [build.md](build.md) is the same build as the flag triple of the table above,
  and the build helper still emits the triple, never a named config.
- `pipeline_demo` runs the T4 leg first, under the run id `<run_id>-t4`, then
  the H100 leg under `<run_id>-h100`, so both reports stay on the
  `nanochat-runs` Volume and the cheap artifact survives a failure of the
  expensive leg. It prints one document with three keys -- `legs`,
  `comparison`, and `gates` -- and exits non-zero when a gate fails.
- The image build of a leg names `//src:nanochat_cli` beside the library and the
  two stage executables, so gate G3 runs the tool inside the H100 container and
  starts no second Bazel build there.

Gates G1 to G3 are section 7.1's subject. Their first two-leg run is row 9 of
[post-training.md](post-training.md), and it ran on 2026-10-31:
the T4 leg measured 128.74 ms per base step and the H100 leg 124.39 ms, and all
three gates passed. The chain reports `startup_check: not_run`, because the mesh
trainer does not call the hardware correctness gate; the fp8 path on Hopper needs
its own row.

The client half is `ModalTarget(function="pipeline")`, which needs a `function`
argument that the class does not have today.
`ModalTarget.prepare` reads `output.out_dir` and raises without it
(`notebooks/target.py`), so the target gains one `job_kind` branch whose
`"pipeline"` case reads `out_root`. The job carries no host path either way,
because the app rewrites `out_root` to `/runs/<run_id>`. `target.map_job` and
`target.job_paths` need no new field table, because a fixture job holds no input
path.

Acceptance: `modal run tools/modal/app.py::pipeline_demo` prints one document
whose T4 leg names `host.arch` `sm_75` at fp32 and whose H100 leg names `sm_90`
at bf16, all three gates pass, and the `nanochat-runs` Volume holds `deliver/`
twice, under `<run_id>-t4` and `<run_id>-h100`.

#### 6.4 The Modal CPU leg is a second image

The Modal image of `tools/modal/app.py` pins one compiled backend:
`NANOCHAT_CPP_LIB` names the CUDA library and the build passes `--config=cuda`.
A container with no card therefore fails inside `nanochat_init`, and
`_check_cards` calls `nvidia-smi`, which does not exist there.

So a CPU leg needs a second image whose build states `--define=backend=cpu` and
whose function omits `gpu=`. Do not add a CPU branch to `tools/modal/app.py`. Keep the
two images apart, because one Bazel output base holds only one build of
`libnanochat_shared.so`.

### 7. The integration gate

`python/tests/pipeline_test.py` calls `pipeline.run(job)` with the `demo`
profile. The assertions are the contract, not the loss values. The parity rungs
own the numbers.

| # | Assertion | Catches |
|---|---|---|
| A1 | Each stage returns status 0 with the step count it was asked for | A shortened schedule |
| A2 | Every loss and reward is finite; the base loss falls from step 1 to step 100 | A dead optimizer |
| A3 | Metadata chain: `base`@100, `sft`@100, `rl`@105, and every recorded config equals the job's | B1, B2, a wrong warm start |
| A4 | The parameter L2 norm differs between base and SFT, and between SFT and RL | A stage that only rewrites its file |
| A5 | One group of rollout rewards holds two distinct values, so its advantages are not all zero | B4 |
| A6 | Two runs from the same SFT checkpoint and the same seed, one with the graded reward and one with a constant reward, give different final parameters | A gradient path that the inherited moments hide |
| A7 | `deliver/chatrl.ckpt` equals the RL checkpoint byte for byte, and the report digests match the bytes | A lost artifact |
| A8 | The report names the stage order, the seeds, and the host facts | An unreviewable run |
| A9 | The delivered file loads through the public container reader, and its metadata reports `rl` at step 105 | A container only the pipeline can read |
| A10 | The eval stage ran, both `bpb` values are finite and inside `[0, 8]`, and each sample is non-empty text | A silent eval, and a wrong `eval_batch` |
| A11 | The RL checkpoint digest is the same before and after the eval stage | An eval that writes |
| A12 | The report names the evaluated checkpoint, its digest, the corpus path, the batch size, and the batch count | An unreviewable eval |
| A13 | `deliver/chatrl.ckpt` holds the RL weights, and its metadata reports stage `rl` | A delivery that follows the wrong stage |

A4 does not prove the warm start, because the loaded moments move the weights on
their own. Read A5 through `rl.run(..., on_step=...)`, which receives the per
rollout rewards; `RlReport` carries no advantage.

A6 cannot be "a constant reward leaves the parameters unchanged". A warm start
carries the optimizer state (`docs/post-training.md` §7), so the inherited AdamW
moments move the weights with an advantage vector of exactly zero. A6 is
therefore a two-run comparison, and it is the one assertion that isolates the
advantage term.

A5 and A6 fail today. A4 passes today for the wrong reason. A3 and the argv that
`nc.sft.run` builds are what prove the chain.

A10 to A13 are the eval stage's gate, and they land with the stage
([post-training.md](post-training.md) Part 3, section 10). A10 reads the
two bits-per-byte numbers and the sampled text out of the eval record, so an
evaluator that reports no bytes, or a batch that cannot form, is a failure and
not a quiet infinity. A11 compares the digest the report carries against the
file after the stage, which is how "the eval writes no checkpoint" (section 5)
stays true. A12 is the reviewability condition: the report names the file it
scored, that file's digest, the corpus, the batch size, and the batch count. A13
is the one that guards the decision to name the RL file in the eval record: the
delivered checkpoint has to be the RL weights at stage `rl`. The batch count of
A12 is one, and a report that claims eight is wrong for the corpus of this plan.

Two targets, one source:

- `//python:pipeline_test`: the `demo` profile, `size = "medium"`, data
  `//bindings:nanochat_shared`, `//src:nanochat_cli`, the parquet
  fixture, and the tokenizer fixture. About 5 s, so it stays in the default
  loop. It holds A1 to A13: the `demo` shape and a short schedule, so the eval
  stage runs at the same size as the three training stages.
- `//python:pipeline_gpu_test`: the same source, `NANOCHAT_CPP_BACKEND=cuda`,
  tags `gpu` and `manual`.

#### 7.1 The three gates of the two legs

A1 to A13 are the gate of one chain on one host. The two Modal legs need three
more, and they live in `tools/modal/app.py` beside the comparison, not in the
pipeline test: the entry point reads two reports and judges them
([post-training.md](post-training.md) Part 3, section 8).

| Gate | Claim | Check |
|---|---|---|
| G1 | bf16 ran on the H100 | `host.precision` is `bf16`, the RL checkpoint metadata records `bf16`, and `host.arch` is `sm_90` |
| G2 | the H100 leg did not regress | the H100 base `step_ms` is at most 1.25 times the T4 base `step_ms` for the same job, and both legs ran with the same `cpu=` |
| G3 | the run matches the model | the H100 container ran `roofline --device h100 --precision bf16 --json` for the job shape and recorded `predicted_step_ms` and `limit`; the measured ratio is inside the recorded band |

G2 is a smoke check, not a throughput claim. Both legs are bound by the host
program, so the gate catches a large regression and a broken device path; it
does not measure the card. G3 starts with the band `[1, 200]`, and the first
two-leg run measured a ratio of 162.8. Twice the measured value is wider than
200, so the band stays `[1, 200]` (row 9 of
[post-training.md](post-training.md)).

The document records two numbers that no gate reads: the MFU, and the cross-leg
speed ratio. At 64 tokens per step, one row of sequence 64, the device bound is
about one millisecond inside a step of tens of milliseconds, so the MFU reads
near 0.01 percent; and the chain is bound by launches, so a speed ratio below one
is as correct as one above it. Recording them keeps a reader from reading either
as a pass.

## Part 3. The scale-up of that chain

_Absorbed from the former `pipeline-scaleup-plan.md`. Its section numbers are
unchanged, so a reference to a section of that document resolves to the same
number here._

### 4. Decisions

| # | Decision |
|---|---|
| P1 | Add a second profile, `standard`. Keep `demo` for the CPU test loop. `Job()` defaults to `standard`. |
| P2 | Keep the sequence length at 64 and the base batch at one row. The committed corpus fills one 65-token row per epoch. |
| P3 | The H100 leg runs at bf16. No leg of the chain runs an fp8 training path. One further image builds fp8 for a benchmark and trains nothing; section 4.2 states it. |
| P4 | Add the eval stage in process, through `api.evaluate`, `api.Model.load`, and `api.Tokenizer`. Add no third stage executable. |
| P5 | The eval stage reads the base checkpoint and the RL checkpoint, so the report holds both numbers. |
| P6 | Run T4 first, then H100, as four functions over two images. One library per image keeps the architecture and the precision honest. |
| P7 | Add one named config, `hopper`, for a hand run only. The build helper keeps its explicit flag triple. |
| P8 | Gate the H100 leg on evidence that bf16 ran, and on a comparison with the T4 leg. Do not gate on an fp8 speedup, and do not gate on MFU. |
| P9 | Change no frozen header. `caps.json` already declares bf16 for the H100. |

#### 4.1 Why the H100 leg runs at bf16

The T4 has no bf16 and no fp8, so the T4 leg stays at fp32. The H100 declares
bf16 and fp8. This plan takes bf16 for three reasons:

- The build helper supports bf16 today. The H100 leg therefore adds no new build
  precision, and no risk to the build path.
- The chain is bound by launches at this shape. Section 14 gives a 0.76 ms device
  bound inside a step of tens of milliseconds, so the precision does not change
  the measured time.
- The bf16 leg on an H100 runs the fp8 operators only when the chain runs the
  hardware correctness gate. The chain's base stage is the mesh trainer, and it
  does not call `RunHardwareCorrectnessGate`, so the first H100 run reported
  `startup_check: not_run`. A hand-wired gate then failed in
  `cublasLtMatmulAlgoGetHeuristic` with status 15 on Hopper (section 14). The
  bf16 leg therefore proves the bf16 training path, not the fp8 operators.

What the leg does not prove is the fp8 training graph and the fp8 planner rule.
Those belong to a later row, and the row needs a Hopper benchmark first.

#### 4.2 The third image, and why it is not a leg

The fp8 campaign of [kernels.md](kernels.md) section 10.4 needs one card run of
the GEMM sweep at fp8, and no chain leg can serve it. A report's
`host.precision` is a fact of the library that ran, so a bf16 image cannot
measure an fp8 build and label the result honestly.

The app therefore holds one more image, `h100-fp8`, and one more card job,
`gemm_bench_fp8`. The image is in `LEGS`, because a lookup by name must reach it.
It is not in `LEG_ORDER`, so no run of the chain ever picks it up. P3 keeps its
rule: the two chain legs stay at fp32 and bf16, and the fp8 image times a
benchmark and trains nothing.

The build helper gained `fp8` in `SUPPORTED_PRECISIONS` for the same reason. It
passes the leg's precision through the one builder, so the image states
`--define=precision=fp8` and no Bazel config of its own.

### 5. The shape and the schedule

The shape has one home, `PROFILES` in `python/nanochat_cpp/pipeline.py`:

| Field | `demo` | `standard` |
|---|---|---|
| `num_layers` | 2 | 8 |
| `hidden_dim` | 64 | 512 |
| `num_heads` | 4 | 8 |
| `num_kv_heads` | 2 | 4 |
| `seq_len` | 64 | 64 |
| `vocab_size` | 487 | 487 |
| `padded_vocab_size` | 512 | 512 |
| `window_pattern` | `L` | `L` |

The schedule grows by about a factor of two:

| Field | Today | New |
|---|---|---|
| `base_steps` | 100 | 200 |
| `sft_steps` | 100 | 200 |
| `rl_steps` | 5 | 8 |
| `num_prompts` | 4 | 4 |
| `num_samples` | 4 | 4 |
| `max_tokens` | 32 | 32 |
| `eval_batches` | none | 1 |
| `eval_samples` | none | 2 |

`eval_batches` is one, and it is not a tunable number in this plan. The committed
corpus holds 119 tokens, so one epoch fills one 65-token row and no more. A batch
of two rows cannot form, and the evaluator then reports no bytes and returns
infinity (`src/data.cc`, `src/eval.cc`). Section 6 gives the consequence.

The device work of one step grows by about 10 times, and the step count grows by
about 2 times. The device bound stays far below the host overhead, so a local CPU
chain costs tens of seconds and the default test loop stays near five seconds.

`_check` rejects a job with a non-positive field in the new list. It also rejects
a profile that breaks three rules of `Config::Validate`
(`include/nanochat/config.h`):

- `hidden_dim` must be divisible by `num_heads`;
- `num_heads` must be divisible by `num_kv_heads`;
- `vocab_size` must not be above `padded_vocab_size`.

### 6. The eval stage

`stage_eval(job, base, rl)` runs after RL and writes no checkpoint.

- Build the model with `api.Model(config(job))` and load it with
  `api.Model.load(checkpoint)`.
- Call
  `api.evaluate(model, corpus["parquet"], tokenizer, steps=job.eval_batches, batch=1, seed=job.seed)`
  for the base checkpoint and for the RL checkpoint. Pass `batch=1` by hand,
  because the default of `api.evaluate` is 8.
- Call `api.Model.generate(prompt, temperature=0.0, max_tokens=job.max_tokens)`
  for `eval_samples` completions. Decode them with `api.Tokenizer.decode`.
- Build the record with `_stage("eval", job.eval_batches, (), rl.checkpoint,
  seconds, extra)`. The RL checkpoint exists, so `_stage` reads its digest and
  its metadata block, and A11 can compare the two.

The `checkpoint` field names the RL file on purpose. `deliver()` copies the
checkpoint of the last stage, so the delivered file stays `chatrl.ckpt`.

The `extra` field holds `bpb_base`, `bpb_rl`, `eval_batches`, `eval_batch`,
`eval_corpus`, `evaluated`, `evaluated_sha256`, and `samples`.

An evaluator call never calls `DataLoader::Reset`. The trainer resets the loader
at the start of each step, and the evaluator does not
(`src/train.cc`, `src/eval.cc`). The report therefore names one batch, not eight.
A tenfold larger corpus would allow more batches, and that corpus needs a
producer. The producer is a separate work item, because the image installs the
`hub` extra and not the `hfdata` extra that holds `pyarrow`
(`python/pyproject.toml`).

Chat accuracy stays out of this plan. `ChatEvaluator` reads Hub datasets
([eval.md](eval.md) section 5), and no stage of this chain reads the Hub.

### 7. The Modal legs: T4 first, then H100 at bf16

`tools/modal/app.py` gains one record, `Leg`, with these fields: name, card
type, card count, architecture, precision, and build configuration.

| Leg | Card | Architecture | Precision | Build command |
|---|---|---|---|---|
| `t4` | `T4` | `sm_75` | `fp32` | `--config=cuda --define=precision=fp32 --@rules_cuda//cuda:archs=sm_75` |
| `h100` | `H100` | `sm_90` | `bf16` | `--config=cuda --define=precision=bf16 --@rules_cuda//cuda:archs=sm_90` |

The build helper emits the flag triple of the table today
(`_build._build_command`). It never emits a named config. The `hopper` config of
P7 is a shorthand for a hand run only.

Both legs link the native ring, which is the one collective, and the chain runs
`world_size=1`. The `h100` simulator config would answer the capabilities from
the declaration; the hardware leg must not use it.

The rest of the change:

- `image_for(leg)` returns one image for one leg. The stage executables,
  `NANOCHAT_CPP_PRECISION`, `NANOCHAT_CUDA_ARCH`, and the build call read
  the leg.
- `_check_cards(leg, job_json)` reads the leg instead of the module globals.
- Four functions: `train`, `pipeline`, `train_h100`, and `pipeline_h100`. The
  first two run the T4 leg, because `notebooks/target.py` looks the name `train`
  up through `modal.Function.from_name`. Keep the name.
- `pipeline_demo` runs the T4 leg first, then the H100 leg. The run identifiers
  are `<run_id>-t4` and `<run_id>-h100`, so both reports stay on the
  `nanochat-runs` Volume.
- The entry point prints one document with three keys: `legs`, `comparison`,
  and `gates`. It exits non-zero when a gate fails.
- Both functions set the same `cpu=` value, so the two legs get the same host
  budget. Modal sizes the CPU by memory when `cpu=` is absent.
- The image build adds `//src:nanochat_cli`. Gate G3 runs the tool inside the
  container, and the tool must not start a second Bazel build there.

`notebooks/target.py` gains a `leg` argument on `ModalTarget`, so a notebook
selects one leg.

The T4 leg runs first. The T4 artifact therefore survives an H100 failure, and a
reviewer reads the cheap leg while the expensive leg runs.
