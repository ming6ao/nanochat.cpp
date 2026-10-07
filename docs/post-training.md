# Post-training: supervised fine-tuning and reinforcement learning

`nanochat.cpp` adapts a pretrained base model to the chat distribution. It then
improves the model with reward. The reference implementation is
`scripts/chat_sft.py` and `scripts/chat_rl.py` in the PyTorch `nanochat` tree.
The Python API in [python.md](python.md) drives both stages.

This is a design document and the status of record for post-training. Keep it
current. When the code changes, change this document in the same commit.
Section 11 lists the state of each piece and names the file that holds it.

The companion documents are [eval.md](eval.md) for evaluation,
[python.md](python.md) for the process seam, [parity.md](parity.md) for the
known differences, and [model.md](model.md) for the model and the data formats.

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
([tokenizer.md](tokenizer.md), [parquet-native.md](parquet-native.md)). The
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
`render_conversation` is not in the tree yet (section 11).

## 4. SFT

### 4.1 Data

The reference mixes SmolTalk (train), MMLU (auxiliary train, x3), and GSM8K
(train, x4). The bridge renders each conversation with `render_conversation`.
The bridge then packs the rows BOS-aligned with best-fit. Unlike pretraining,
SFT pads the remainder of a row instead of cropping. The bridge masks the
padding targets to `-1`. Validation uses the matching test splits.

The native C++ loader in `src/data.cc` reads parquet documents and tokenizes
during the run. It uses best-fit packing for pretraining. The SFT path needs a
chat renderer and a mask, so the bridge owns the SFT packer. `DataLoader` does
not know about chat data.

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

Implementation: `TrainLoop` in `src/train.cc` provides the step loop, the
scheduler, the logger, and checkpointing. The dataset-progress schedule and the
SFT warm start are not in the tree yet (section 11).

## 5. RL

### 5.1 Rollout

For each training example, per step:

1. `render_for_completion` gives the prompt tokens.
2. `GenerateBatch` samples `num_samples` completions. It uses the temperature,
   the top-k value, `max_new_tokens`, and a per-sampling-step seed. It returns
   token ids and masks.
3. The bridge decodes the samples, computes the reward, and forms
   `advantage = reward - mean(reward)` across the samples of that example.
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

- `NCMDL001` is the flat parameter file that `TrainModel::Save` writes
  (`src/model.cc`).
- `NCHKPT01` is the self-describing container that `Checkpointer` uses
  (`src/data.cc`). The reference converter in `python/nanochat_cpp/checkpoint.py`
  writes the same format.

Both containers hold parameters. The `Checkpoint` class also carries an
optimizer-state section. That section holds the AdamW first and second moments
and the Muon momentum and second-moment buffers. The record names are stable,
so a later load can match them to the optimizer groups.
`Checkpointer::SaveModel` and `Checkpointer::LoadModel` have `Optimizer`
overloads. A parameter-only file stays loadable.

Pretrain checkpoints load directly. Reference base and SFT checkpoints are torch
`.pt` files. `tools/convert_checkpoint.py` remaps them. The converter also
reads the reference optimizer state.

Implementation: `include/nanochat/data.h`; `src/optim_state.h`; `src/optim.cc`;
`src/train.h`; `src/optimizer_state_test.cc`;
`python/nanochat_cpp/checkpoint.py`; `tools/convert_checkpoint.py`.

## 8. Orchestration and the sandbox

- **SFT** is one job. The bridge materializes the packed shard, and then it
  launches the C++ training binary under `tools/nanochat train`.
- **RL** is on-policy. Generation and training interleave, and the optimizer
  moments persist. The target is a persistent worker. The worker holds the
  model and the optimizer. It exchanges rollout and advantage messages with the
  bridge over pipes.
- For correctness, the same step logic exists as a file-driven `rl_step` binary.
  That binary consumes a fixture. It is the RL analog of the training-parity
  harness, and it is the gate for the objective.
- Every run goes through `tools/nanochat`, so the GPU broker and the resource
  sandbox apply ([sandbox.md](sandbox.md)).

Implementation: not in the tree yet (section 11).

## 9. Invariants and non-goals

- One fixed graph; no autograd. RL reuses the classifier backward.
- Valid-token normalization is the single source of truth for the mean.
  Pretraining, SFT, and RL share it.
- The C++ model engine does not depend on the tokenizer. The bridge owns the
  chat template and the loss mask.
- Single process; no distributed data parallel ([parity.md](parity.md) D6).
- No native chat template in C++.

## 10. Parity

`parity.md` is the canonical list. The table below lists the entries that
post-training uses. Add a new entry to `parity.md` when you record a new
difference.

| ID | Status | Difference |
|---|---|---|
| E1 | `open` | Sampled modes use the C++ RNG, not torch; greedy is bit-identical |
| E3 | `equivalent` | Reference checkpoints converted torch to NCHKPT01 |
| E4 | `equivalent` | CORE and chat scoring use the C++ forward with the reference task logic |
| E5 | `out-of-scope` | No distributed evaluation |
| P1 | `open` | The calculator tool is absent in the MVP; the model samples its own python blocks |
| P2 | `open` | A file-driven `rl_step` makes the rollouts lag one optimizer step |
| P3 | `equivalent` | SFT starts from a converted reference base checkpoint |

## 11. Status

The table records the state of each piece. Update the table when the code
changes.

| Piece | State | Location |
|---|---|---|
| Masked, valid-token mean | Done | `src/model.cc`; `src/masked_loss_test.cc`; `tests/post_training_core_test.cc` |
| Weighted policy-gradient backward | Done | `include/nanochat/model.h`; `include/nanochat/kernels.h`; `src/model.cc`; `backends/cpu/kernels.cc`; `backends/cuda/kernels/classifier.cu` |
| Weighted backward in the C ABI and the Python API | Done | `include/nanochat/capi.h`; `bindings/nanochat_capi.cc`; `python/nanochat_cpp/api.py` |
| Batched sampling with a per-row stop and a mask | Done | `src/generate.cc`; `include/nanochat/model.h` |
| Generation-parity gate, greedy | Done | `tests/generate_parity_test.cc` |
| Task logic for GSM8K, ARC, MMLU, and HumanEval | Done | `python/nanochat_cpp/tasks.py` |
| ChatCORE and generative chat evaluation | Done | `python/nanochat_cpp/chat.py` |
| `render_for_completion` prompt builder | Done | `python/nanochat_cpp/chat.py` |
| Checkpoint path resolution for base, SFT, and RL | Done | `python/nanochat_cpp/checkpoint.py` |
| Reference `.pt` conversion into NCHKPT01 | Done | `tools/convert_checkpoint.py` |
| Optimizer-state records in NCHKPT01 | Done | `include/nanochat/data.h`; `src/optim_state.h`; `src/optim.cc`; `src/optimizer_state_test.cc` |
| SFT conversation renderer (`render_conversation`) | Missing | No file |
| SFT packer with padding and a loss mask | Missing | `src/data.cc` has the pretraining packer only |
| SFT loop with a dataset-progress schedule and a warm start | Missing | `src/train.cc` has a step-based loop |
| RL step binary and the RL parity fixture | Missing | No `rl_step` |
| RL orchestration worker | Missing | No worker |
| SFT and RL bridge commands | Missing | `python/nanochat_cpp` |

## 12. Remaining work

Each phase names the owner and the gate. The interface changes touch frozen
headers, and the architect owns those headers (`AGENTS.md` section 1).

### Phase 1 — SFT data path

Owner: Python surface for the renderer and the packer; data pipeline for a
native shard reader.

1. Add `render_conversation` to the Python bridge.
2. Add the BOS-aligned best-fit packer with padding. Mark the padding targets
   with `-1`.
3. Add `tools/dump_sft_fixture.py` for a fixed conversation set.

Gate: `//python:chat_test` and a new packer parity test.

### Phase 2 — SFT loop

Owner: harness for `src/train.cc` and `include/nanochat/scheduler.h`.

1. Add a progress-based learning-rate schedule.
2. Add a flat Muon momentum option and the 300-step ramp.
3. Inherit the hyperparameters from the base checkpoint metadata.
4. Warm-start the optimizer moments from the base checkpoint.

Gate: A short SFT run at T2. Compare the loss curve to the reference.

### Phase 3 — RL step and parity fixture

Owner: harness for `src/`; oracle for `tools/**` and `tests/**`.

1. Add a file-driven `rl_step` binary.
2. Add `tools/dump_rl_fixture.py`. Record the reference advantages, the targets,
   and the first-step trajectory.
3. Normalize by `num_valid * num_passes * examples_per_rank`.

Gate: `//tests:rl_parity_test` at T2.

### Phase 4 — RL orchestration

Owner: harness for the worker; Python surface for the bridge.

1. Add the persistent worker with a pipe protocol.
2. Add the SFT and RL commands to the bridge.

Gate: An RL smoke run at T2.

### Phase 5 — Tool forcing (optional)

Owner: workflow for the streaming protocol; Python surface for the bridge.

1. Add the streaming `chat_engine` protocol.
2. Force the tool-output tokens in the generation loop.

Gate: A parity measurement at T1.
