# Post-training: SFT and RL

How `nanochat.cpp` adapts a pretrained base model to the chat distribution and
then improves it with reward. Post-training reproduces `scripts/chat_sft.py`
(supervised fine-tuning) and `scripts/chat_rl.py` (reinforcement learning on
GSM8K) through the Python bridge.

This is a design document. Evaluation is a separate, forward-only concern
([eval.md](eval.md)); the process seam is [python-bridge.md](python-bridge.md);
known divergences are [parity.md](parity.md).

## 1. Scope

- **SFT** turns a base model into a chat model on a fixed conversation mixture.
- **RL** turns a chat model into a better problem solver using on-policy
  rollouts and a reward.

Non-goals: distributed training, preference optimization (DPO and friends), and
a native tokenizer or chat template in C++. The C++ runtime sees only token ids
and a mask; the chat template lives in the reference tokenizer, called from
Python.

## 2. The shared training step

SFT and RL are two instances of one mechanism: a masked, weighted,
mean-normalized cross-entropy over token sequences. Only three things differ
between them — where the sequences come from, the per-token weight, and the
normalizer.

For a batch of `rows = batch * seq` positions with per-token loss `l_i` (zero at
ignored positions), per-token weight `w_i`, and normalizer `Z`:

```
L = (1 / Z) * sum_i  w_i * l_i
```

| | Sequence source | Weight `w_i` | Normalizer `Z` |
|---|---|---|---|
| Pretrain | shards | 1 | valid tokens |
| SFT | conversation mixture | 1 on supervised tokens, 0 elsewhere | valid tokens |
| RL | on-policy rollouts | advantage of the sequence on sampled tokens, 0 elsewhere | valid tokens x passes x examples per rank |

The model graph is identical in all three cases. This is the central design
decision: RL does not need a second backward pass. It needs a per-row multiplier
on the existing classifier gradient.

### 2.1 Valid-token normalization

The reference uses `F.cross_entropy(..., ignore_index=-1)` with the default mean
reduction, which divides by the number of **non-ignored** targets. The current
C++ `ForwardLoss` divides by `rows` (all positions). The two agree only when no
target is ignored, which is true for pretraining but false for SFT and RL.

The shared step therefore normalizes by the valid count:

- `ForwardLoss` returns the mean over valid targets (identical to today when all
  targets are valid, so pretraining is unchanged).
- `BackwardAccumulate(scale)` divides by the valid count instead of `rows`.
- `BackwardWeighted(row_weights, scale)` applies `row_weights[i] * scale` to row
  `i` instead of a uniform scale.

`ClassifierBackward` already zeroes ignored rows and the padded vocabulary tail,
so a row weight of 0 is exactly "do not train this position".

### 2.2 Interface

```cpp
// include/nanochat/model.h (additive)
class Model {
  // Weighted policy-gradient backward. `row_weights` has batch*seq entries;
  // ignored rows are already zeroed by the classifier. Accumulates into the
  // parameter gradients (does not zero them).
  virtual void BackwardWeighted(const float* row_weights, float scale) = 0;
};
```

`ClassifierParams` gains `const float* row_scale = nullptr;` on both backends
(null is the current behavior). A `backends/cuda/kernels` finite-difference test covers
the weighted path, and the RL parity fixture covers the composed step.

## 3. The chat data model

The bridge owns the chat template and the loss mask; the C++ side receives token
ids and a `-1` mask on the targets.

- **`render_conversation`** (SFT) returns `(ids, mask)`. `mask = 1` for assistant
  text, `<|assistant_end|>`, and the python tool-call tokens
  `<|python_start|> expr <|python_end|>`; `mask = 0` for BOS, user tokens, and the
  tool-output tokens `<|output_start|> result <|output_end|>`. The model learns
  to emit tool calls but never the results, which are supplied at inference time.
- **`render_for_completion`** (RL and generative evaluation) drops the reference
  assistant message and appends `<|assistant_start|>` to prime a completion.

Masking is applied on the shifted targets: `targets[mask == 0] = -1`. The same
rule covers prompt tokens, tool outputs, and (in SFT) padding.

This is why SFT and RL share the backward: in both cases the loss is a weighted
sum over the *sampled or supervised* positions, and the mask is the only
difference in where the weights are zero.

## 4. SFT

### 4.1 Data

A `TaskMixture` of SmolTalk (train), MMLU (auxiliary train, x3), and GSM8K
(train, x4), rendered with `render_conversation` and packed BOS-aligned with
best-fit. Unlike pretraining's best-fit packer, SFT **pads** the remainder of a
row instead of cropping, and masks the padding targets to `-1`. Validation uses
the matching test splits.

The packer is a Python data-pipeline concern ([data.md](data.md)). If the shard
is materialized for the C++ loader, the bridge writes the rendered ids and the
`-1`-masked targets; the C++ `DataLoader` does not need to know about chat.

### 4.2 Loop

- Load the base checkpoint and inherit hyperparameters from its metadata
  (`max_seq_len`, batch sizes, learning rates) unless overridden.
- Warm-start the optimizer moments from the base checkpoint's optimizer state
  (LRs are reset to the fresh SFT values).
- Forward + backward over `grad_accum` micro-batches; optimizer step.
- LR schedule by dataset progress (warmup, constant, warmdown); Muon momentum
  warms from 0.85 to 0.95 over 300 steps.
- Periodic val bpb and ChatCORE evaluation (reusing [eval.md](eval.md)).
- Save to `chatsft_checkpoints/<tag>` with optimizer state.

SFT is a one-shot subprocess from the bridge, exactly like `base_train`: the
loop does not depend on fresh model output.

## 5. RL

### 5.1 Rollout

For each training example, per step:

1. `render_for_completion` gives the prompt tokens.
2. `GenerateBatch` samples `num_samples` completions (temperature, top-k,
   `max_new_tokens`, a per-sampling-step seed), returning token ids and masks.
3. The bridge decodes, computes the reward, and forms
   `advantage = reward - mean(reward)` across the samples for that example.
4. Prompt and forced (tool-output) positions are masked; sampled positions are
   trained. This is the same `mask = 0` rule as SFT.

The calculator tool is the one place RL differs structurally from SFT: the
reference engine detects a python block, evaluates the expression, and forces
the output tokens. That requires decoding inside the generation loop, so it
cannot run in a tokenizer-agnostic C++ engine. Two options are on the table:

- **MVP:** `GenerateBatch` does not force tool outputs. The model still samples
  its own `<|python_start|>` blocks; the reward is read from the final `####`
  answer. Record a parity entry and measure pass@k before investing more.
- **Parity:** a streaming `chat_engine` protocol where the bridge receives one
  token column at a time, decodes python blocks, and sends forced tokens back.
  This is also what an interactive `chat_cli` needs.

### 5.2 Objective

Per micro-batch of one example's rollouts:

```
logp_i = -l_i
pg_obj = sum_i advantage[seq(i)] * logp_i
L = -pg_obj / (num_valid * num_passes * examples_per_rank)
```

No KL term (there is no reference model), no PPO ratio or clip (the data is
on-policy), and DAPO-style token-level normalization. `num_valid` is counted per
pass; gradients accumulate across passes with `BackwardWeighted`, and the
optimizer steps once per example group.

### 5.3 Optimizer and schedule

The standard nanochat grouping with `init_lr_frac = 0.05` applied to every
group, a linear rampdown `1 - it / num_steps`, Muon momentum held constant at
0.95 (the RL script never updates it), and weight decay 0. The C++ scheduler
expresses this with `warmdown_ratio = 1`, `final_lr_frac = 0`, and a flat Muon
momentum. Checkpoints go to `chatrl_checkpoints/<tag>`.

### 5.4 Evaluation

Every `eval_every` steps, report GSM8K pass@k on the test split using the
generative path from [eval.md](eval.md).

## 6. Generation for rollouts

RL rollouts and generative evaluation share `GenerateBatch`: prefill once, clone
the KV cache per sample, decode in lockstep, stop each row on its own terminal
token, and return ids and masks. The graph, the cache clone, and the sampling
parameters are documented in [model.md](model.md). The C++ engine is
tokenizer-agnostic; terminal token ids are passed in.

## 7. Checkpoints and optimizer state

- Pretrain checkpoints are NCHKPT01 and load directly.
- Reference base/SFT checkpoints are torch `.pt` and are converted by
  `checkpoint.py`.
- Post-training must persist optimizer state (moments) for warm-start and
  resume. NCHKPT01 currently stores model parameters only; add optimizer-state
  records (additive). SFT warm-starts from the base optimizer state; RL keeps its
  state in memory across the run and writes it on checkpoint.

## 8. Orchestration and the sandbox

- **SFT** is a one-shot run: the bridge materializes the packed shard, then
  launches the C++ training binary under `tools/nanochat train`, exactly like
  `base_train`.
- **RL** is on-policy: generation and training interleave, and optimizer moments
  persist. The target is a persistent worker holding the model and optimizer,
  exchanging rollout and advantage messages with the bridge over pipes. For
  correctness, the same step logic is available as a file-driven `rl_step`
  binary that consumes a fixture; that is the RL analog of the training-parity
  harness and the exact gate for the objective.
- Every run goes through `tools/nanochat` so the GPU broker and resource sandbox
  apply ([sandbox.md](sandbox.md)).

## 9. Invariants and non-goals

- One fixed graph; no autograd. RL reuses the classifier backward.
- Valid-token normalization is the single source of truth for the mean, shared
  by pretrain, SFT, and RL.
- Tokenizer-agnostic C++: the chat template and mask live in the bridge.
- Single process; no DDP ([parity.md](parity.md) D6).
- No native tokenizer, chat template, or best-fit packer in C++ in this plan.

## 10. Parity

| ID | Status | Difference |
|---|---|---|
| E2 | `open`/`closed` | calculator tool absent (MVP) or arithmetic-only with C++ float formatting |
| E3 | `equivalent` | reference checkpoints converted torch -> NCHKPT01 |
| E6 | `open` | SFT prerequisite satisfied by a converted reference checkpoint, not a native port |
| E7 | `open` | file-based `rl_step` makes rollouts lag one step; the worker removes this |
| E5 | `out-of-scope` | no DDP |

## 11. Status and roadmap

Not implemented. To build, in order:

1. Valid-token normalization in `ForwardLoss`/`BackwardAccumulate` (also
   correct for SFT and harmless for pretrain).
2. `BackwardWeighted` + the classifier row-scale kernel + a finite-difference
   test.
3. `GenerateBatch` ([model.md](model.md)).
4. Optimizer-state records in NCHKPT01 and the checkpoint converter.
5. `rl_step` + the RL parity fixture (the objective gate).
6. The SFT bridge and run.
7. The RL orchestration bridge and worker.

Gates: the weighted-backward finite-difference test and the RL parity fixture at
T0/T2; generation parity at T1; an RL smoke run at T2. See
[testing.md](testing.md).
