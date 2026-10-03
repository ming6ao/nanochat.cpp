# Plan: evaluation and reinforcement learning

Status: evaluation phases implemented; reinforcement learning proposed. Owner:
architect/integrator. Scope: bring `nanochat.cpp` to feature parity with the
PyTorch reference evaluation and reinforcement-learning scripts
(`scripts/base_eval.py`, `scripts/chat_eval.py`, `scripts/chat_rl.py`) on the
single-GPU Pascal host, within the existing process seam
([python-bridge.md](python-bridge.md)).

This document is the work plan. Phases 1 and 2 (evaluation, workstreams
E0-E5) are implemented on `sliceme/eval`; the resulting state is described in
[eval.md](eval.md) and recorded in [parity.md](parity.md). The evaluation
sections below are now a record of what was built. Phases 3 and 4
(reinforcement learning and SFT, workstreams E6-E9) are not implemented, and
their sections remain a plan.

Every interface change below is additive to the frozen headers unless
explicitly noted.

## 1. Goal

Run the three reference workflows through the C++ runtime:

```bash
# Base evaluation: bits-per-byte, CORE, samples.
python -m nanochat_cpp.base_eval --eval core,bpb,sample --model-tag d8

# Chat evaluation: ARC, MMLU, GSM8K, HumanEval.
python -m nanochat_cpp.chat_eval -i sft -a GSM8K|ARC-Easy

# Reinforcement learning on GSM8K.
python -m nanochat_cpp.chat_rl --num-epochs 1 --examples-per-step 16 \
    --num-samples 16 --device-batch-size 8
```

The command names mirror the reference scripts, exactly as
`python -m nanochat_cpp.base_train` mirrors `scripts/base_train.py`.

## 2. What the reference scripts do

### 2.1 `base_eval.py` (`--eval core,bpb,sample`)

| Mode | Compute | Data |
|---|---|---|
| `bpb` | forward-only cross-entropy, normalized by target byte length | train + val token loaders |
| `sample` | 7 conditioned prompts greedy, 8 unconditioned at temperature 1.0 | tokenizer only |
| `core` | DCLM CORE: 22 ICL tasks, three scoring types | `eval_bundle.zip` (core.yaml, JSONL, random baselines) |

CORE scoring types: `multiple_choice` and `schema` pick the candidate
continuation with the lowest mean per-token loss; `language_modeling` requires
the argmax prediction to match every token of the continuation. Few-shot
examples are sampled with `random.Random(1234 + idx)`. The final metric is the
mean of `(accuracy - 0.01*random_baseline) / (1 - 0.01*random_baseline)`.

### 2.2 `chat_eval.py`

| Task | Type | Evaluation |
|---|---|---|
| ARC-Easy, ARC-Challenge, MMLU | categorical | one forward per batch; argmax over the logits of the answer-letter token ids at the final prompt position |
| GSM8K | generative | sample, decode, compare the `####` number to the reference |
| HumanEval | generative | sample, extract the code block, execute it against the test, check success |

`ChatCORE` is the mean centered accuracy over all five tasks with baselines
ARC/MMLU 0.25, GSM8K/HumanEval 0.0.

### 2.3 `chat_rl.py` (GSM8K GRPO/REINFORCE)

Per step, for each training example:

1. `render_for_completion(conversation)` -> prompt tokens (keeps
   `<|assistant_start|>`, drops the reference assistant message).
2. Sample `num_samples` completions with `Engine.generate_batch`
   (temperature, top-k, per-sampling-step seed), recording the per-token mask.
3. Reward each completion (`GSM8K.reward` = `####` answer match).
4. `advantage = reward - mean(reward)` (no z-score, no KL, no PPO ratio).
5. `logp = -model(inputs, targets, loss_reduction='none')`;
   `pg_obj = (logp * advantage[:, None]).sum()`;
   divide by `num_valid * num_passes * examples_per_rank`; `loss = -pg_obj`.
6. `backward()` accumulates across passes; `optimizer.step()` once per step.

Optimizer: the standard nanochat grouping with `init_lr_frac = 0.05` applied to
every group, a linear rampdown `1 - it/num_steps`, Muon momentum held at 0.95
(the RL script never updates it), and `weight_decay = 0`. Checkpoints go to
`chatrl_checkpoints/<tag>`. Every `eval_every` steps it reports GSM8K pass@k.

## 3. Current state and gap

Already present: `EvalBpb`/`eval_main`, the train graph with `ForwardLoss`,
`Backward`, `BackwardAccumulate`, the optimizer and scheduler, single-sequence
`Prefill`/`Decode`/`generate_main`, the NCHKPT01 checkpoint container, and the
Python bridge (`base_train`, `config`, `data`, `launcher`, `reference`).

| Capability | Reference | nanochat.cpp now | Needed |
|---|---|---|---|
| bpb eval | `evaluate_bpb` | `EvalBpb`, `eval_main` | train split, `split_tokens` -> steps, tag/step loading |
| base sampling | `Engine.generate_batch` | `Prefill`/`Decode`, one row | multi-sample batch, decode in bridge |
| CORE eval | `core_eval.py` + eval_bundle | none | sequence scorer binary, jinja rendering in the bridge |
| chat categorical | `run_categorical_eval` | none | logits-at-position scorer |
| chat generative | `Engine` + `task.evaluate` | none | batched generation, sandboxed code execution |
| RL training | `chat_rl.py` | none | weighted policy-gradient forward/backward, rollout driver |
| batched generation | `KVCache.prefill` clone, per-row masks | none | batch KV cache, clone, mask output, stop ids |
| tokenizer | BPE | `LoadTokenizer` stub returns null | bridge uses the reference tokenizer; C++ stays tokenizer-agnostic |
| chat template | `render_for_completion` | none | bridge uses the reference tokenizer |
| calculator tool | `Engine.use_calculator` | none | C++ calculator or a streaming protocol |
| reference checkpoint | torch `.pt` | NCHKPT01 only | torch -> NCHKPT01 converter |
| SFT | `chat_sft.py` | none | prerequisite for a trained RL policy |

## 4. Architecture decisions

### D-A. Keep the process seam

Python owns tokenization, chat templates, task datasets, few-shot sampling,
reward extraction, and code execution. C++ owns all model compute. This is the
existing [python-bridge.md](python-bridge.md) contract and it keeps the C++
runtime free of torch, pyarrow, jinja, and the BPE tokenizer.

### D-B. The C++ engine stays tokenizer-agnostic

Special token ids (`<|assistant_end|>`, `<|bos|>`, `<|python_start|>`, ...) are
passed as integer CLI arguments from the bridge, which reads them from the
reference tokenizer. The engine never decodes text.

Consequence: the calculator tool cannot run inside the C++ engine without a
detokenizer. Two options are described in D-C; tool use is deferred to the
parity phase.

### D-C. Generation: a batch primitive first, a stream protocol later

- **Batch primitive (MVP).** Add `GenerateBatch` to the model API: prefill
  once, clone the KV cache per sample, decode in lockstep, stop each row on its
  own terminal token, return token ids and a `1`/`0` mask per position. Used by
  base sampling, chat generative evaluation, and RL rollouts. No tool use.
- **Stream protocol (parity, optional).** A persistent `chat_engine` worker
  with a line protocol: the bridge sends a prompt and receives one token column
  at a time, so the bridge can decode python blocks, evaluate the expression,
  and send forced tokens back. This is the only faithful way to keep the
  tokenizer in Python and reproduce the calculator. It is also what `chat_cli`
  needs. Defer until `GenerateBatch` is proven.

### D-D. RL backward is a per-row scale on the existing classifier backward

The classifier backward already produces the sum-reduction gradient and zeroes
ignored rows (`target < 0`) and the padded vocabulary tail. The RL objective
differs from the training mean only by a per-row factor (the sequence's
advantage) and a global normalizer. Add:

- an optional `const float* row_scale` to `ClassifierParams` (null means 1),
  applied on both backends; and
- `Model::BackwardWeighted(const float* row_weights, float scale)`, which runs
  the existing backward with `row_weights` and a global `scale` instead of the
  uniform `scale / rows`.

This is a small, additive change and reuses the entire hand-written backward
graph. The RL objective value is computed host-side from the saved per-token
losses (`TrainModel::losses()`), as `EvalBpb` already stages them.

### D-E. Evaluation scoring is a forward-only binary

CORE and categorical chat evaluation need per-position losses and argmax over a
padded batch. Add `score_main`: read a fixture of padded token sequences and
candidate spans, run one `ForwardLoss`, write per-position NLL and argmax. The
bridge computes the task decisions (few-shot sampling, mean-loss argmin, exact
match, centering). This keeps the task semantics in the reference Python where
they are already correct.

### D-F. Reference checkpoints are converted, not loaded

The reference stores torch `.pt` state dicts; the C++ container is NCHKPT01. Add
`tools/convert_checkpoint.py` (torch -> NCHKPT01, name remap, dtype policy) so
`--model-tag d24` works against nanochat's released base/SFT checkpoints.
C++-trained checkpoints need no conversion.

### D-G. SFT is a prerequisite, not part of this plan

`chat_rl.py` loads an `sft` checkpoint. A base checkpoint has never seen the
chat template or the calculator format, so reward is ~0 and the RL signal is
uninformative. Two ways to unblock RL:

1. **Converter path (lower effort):** run the reference `scripts.chat_sft` once
   (or download a released SFT checkpoint) and convert it with D-F.
2. **Native SFT path (larger):** port the SFT data mixture and best-fit packer.
   This is a separate plan; it shares the row-scale/weighted-backward machinery
   because SFT masks prompt tokens exactly like RL masks them.

Recommendation: unblock RL with the converter, and schedule native SFT after
the RL math is proven.

### D-H. On-policy orchestration uses a persistent worker

On-policy RL interleaves generation and training, and the optimizer moments must
persist across steps. Reloading a checkpoint per step is too slow. The target is
a persistent `chat_rl` worker that holds the model and optimizer and exchanges
rollout/advantage messages with the bridge over pipes. For correctness bring-up
and parity, the same step logic is available as a file-driven binary
(`rl_step`) that consumes a fixture and is the RL analog of `train_parity`.

## 5. Interfaces to add

### 5.1 Model API (architect-owned, additive)

```cpp
// include/nanochat/model.h

// RL: accumulate a weighted policy-gradient backward. `row_weights` has
// batch*seq entries; ignored rows (target < 0) are already zeroed by the
// classifier. Accumulates into the parameter gradients (does not zero them).
virtual void BackwardWeighted(const float* row_weights, float scale) = 0;

// Batched sampling. Prefill once, clone the cache per sample, decode in
// lockstep, stop each row on its own terminal id. Tokenizer-agnostic.
struct GenerateParams {
  int num_samples = 1;
  int max_tokens = 256;
  float temperature = 1.0f;
  int top_k = 0;
  std::uint64_t seed = 42;
  int stop_id = -1;  // e.g. <|assistant_end|>
  int bos_id = -1;   // e.g. <|bos|>
};

struct GeneratedSequence {
  std::vector<int> tokens;             // prompt + generated, no terminal token
  std::vector<std::uint8_t> mask;      // 1 sampled, 0 prompt/forced
};

void GenerateBatch(Model* model, const int* prompt, int prompt_len,
                   const GenerateParams& params,
                   std::vector<GeneratedSequence>* out);
```

### 5.2 Kernel seam (architect-owned, additive)

`ClassifierParams` gains `const float* row_scale = nullptr;`. Both backends
apply it per row in `ClassifierBackward`; null is the current behavior. A
`dev/kernels` finite-difference test covers the weighted path.

### 5.3 New C++ binaries (`src/`, harness-owned)

| Binary | Reads | Writes |
|---|---|---|
| `score_main` | eval fixture: padded sequences + candidate spans | per-position NLL + argmax |
| `rl_step` | RL fixture: initial params, rollout batches, advantages | per-step loss, gradient norm, param norms, final params |
| `chat_rl` (worker) | line protocol: prompts, then advantages | completions, then step metrics |
| `generate_main` (extended) | `--num-samples`, `--stop-id`, `--bos-id`, `--out` | token ids + masks |

### 5.4 New Python modules (`python/nanochat_cpp/`, harness-owned)

| Module | Role |
|---|---|
| `base_eval.py` | drive bpb, sample, and CORE |
| `chat_eval.py` | drive categorical and generative tasks, ChatCORE |
| `chat_rl.py` | render prompts, reward, drive the RL worker, save checkpoints |
| `tasks.py` | thin import of the reference `tasks/` + `nanochat.execution` |
| `eval_fixture.py` | serialize/deserialize the score and RL fixtures |
| `checkpoint.py` | torch <-> NCHKPT01 conversion |

### 5.5 New fixtures and tools (oracle-owned)

| Artifact | Generator | Consumer |
|---|---|---|
| `tests/data/generate_parity.bin` | `tools/dump_generate_fixture.py` | `//tests:generate_parity_test` |
| `tests/data/rl_parity.bin` | `tools/dump_rl_fixture.py` | `//tests:rl_parity_test` |
| `tests/data/core_eval.bin` | `tools/dump_eval_fixture.py` | `//tests:score_parity_test` |

## 6. Workstreams

| # | Workstream | Owns | Depends on |
|---|---|---|---|
| E0 | Checkpoint converter and loader path | `tools/convert_checkpoint.py`, bridge `checkpoint.py` | — |
| E1 | Batched generation engine | `src/generate.cc`, `include/nanochat/model.h` (with architect), `src/generate_main.cc` | — |
| E2 | Weighted backward and row-scale kernel | `src/model.cc`, `backends/*/kernels/classifier.*`, `dev/kernels` | architect header change |
| E3 | Sequence scorer | `src/score_main.cc`, `src/eval.cc` | — |
| E4 | Base eval bridge | `python/nanochat_cpp/base_eval.py`, `config.py` | E0, E1, E3 |
| E5 | Chat eval bridge | `python/nanochat_cpp/chat_eval.py`, `tasks.py` | E1, E3 |
| E6 | RL step and parity fixture | `src/rl_step.cc`, `tools/dump_rl_fixture.py`, `tests/rl_parity_test.cc` | E2 |
| E7 | RL orchestration bridge | `python/nanochat_cpp/chat_rl.py`, `chat_rl` worker | E1, E2, E6 |
| E8 | Calculator tool (optional parity) | `src/calculator.cc`, `chat_engine` protocol | E1 |
| E9 | Native SFT (separate plan) | `src/sft.cc`, bridge `chat_sft.py` | E2 |

Dependency graph: E0 and E1 are independent and can start first. E2 unblocks E6.
E3 unblocks E4 and E5. E6 unblocks E7. E8 and E9 are later.

## 7. Phasing

### Phase 1 — eval on a converted checkpoint (T0/T2)

1. E0: convert a small reference base checkpoint; `eval_main` loads it and
   reproduces the reference bpb within tolerance.
2. E3: `score_main` + a tiny CORE fixture; the bridge reproduces CORE decisions
   on a few examples.
3. E1: `GenerateBatch` on the CPU backend; greedy output matches a naive
   single-row loop.
4. E4: `base_eval.py` end to end (bpb + sample + core) on the tiny model.

Gate: `tools/nanochat test` green; a T2 run of `base_eval.py` on a d8 model
matches the reference bpb to `1e-4` and CORE to `1e-3` on a 100-example subset.

### Phase 2 — chat eval

1. E5: categorical tasks (MMLU, ARC) through `score_main`; generative GSM8K
   through `GenerateBatch`; HumanEval through the reference `execute_code`
   under the sandbox.
2. ChatCORE reported when all five tasks run.

Gate: `chat_eval.py` matches the reference per-task accuracy on a fixed
`--max-problems` subset; generation parity fixture passes on CPU and GPU.

### Phase 3 — RL math (T0/T2)

1. E2: `BackwardWeighted` + row-scale kernel + finite-difference test.
2. E6: `dump_rl_fixture.py` records a reference PG trajectory; `rl_step` matches
   the per-step loss, gradient norm, and parameter trajectory.
3. Bridge `chat_rl.py` runs a short RL run from a converted SFT checkpoint and
   reports GSM8K pass@k.

Gate: `//tests:rl_parity_test` strict (loss `1e-5`, param `1e-4`); a 10-step
RL smoke run completes and reward is non-degenerate.

### Phase 4 — throughput and parity polish

1. E7 persistent worker (no per-step checkpoint reload).
2. E8 calculator tool + `chat_engine` protocol, if the measured pass@k shows
   tool use matters.
3. Benchmark generation and RL step throughput; record the commit hash.

## 8. Test gates

| Gate | Tier | Command |
|---|---|---|
| Weighted-backward finite difference | T0 | `tools/nanochat test //src:model_gradient_test` |
| Row-scale classifier parity | T0 | `tools/nanochat test //tests:score_parity_test` |
| Batched generation vs naive loop | T0 | `tools/nanochat test //src:generate_test` |
| Generation parity fixture | T1 | `tools/nanochat test --gpu //tests:generate_parity_test` |
| RL trajectory parity | T0/T2 | `tools/nanochat test //tests:rl_parity_test` |
| Base eval on a converted checkpoint | T2 | `tools/nanochat verify -- python -m nanochat_cpp.base_eval ...` |
| Chat eval subset | T2 | `tools/nanochat eval -- python -m nanochat_cpp.chat_eval ...` |
| RL smoke | T2 | `tools/nanochat train -- python -m nanochat_cpp.chat_rl ...` |

Fixtures are data: the GPU tests never import torch. All runs go through
`tools/nanochat` ([testing.md](testing.md), [sandbox.md](sandbox.md)).

## 9. Parity entries to record

Add to [parity.md](parity.md) as the work lands:

| ID | Status | Difference |
|---|---|---|
| E1 | `open` | C++ generation RNG differs from torch; only greedy is bit-identical (extends D4) |
| E2 | `open`/`closed` | calculator tool: absent (MVP) or arithmetic-only with C++ float formatting vs Python `repr` |
| E3 | `equivalent` | reference checkpoints converted torch -> NCHKPT01; dtype and name mapping |
| E4 | `equivalent` | CORE/chat scoring uses C++ forward with reference jinja/few-shot/task logic |
| E5 | `out-of-scope` | no DDP (extends D6); single-process evaluation and RL |
| E6 | `open` | SFT prerequisite satisfied by a converted reference checkpoint, not a native port |
| E7 | `open` | RL rollouts may lag one step if the file-based `rl_step` path is used instead of the worker |

## 10. Risks and open questions

- **Calculator fidelity.** Python `eval` float formatting (`str(10.0)` vs
  `"10"`) can change the tokens the model sees. If tool use is ported, gate it
  with a fixture and accept an `open` parity entry for formatting.
- **Reward without tool use.** If the model emits `<|python_start|>` blocks and
  no outputs are forced, completions get longer and `####` extraction may fail.
  Measure pass@k before committing to E8; the converter + SFT checkpoint may
  already produce well-formed answers.
- **KV cache memory.** Batched generation with `num_samples = 16` at
  `max_new_tokens = 256` on an 11 GB card needs a per-layer cache sized for
  `prompt + max_tokens` per row. Cap `num_samples` by `device-batch-size` as the
  reference does, and validate against the sandbox memory profile.
- **Optimizer state persistence.** `Checkpointer` currently saves model
  parameters only. The persistent worker keeps moments in memory; if a
  resumable RL run is required, add optimizer-state save/load to the container
  (a small additive change).
- **HumanEval execution.** Running generated code is the one place the bridge
  executes untrusted input. It must run inside the resource sandbox and use the
  reference `execute_code` guards; document that this is not a security sandbox.
- **CORE bundle.** `eval_bundle.zip` is a download; the fixture test must not
  depend on the network. Commit a tiny synthetic CORE fixture for the test and
  download the real bundle only for a T2 run.

## 11. Definition of Done

For each workstream, per [testing.md](testing.md): build warning-free, CPU tests
green, a small-shape GPU correctness test, a finite-difference check where a
backward changes, a fixture match within tolerance, and `clang-format` clean.
The overall plan is done when:

1. `python -m nanochat_cpp.base_eval --eval core,bpb,sample` runs on a converted
   reference checkpoint and matches the reference within tolerance.
2. `python -m nanochat_cpp.chat_eval` runs all five tasks and reports ChatCORE.
3. `python -m nanochat_cpp.chat_rl` runs a short RL run from an SFT checkpoint,
   and `//tests:rl_parity_test` matches the reference PG trajectory.
4. Every difference is recorded in [parity.md](parity.md) with a status.
