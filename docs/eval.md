# Evaluation

How `nanochat.cpp` measures a trained model. Evaluation is forward-only: it
reads a checkpoint, runs the inference graphs, and reduces the result to a
metric. It never computes gradients, never steps an optimizer, and never writes
a checkpoint. The base and chat evaluation scripts (`scripts/base_eval.py`,
`scripts/chat_eval.py`) are reproduced through the Python bridge.

This is a design document. The interface inventory is in
[model.md](model.md); the process seam is in [python-bridge.md](python-bridge.md);
known divergences are in [parity.md](parity.md).

## 1. Scope

Three families of measurement:

| Family | Reference | Metric |
|---|---|---|
| Bits per byte | `nanochat.loss_eval.evaluate_bpb` | tokenization-invariant cross-entropy on train/val shards |
| CORE | `nanochat.core_eval` + `eval_bundle.zip` | mean centered ICL accuracy over 22 tasks |
| Chat accuracy | `scripts/chat_eval.py` | ARC/MMLU categorical; GSM8K/HumanEval generative; ChatCORE |
| Samples | `scripts/base_eval.py --eval sample` | qualitative text |

Non-goals: training, optimizer state, distributed evaluation, and any metric
that requires gradients. Online metrics during training (val bpb, ChatCORE) are
the post-training loop's concern ([post-training.md](post-training.md)), but they
reuse the primitives here.

## 2. Architecture: metric in Python, compute in C++

Evaluation splits along the existing process seam:

- **Python** owns everything that is not tensor math: the eval bundle, jinja
  prompt rendering, few-shot sampling, task datasets, answer extraction, code
  execution, and metric reduction (mean-loss argmin, centering, ChatCORE).
- **C++** owns the model compute behind three primitives.

| Primitive | Shape of the work | Drives |
|---|---|---|
| `EvalBpb` (exists) | forward cross-entropy over token shards, sum nats / sum bytes | bpb |
| `ScoreBatch` (new) | forward over a padded token batch; emit per-position NLL and argmax | CORE, categorical chat |
| `GenerateBatch` (new) | prefill + batched decode with per-row stop and masks | samples, generative chat |

The C++ side is tokenizer-agnostic. The bridge tokenizes with the reference BPE
tokenizer and passes token ids, start/end indices, focus token ids, and stop
token ids as data or CLI arguments. No tokenizer, jinja, pyarrow, or torch enters
the C++ runtime.

`ScoreBatch` and `GenerateBatch` are the same primitives the post-training loop
uses (for scoring and rollouts respectively); they live in the model layer, not
in evaluation.

## 3. Compute primitives

### 3.1 Bits per byte

`EvalBpb(model, loader, steps)` already exists and matches
`evaluate_bpb`. It sums per-token NLL over `steps` batches, sums the source-byte
length of each target token from the loader's byte table, and returns
`total_nats / (ln(2) * total_bytes)`. Special tokens and masked targets have zero
bytes and are skipped. No change is needed beyond exposing train and val splits
and mapping the reference's `--split-tokens` to a step count.

### 3.2 `ScoreBatch`

A single forward over a batch of independent token sequences, returning the
per-position NLL and the argmax prediction at every position. It is the compute
half of CORE and of categorical chat evaluation.

Design constraints:

- Input is a fixture of `N` padded sequences plus, per sequence, a candidate
  span `[start, end)` and optionally a focus set of token ids.
- One `ForwardLoss` call with shifted targets; masked positions use the ignore
  index. The saved per-token losses (`TrainModel::losses`) and the argmax of
  `raw_logits` are staged to the host.
- Output is per-position NLL and argmax, plus the focused logits when a focus set
  is given (categorical chat only reads the letter tokens at the answer
  position).
- The binary is `score_main`; it is stateless and writes a result file. The
  fixture format is owned by the bridge (`eval_fixture.py`) and mirrored by the
  test dumper.

The reduction is deliberately *not* in C++. Whether a CORE example is correct is
"lowest mean loss among candidates" or "exact argmax match"; whether an ARC
example is correct is "argmax over the answer letters". Those semantics stay in
Python next to the reference implementation.

### 3.3 `GenerateBatch`

Batched sampling: prefill once, clone the KV cache per sample, decode in
lockstep, stop each row on its own terminal token, and return token ids plus a
per-position `1`/`0` mask. See [model.md](model.md) for the graph and the
sampling parameters. Evaluation uses greedy decoding (temperature 0) for
deterministic metrics and a fixed seed for samples.

## 4. Base evaluation

`base_eval.py` runs up to three modes (`--eval core,bpb,sample`).

### 4.1 Bits per byte

For each of the train and val splits: materialize or reuse the shard, build a
`DataLoader`, and call `EvalBpb`. The reference's `--split-tokens` is converted
to `steps = split_tokens / (device_batch_size * seq_len)`. Result:
`{split: bpb}`.

### 4.2 Samples

The bridge holds the fixed prompt list, tokenizes each with a leading
`<|bos|>`, and calls `GenerateBatch`: greedy for the conditioned prompts, and
temperature 1.0 with `num_samples = 8` for the unconditioned case. The bridge
decodes and prints. This is the only base-eval mode whose output is text rather
than a number.

### 4.3 CORE

CORE is the largest piece. The bridge reproduces `evaluate_core` exactly; only
the forward pass is C++.

1. **Bundle.** Ensure `eval_bundle.zip` is present and unzipped under the base
   directory (network on first use only).
2. **Tasks.** Read `core.yaml` for the ICL task list, `eval_meta_data.csv` for
   the random baselines, and the per-task JSONL from `eval_data/`.
3. **Few-shot.** Shuffle each task's data with `random.Random(1337)`, then for
   each evaluated example sample `num_fewshot` other examples with
   `random.Random(1234 + idx)`.
4. **Render.** Three task types, each with a jinja template:
   - `multiple_choice`: contexts share a prefix, continuations differ.
   - `schema`: contexts differ, the continuation is shared (common suffix).
   - `language_modeling`: two prompts, without and with the continuation.
   The bridge tokenizes each candidate and computes the candidate span from the
   common prefix or suffix.
5. **Score.** Pack the sequences into a `ScoreBatch` fixture and run
   `score_main`.
6. **Reduce.** MC and schema pick the candidate with the lowest mean NLL over
   the span; LM requires every argmax prediction in the span to match. Center
   with `(accuracy - 0.01 * baseline) / (1 - 0.01 * baseline)`; CORE is the mean
   over tasks. Write the CSV in the reference format.

A tiny synthetic CORE fixture is committed so the T0 test never needs the
network or the real bundle.

## 5. Chat evaluation

`chat_eval.py` runs five tasks and reports per-task accuracy and ChatCORE.

### 5.1 Categorical (ARC-Easy, ARC-Challenge, MMLU)

The bridge renders the multiple-choice prompt (`render_mc`), tokenizes, and
finds the answer position (the last prompt token) and the token id of each
answer letter. It batches problems, calls `ScoreBatch` with the focus set, and
argmaxes the focused logits. `ScoreBatch` returns logits only at the requested
position and token ids, so the bridge never moves full `(B, T, V)` tensors.

The letter token ids are cached; each letter is asserted to be a single token,
as in the reference.

### 5.2 Generative (GSM8K, HumanEval)

The bridge renders the completion prompt, calls `GenerateBatch` with the
configured temperature, top-k, sample count, and `max_new_tokens`, then decodes
and evaluates:

- **GSM8K**: extract the number after `####` from the completion and compare to
  the reference answer.
- **HumanEval**: extract the first code block, append the test harness, and run
  it through the reference `nanochat.execution.execute_code`. This is the one
  place evaluation executes generated code; it runs inside the resource sandbox
  and uses the reference guards (fresh interpreter, rlimits, scrubbed
  environment, timeout). It is not a security boundary against adversarial code.

### 5.3 ChatCORE

When all five tasks ran, ChatCORE is the mean centered accuracy with baselines
ARC/MMLU 0.25 and GSM8K/HumanEval 0.0. The categorical subset is reported
separately, mirroring the post-training loop's logging.

## 6. Checkpoints and the tokenizer

- Evaluation loads a checkpoint by `source` (`base`/`sft`/`rl`), model tag, and
  step. C++-trained checkpoints are NCHKPT01 and load directly.
- Reference checkpoints are torch `.pt` state dicts. A converter
  (`checkpoint.py`) remaps names and dtypes to NCHKPT01 so released nanochat
  models can be evaluated. This is the only path by which reference weights enter
  the runtime.
- The tokenizer is the reference BPE artifact, loaded in Python. The C++ runtime
  receives only integer ids.

## 7. Invariants and non-goals

- Forward-only: no gradients, no optimizer, no checkpoint writes.
- Deterministic where the reference is deterministic (greedy decoding, fixed
  few-shot seeds); sampled modes are seeded.
- Tokenizer-agnostic C++: ids in, ids and logits out.
- Single process; no DDP ([parity.md](parity.md) D6).
- The CORE bundle and task datasets are downloaded by the bridge, never by a C++
  binary; committed fixtures keep the tests hermetic.

## 8. Parity

| ID | Status | Difference |
|---|---|---|
| E3 | `equivalent` | reference checkpoints converted torch -> NCHKPT01 |
| E4 | `equivalent` | CORE/chat scoring uses the C++ forward with reference task logic |
| E1 | `open` | sampled modes use the C++ RNG, not torch (greedy is bit-identical) |
| E5 | `out-of-scope` | no distributed evaluation |

## 9. Status and roadmap

All of the evaluation workstream described in this document is implemented:

1. `checkpoint.py` converter so a reference checkpoint can be loaded.
2. `ScoreBatch` + `score_main` + the fixture format.
3. `GenerateBatch` ([model.md](model.md)).
4. `base_eval.py` (bpb, sample, CORE).
5. `chat_eval.py` (categorical, generative, ChatCORE).

The differences that remain are tracked in [parity.md](parity.md): E3
(reference checkpoints converted torch -> NCHKPT01) and E4 (CORE and chat
scoring use the C++ forward with the reference task logic) are `equivalent`;
E1 (the sampled modes use the C++ RNG rather than torch, while greedy decoding
is bit-identical) is `open`; E5 (no distributed evaluation) is `out-of-scope`
for the single-GPU target. No evaluation item remains on the roadmap.

Gates: a synthetic CORE fixture at T0; a converted-checkpoint bpb match and a
100-example CORE match at T2; a generation parity fixture at T1. See
[testing.md](testing.md) for the test tiers.
