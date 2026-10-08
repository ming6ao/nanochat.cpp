# Reference parity: known differences from PyTorch nanochat

`nanochat.cpp` reimplements the architecture and training recipe of the PyTorch
reference (`scripts/base_train.py`, `nanochat/gpt.py`, `nanochat/optim.py`,
`nanochat/dataloader.py`). This document tracks every known difference between
the two so that a divergence in a loss curve, a metric, or a parameter can be
attributed quickly. It is the companion to
[testing.md](testing.md) (how parity is gated) and
[python.md](python.md) (how the reference is driven).

Status values:

| Status | Meaning |
|---|---|
| `closed` | fixed; no remaining difference |
| `equivalent` | a different implementation of the same mathematics, up to floating-point order |
| `open` | a real behavioral difference that can change a loss curve or a metric |
| `out-of-scope` | intentionally not implemented on this host |

## Summary

| ID | Status | Area | Difference |
|---|---|---|---|
| D1 | `closed` | data pipeline | BOS-aligned best-fit packing matches the reference |
| D2 | `equivalent` | attention | fused RMSNorm→RoPE order vs RoPE→RMSNorm |
| D3 | `equivalent` | attention | cuBLAS GEMM + softmax kernels vs SDPA math backend |
| D4 | `equivalent` | initialization | custom xorshift RNG vs the PyTorch RNG |
| D5 | `equivalent` | optimizer | gradient norm computed for logging with `--clip 0` |
| D6 | `open` | scaling | distributed data parallel training is absent |
| D7 | `out-of-scope` | runtime | no `torch.compile`; disabled on Pascal anyway |
| D8 | `equivalent` | harness | no batch prefetch overlap during backward |
| D9 | `closed` | optimizer | C++ schedule warmdown count matches Python's `round` (half to even) |
| E1 | `open` | evaluation | sampled modes use the C++ RNG, not torch; greedy is bit-identical |
| E3 | `equivalent` | evaluation | reference checkpoints converted torch -> NCHKPT01 |
| E4 | `equivalent` | evaluation | CORE and chat scoring use the C++ forward with reference task logic |
| E5 | `out-of-scope` | evaluation | no distributed evaluation |
| P1 | `open` | post-training | calculator tool absent in the MVP; the model samples its own python blocks |
| P2 | `open` | post-training | file-driven `rl_step` makes the rollouts lag one optimizer step |
| P3 | `equivalent` | post-training | SFT starts from a converted reference base checkpoint |
| T1 | `equivalent` | tokenizer | native byte pair encoding trainer and encoder vs `rustbpe` + `tiktoken` |
| T2 | `equivalent` | tokenizer | portable `NCTOKEN1` artifact and `int64` pair counts vs the reference pickle and `int32` |
| T3 | `open` | tokenizer | stream decode buffers an incomplete UTF-8 suffix instead of one U+FFFD per token |

## How parity is verified

Two data fixtures pin the model and optimizer against the reference:

- `tests/debug_state.bin` (forward, backward, and a few optimizer steps) via
  `tools/dump_oracle.py`.
- `tests/data/train_parity.bin` (a full training trajectory: initial
  parameters, the exact token/target batch at every step, and the per-step
  loss, gradient norm, and parameter norms) via `tools/dump_train_fixture.py`.

Both fixtures **share the initial parameters and the exact batches** with the
C++ side, so they isolate the model, backward pass, optimizer, and schedules
from the data pipeline. That is deliberate: it means a failure points at the
graph, and it also means D1 is *not* covered by these gates.

```bash
tools/nanochat test //tests:oracle_test //tests:train_parity_test
tools/nanochat test --gpu //tests:oracle_cuda_test //tests:train_parity_cuda_test
```

A third fixture pins the tokenizer. `tests/data/tokenizer_fixture.bin` records
the reference merges, token bytes, split cases, encode cases, and decode cases
from `tools/dump_tokenizer_fixture.py`. The two C++ tests match it on the CPU and
never import Python:

```bash
tools/nanochat test //src/tokenizer:tokenizer_parity_test \
    //src/tokenizer:bpe_trainer_test
```

## The staged numeric-trace contract (Path A)

[numerics-integration.md](numerics-integration.md) defines the staged precision
contract of the numeric integration test. That document calls the design Path A.
`//tests:numerics_trace_test` and its GPU sibling
`//tests:numerics_trace_cuda_test` gate the contract against one committed
golden file, `tests/data/numerics_golden_10l.bin`.

The plan states the reduction in section 1 and the four clauses in section 3.
The gate applies them as follows:

1. The CPU leg and the simulator leg compare byte for byte.
2. The CPU leg and the simulator leg repeat across runs. The GPU leg repeats
   within the tolerance table, because the atomic scatter-adds make the byte
   order a non-contract.
3. The GPU leg compares the scalars within the tolerance table of section 4.
   It skips the parameter hashes and the gradient hashes.
4. The GPU leg compares the greedy identifiers exactly, and every recorded
   logit margin must stay above the tie threshold of section 6.4.

The GPU leg also runs the trace twice and compares the two runs, which gates
the determinism clause.

The contract does not cover the SFT loop or the RL loop. It covers the SFT
arithmetic and the RL arithmetic only, as the plan states. The status table in
[post-training.md](post-training.md) still marks the SFT renderer, the SFT
packer, the SFT loop, the `rl_step` binary, and the RL fixture as **Missing**.

## Open differences

### D1 — Data loader packing and ordering

**Status:** `closed`. The document-mode `DataLoader` in `src/data.cc`
implements the reference packing. `//tests:loader_parity_test` pins the C++
rows against the reference rows on a fixed corpus.

**Reference** (`nanochat/dataloader.py`,
`tokenizing_distributed_data_loader_with_state_bos_bestfit`):

- BOS-aligned best-fit packing: every row starts with BOS; documents are placed
  largest-first; when nothing fits, the shortest buffered document is cropped
  to fill the row exactly. Row capacity is `T + 1`, so `inputs = row[:-1]` and
  `targets = row[1:]` are independent per row.
- Documents are iterated sequentially over parquet row groups, sharded across
  ranks and cycling epochs.
- Roughly 35% of tokens are discarded to cropping at `T = 2048` (higher at
  shorter `T`), which means more unique documents are consumed per step.

**nanochat.cpp** (`src/data.cc`, `DataLoader`):

- `DataLoader` reads raw documents from a `DocumentSource` and tokenizes them
  with the reference encoder ([parquet-native.md](parquet-native.md)).
- A producer thread refills a document buffer to `document_buffer`, the
  consumer picks the largest document that fits, and crops the shortest
  document when none fits. Row capacity is `T + 1`.
- Documents are iterated sequentially over parquet row groups. `Reset` re-opens
  the source, so a training run cycles epochs.

**Impact:** none. The packing is the reference packing.

**Evidence:** `//tests:loader_parity_test` compares the C++ rows to a fixture
that `tools/dump_loader_fixture.py` produced from the reference loader. The
fixture shares the documents, the tokenizer, the buffer size, and the source
batch size with the C++ side.

### E1 — Sampled evaluation RNG

**Status:** `open`. Evaluation runs its generative modes through the C++
`GenerateBatch` primitive ([eval.md](eval.md) §4.2, §5.2), so they draw from the
xorshift64\* source tracked in D4 rather than from the PyTorch global RNG the
reference uses. Greedy decoding is bit-identical: it is an argmax with no random
draw, and the generation parity fixture (`tests/data/generate_parity.bin`,
`//tests:generate_parity_test`) pins it on the CPU and CUDA backends. The
sampled paths are:

- The base-eval sample mode: the unconditioned prompts at temperature 1.0 with
  `num_samples = 8` (the conditioned prompts stay greedy).
- The `ChatEvaluator` generative tasks (GSM8K, HumanEval): the configured
  temperature, top-k, and sample count.

**Impact:** the sampled text differs from the reference, so GSM8K and HumanEval
pass rates and sample outputs are not run-for-run comparable. The deterministic
metrics (bpb, CORE, categorical chat) are unaffected.

**Evidence:** the generation parity fixture covers greedy decoding only; the
sampled outputs depend on the torch RNG state, so there is no committed fixture
for them. The T2 chat evaluation gate compares per-task accuracy on a fixed
subset, not the sampled text.

**Fix direction:** none required for parity of the deterministic metrics.
Reproducing the reference sampled text exactly would mean reimplementing the
PyTorch RNG and its state layout, which is not worth it. Treat sampled metrics
as nanochat.cpp's own seeded distribution.

### P1 — Calculator tool

**Status:** `open`. The reference `scripts/chat_rl.py` runs a calculator tool
inside the generation loop. The engine detects a python block between
`<|python_start|>` and `<|python_end|>`, evaluates the expression, and forces
the result tokens. The `GenerateBatch` primitive never decodes text, because it
is tokenizer-agnostic. The MVP cannot force the tool output, so the model
samples its own python blocks. The bridge reads the reward from the final
`####` answer.

**Impact:** the RL reward and the GSM8K pass rate can differ from the
reference. The reference supplies exact arithmetic results. The MVP relies on
the arithmetic of the model.

**Evidence:** none yet. The RL parity fixture (`post-training.md` section 12)
is the gate.

**Fix direction:** the streaming `chat_engine` protocol in `post-training.md`
section 5.1. The bridge receives one token column at a time, decodes python
blocks, and sends the forced tokens back.

### P2 — File-driven RL step

**Status:** `open`. The reference `scripts/chat_rl.py` generates a rollout with
the current weights, computes the reward, and steps the optimizer in one
process. The nanochat.cpp correctness gate is a file-driven `rl_step` binary
that consumes a fixture. The binary reads a recorded rollout and a recorded
advantage, so the rollout lags the optimizer by one step. The persistent worker
removes the lag.

**Impact:** the RL trajectory can differ from the reference. The policy that
produced the rollout is one step behind the policy that takes the update.

**Evidence:** the RL parity fixture is the gate (`post-training.md` section
12). It is not in the tree yet.

**Fix direction:** the persistent worker in `post-training.md` section 8. The
worker holds the model and the optimizer and exchanges rollout and advantage
messages with the bridge over pipes.

### T3 — Stream decode of an incomplete UTF-8 sequence

**Status:** `open`. `tiktoken` decodes each token by itself. A token that holds
a partial UTF-8 sequence becomes one replacement character, U+FFFD, and the next
token cannot repair it. nanochat.cpp's `Tokenizer::Decode` copies that behavior
exactly, so a per-token consumer is bit-identical.

The generation path uses `TokenStreamDecoder` instead. `Push` holds an
incomplete UTF-8 suffix until the next token completes it. `Flush` emits the
held bytes with replacement at the end of a row. The 32768 vocabulary has 74
merges with this property, for example the two bytes `\xe2\x80`.

**Impact:** the streamed text of a sampled row can differ from the reference text
when a row ends in the middle of a UTF-8 sequence. The deterministic metrics are
unaffected: bits-per-byte tokenizes the bytes, and CORE and categorical chat
scoring use token ids and logits. The sampled text is not a committed metric,
the same position as E1.

**Evidence:** `//src/tokenizer:tokenizer_parity_test` covers both modes. It checks `Decode`
per token against the reference and checks that `Push` repairs a split sequence.
The generate parity fixture is greedy and token-level, so it does not compare
decoded text.

**Fix direction:** none required. Exact stream parity would mean re-emitting the
stray U+FFFD for every partial token. The buffer is the documented local choice
(`docs/tokenizer.md` section 8.3). A consumer that needs exact reference text
calls `Decode` per token.

## Equivalent differences

### D2 — QK-norm and RoPE order

The reference applies RoPE, then RMSNorm over `head_dim`, then a `1.2` scale
(`CausalSelfAttention.forward`). nanochat.cpp fuses the three as
RMSNorm→RoPE→scale in `QkPrepForward` (`src/ops.cc`,
`backends/cuda/kernels/qk_prep.cu`). The two orders are mathematically
identical: RoPE is an orthogonal rotation within `head_dim` and RMSNorm is a
scalar rescale, so `RMSNorm(R x) = R · RMSNorm(x)`. Only floating-point
rounding differs. See [kernels.md](kernels.md).

### D3 — Attention backend

The reference uses the SDPA math backend on Pascal (`nanochat/flash_attention.py`
falls back from Flash Attention 3, which needs SM 80+). nanochat.cpp computes
the scores, the probability-value product, and every backward product as
batched cuBLAS GEMMs, with the softmax and its gradient as custom row kernels
(`backends/cuda/kernels/attention.cu`). Both compute the same causal, optionally
sliding-window, grouped-query softmax attention with the same mask and the same
statistics contract; only the summation order differs. See
[performance.md](performance.md).

### D4 — Initialization RNG

The reference initializes with PyTorch's global RNG; nanochat.cpp uses a
xorshift64\* source in `src/model.cc` (`InitWeights`). The **distributions** are
identical (verified against `GPT.init_weights`): `wte` normal std 0.8, `lm_head`
normal std 0.001, attention projections uniform with standard deviation
`1/sqrt(hidden)`, `c_proj` zero, `mlp.c_fc` at 0.4× that scale, and the scalar,
smear-gate, value-embedding, and value-gate initializations all match. The
**concrete values** differ, so the same seed does not reproduce the reference
weights; the parity fixtures sidestep this by sharing the initial parameters.

### D5 — Gradient norm during the optimizer step

The reference does no gradient clipping in `scripts/base_train.py` and does not
compute a norm. nanochat.cpp's optimizer computes a global norm every step
(`kernels::GlobalNorm` with `clip = 0`) so the logger can report it. This is
numerically the identity (the clip scale is 1) and does not change the update;
it does add one device-to-host synchronization per step. It is not a
training-math difference.

### D8 — Batch prefetch overlap

The reference prefetches the next batch while the GPU runs the backward pass
(`x, y, ... = next(train_loader)` inside the accumulation loop). nanochat.cpp
loads the next batch at the top of the next step. This is a throughput
difference only; the sequence of batches presented to the model is the same.

### D9 — Warmdown iteration count rounding

**Status:** `closed`. The reference `scripts/base_train.py` computes the
warmdown length with Python's `round`. Python rounds half to even, so
`round(2.5)` is 2. `Scheduler::WarmdownIters()` in `src/optim.cc` used
`std::lround`, which rounds half away from zero, so `std::lround(2.5)` is 3.
The two forms differ only when `warmdown_ratio * num_iterations` is exactly a
half-integer.

A 5-step `tools/dump_train_fixture.py` run at the section 6.2 shape uses the
default ratio `0.5`, so the product is `2.5`. The C++ warmdown then started one
step early and changed the learning rate and the Muon momentum at the last two
steps. The loss error was `0.146` and the parameter L2 error was `0.372`
against the PyTorch reference. The fixture is not committed; regenerate it with
`tools/dump_train_fixture.py --steps 5`.

`WarmdownIters()` now uses `std::nearbyint` under the default `FE_TONEAREST`
mode, and it evaluates the product in `double`. This matches Python's `round`.
The regression test is `TestWarmdownRounding` in `src/optim_test.cc`. Evidence:
`//tests:train_parity` at the section 6.2 shape passes with a loss error of
`9.54e-07`.

### E3 — Reference checkpoint conversion

The reference saves checkpoints as torch `.pt` state dictionaries; the C++
runtime loads only the NCHKPT01 container. `tools/convert_checkpoint.py` and
`python/nanochat_cpp/checkpoint.py` remap the reference parameter names and
apply the dtype policy so a released base or SFT checkpoint can be loaded by
`EvalBpb` and `score_main`. The conversion re-lays the reference weight values
into the container without recomputing them, so the model being evaluated is the
reference model; the only arithmetic difference is the forward-pass rounding
already tracked in D2 and D3. See [eval.md](eval.md) §6. A converted `d8`
checkpoint reproduces the reference bits-per-byte within the T2 tolerance.

### E4 — CORE and chat scoring

The reference computes CORE (`nanochat.core_eval`) and categorical chat scoring
with the torch forward and keeps every task decision in Python. nanochat.cpp
runs one `ForwardLoss` over a padded batch in `score_main` (`ScoreBatch`) and
returns per-position NLL, argmax, and focused logits; the bridge reproduces the
reference task logic unchanged: the `random.Random(1337)` and
`random.Random(1234 + idx)` few-shot sampling, the jinja rendering and candidate
spans, the lowest-mean-NLL argmin and exact-argmax match, the
`(accuracy - 0.01 * baseline) / (1 - 0.01 * baseline)` centering, and the
ChatCORE mean. Only the forward arithmetic differs, in the same
floating-point-order sense as D2 and D3; each decision is the same function of
the logits. See [eval.md](eval.md) §4.3 and §5.

### P3 — SFT base checkpoint provenance

The reference `scripts/chat_sft.py` loads its base model from the reference
`base_checkpoints` directory. A reference-comparable nanochat.cpp SFT run loads
the same base model through the conversion in E3. The conversion re-lays the
reference weights without recomputation, so the SFT start is the reference
start. The optimizer-state section carries the reference moments, so the SFT
warm-start uses the same starting state. The C++ `train_main` can also pretrain
a native base checkpoint, and that path is not the parity prerequisite.

**Impact:** none. The SFT start is the reference start.

**Evidence:** E3 and `//src:optimizer_state_test`. The SFT loop is not in the
tree yet (`post-training.md` section 11).

### T1 — Native tokenizer reimplementation

The reference trains the vocabulary with `rustbpe`. It encodes and decodes with
`tiktoken`. nanochat.cpp reimplements both in C++: `src/tokenizer/bpe_trainer.cc`,
`src/tokenizer/split_pattern.cc`, and `src/tokenizer/tokenizer.cc`. The two
implementations are the same function.

The trainer pops the same pair: the largest count first, the smallest pair on a
tie, and the left id before the right id. The merge index gives the same token
id. The encoder merges the lowest-rank pair. The Unicode tables use Unicode
16.0.0, the version inside `tiktoken`.

The reference fixture pins the merges rank by rank, the token bytes, the split
cases, the encode cases, and the decode cases. The CPU tests
`//src/tokenizer:tokenizer_parity_test` and `//src/tokenizer:bpe_trainer_test`
match the fixture. The port is equivalent, not merely close.

### T2 — Portable artifact and wider pair counts

The reference saves the tokenizer as a `pickle` plus a `torch` byte-length
tensor. nanochat.cpp defines the little-endian `NCTOKEN1` container
([tokenizer.md](tokenizer.md) section 5). The artifact needs neither `tiktoken`
nor `torch` to load.

The byte content is the same. The loader derives the token bytes from the ordered
merge pairs. The native trainer writes the pairs in the reference rank order.

The trainer counts pairs with `int64` instead of the reference `int32`. The wider
type changes no result below the `int32` limit. The fixture-scale corpora stay
far below it. This is a container and integer-width difference only.

### D6 — Distributed data parallel training

**Status:** `open`. The reference shards documents by rank, reduces the
gradients with an all-reduce, and makes the ranks agree on `inf` and `nan`
handling. `nanochat.cpp` runs one process. The project now targets two T4 cards
in a Kaggle Notebook, so data parallel training enters the scope. The design is
in [distributed-design.md](distributed-design.md).

The tree implements the C++ host reference and the harness integration for
architecture support. The model's loss is a mean, so the harness scales each
local backward pass by `1 / world_size` and the all-reduce sums the result. The
global gradient becomes the mean over the global batch, which matches the
one-rank update.

One difference stays open. `nanochat.cpp` shards documents by a stride. The
reference splits one global token batch into contiguous rank slices. The stride
re-partitions the same document pool, so a two-rank run and a one-rank run do
not produce the same tokens in one step. The CPU gate pins the mean scale and
the equality of the two ranks. The validation-loss equality needs the P3 and P4
work.

The NCCL path and the Kaggle notebook stay in P3. A resumed run restarts the
epoch until the checkpoint stores the document index (P4).

## Out of scope

### D7 — `torch.compile`

The reference compiles the model and the fused optimizer steps. On Pascal this
is disabled (`TORCH_COMPILE_DISABLE=1`; Triton needs SM 70+), so it has no
effect on the current host and is not a parity item here. On newer hardware it
would fuse elementwise work and change host dispatch cost, not the mathematics.

### E5 — Distributed evaluation

The reference shards evaluation across ranks (each rank scores a slice of the
token shard or the task set) and reduces the metric. nanochat.cpp evaluates in a
single process, so the evaluated token count and the task count are bounded by
one device. This is the evaluation-side counterpart of D6 and is out of scope
for the single-GPU target.

## Adding a difference

When you find a divergence, add a row to the summary table and a section with:
the reference location, the nanochat.cpp location, the observed impact, the
evidence (which gate does or does not cover it), and the fix direction. Prefer
`equivalent` only when the two forms are provably the same function; otherwise
mark it `open` until measured.
