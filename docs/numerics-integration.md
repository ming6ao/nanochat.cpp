# Numerics integration test: CPU, simulator, and Pascal GPU

This document defines an integration test for numeric agreement across three
targets. The three targets are the CPU backend, the CUDA simulator, and the
Pascal GPU. The test uses a 10-layer model. It covers training, evaluation,
supervised fine-tuning (SFT), and reinforcement learning (RL).

The test follows Path A. Path A is a staged precision contract. This document
also records what Path A does not do.

Status: phases 0 to 5 are complete. Phase 6 steps 1 and 2 are complete. Phase 6
step 3 stays open: the optional `--trace` option of the harness. The trace
library, the emitter, the two Bazel targets, the fixture, the simulator wiring,
and the merge-gate entry are present. The fixture holds one section per
precision.

## 1. The reduction, stated first

Path A does not compare floating-point results across targets. It compares the
results that two arithmetic implementations can agree on. State them before the
design.

1. Every leg compares the fixture byte for byte. There is one comparison mode,
   and a difference is a failure.
2. The fixture holds the **exact class** only: configuration, inputs, token ids,
   masks, greedy identifiers, argmax, and valid-target counts. Every one of them
   is an integer or a fixed constant.
3. The fixture carries one section per precision (`fp32/`, `bf16/`, `fp8/`), and
   each build compares the section that names its own precision.
4. The caps invariant is exact, and `//tests:sim_numerics_test` proves it in one
   process. That test changes `NANOCHAT_SIM_PROFILE` and requires every recorded
   value to be bit-identical. It is the guard that no numeric path reads
   `GetCaps()`.
5. Floating-point results are not records. Their contract is the per-stage
   parity tests of [testing.md](testing.md), each with one bound and one
   committed fixture.
6. The device leg repeats across two runs and compares the discrete records
   again. The device kernels use atomic scatter-add operations, so a byte order
   is not a contract, but a discrete result is.

The status table in [post-training.md](post-training.md) marks the SFT renderer,
the SFT packer, the SFT loop, the `rl_step` binary, and the RL fixture as
**Done**. This trace gives kernel-level coverage of the discrete results only.
The SFT arithmetic evidence is `//tests:sft_step_parity_test`, not this test.

## 2. What the recon established

Five facts shape the design.

1. The CPU backend uses `double` accumulators. See the GEMM microkernel in
   `backends/cpu/kernels.cc`. The Pascal kernels use `float` accumulators, and
   GEMM calls cuBLAS. [parity.md](parity.md) records this difference as `D2`
   and `D3` with status `equivalent`.
2. Under `--config=sim`, the CPU loops compute every value. The cap override is
   the only change. `tests/sim_numerics_test.cc` asserts the invariant. The
   "simulator" leg is therefore a cap-override regression guard. It is not a
   third numeric implementation.
3. No test compares two backends against each other. Every parity gate
   compares one backend against a committed fixture. See [testing.md](testing.md).
4. The CUDA device kernels use atomic operations. The attention key and value
   gradients, the embedding scatter-add, the value-gate weight gradient, and
   the block combine all use atomics. Atomic order is not a contract.
5. A bf16 or fp8 build rounds the logits to 7 mantissa bits, so the top two
   logits can tie exactly. The fixture therefore carries one section per
   precision, and each build compares its own section.

One more fact fixes the architecture. Bazel selects the backend at link time
(see `src/BUILD.bazel`). The CPU backend and the CUDA backend define the same
`nanochat::kernels::*` symbols. One process cannot hold both. Each leg compares
its own section of one committed fixture.

## 3. The contract

The contract has four clauses.

1. Each leg equals its fixture section, byte for byte.
2. The device leg reproduces its section across two runs.
3. The greedy token identifiers and the argmax equal the section on every
   target.
4. The caps invariant is exact: a simulated profile changes no value.

Clause 3 uses parity entry `E1`. [parity.md](parity.md) records greedy
generation as bit-identical. Clause 3 holds only for a peaked model head.
Section 6.3 defines the tie rule.

## 4. Record classes and where the bounds live

The fixture has one class of record, and two kinds of value that are not
records.

| Value | Where it lives | Comparison |
|---|---|---|
| Configuration, inputs, token ids, masks, greedy identifiers, argmax, valid-target counts | The fixture | Byte for byte, on every backend and every precision |
| Logit margins | The run, printed | The tie rule of section 6.3. The fp32 leg fails below `1e-5` |
| Loss, gradient norm, parameter L2, bits per byte, reward, advantage | The parity fixtures | One bound per test, below |

| Artifact | Test | Bound |
|---|---|---|
| Train loss, gradient norm, parameter L2 | `//tests:train_parity_test` | [testing.md](testing.md) |
| The same on the device | `//tests:train_parity_cuda_test` | [testing.md](testing.md) |
| Evaluation bits per byte and per-batch loss | `//tests:score_parity_test` | [testing.md](testing.md) |
| SFT loss, gradient norm, parameter L2, learning rate, momentum | `//tests:sft_step_parity_test` | [testing.md](testing.md) |
| RL reward, advantage, loss, gradient norm | `//tests:rl_parity_test` | Exact against `tests/data/rl_parity.bin` |
| The fp8 floating-point bounds | The fp8 parity tests | [simulator-precision.md](simulator-precision.md) section 7 |

This gate applies no numeric tolerance. It has no per-group table, no skip, no
warning path, and no host pin. A record mismatch is a failure.

## 5. Test architecture: fixture replay

The test follows the existing `oracle_test` pattern. One source file builds as
two Bazel targets. One committed fixture is the reference.

- The **fixture** `tests/data/numerics_golden_10l.bin` comes from the CPU
  emitter. A reviewer checks the file. A developer commits the file. The file
  does not need PyTorch at runtime.
- The **CPU target** `//tests:numerics_trace_test` runs the model, then
  compares its section against the fixture in process.
- The **simulator leg** runs the same target under `--config=sim`.
- The **CUDA target** `//tests:numerics_trace_cuda_test` shares the source and
  carries the tag `gpu`.
- The **comparison runs in process**, by record name.

The fixture is portable by construction. Every record is an integer or a fixed
constant, so the same bytes come out of any host. A re-emission that differs is
a real change, not a host difference. The generator checks this: a re-emission
on this host reproduced the committed sha256 byte for byte.

### 5.1 Fixture format

The fixture reuses the little-endian record layout of `src/container.h` under
its own magic `NANONUM1`. The container is `[8-byte magic][u32 version][u32
count]`, then one record per `[u16 name length][name][u8 dtype][u8 width][u64
shape...][payload]`.

Every record name carries its section as a prefix. The fixture holds the
sections in the order `fp32`, `bf16`, `fp8`, and each section carries the same
63 names in the same order.

| Record group | Names |
|---|---|
| Identity | `header/version`, `header/magic`, `header/seed`, `header/steps` |
| Configuration | `config/layers`, `config/heads`, `config/kv_heads`, `config/hidden`, `config/seq`, `config/batch`, `config/vocab`, `config/padded_vocab`, `config/window_pattern`, `config/value_embedding`, `config/rope_base`, `config/parameters` |
| Optimizer and schedule | `opt/*`, `sched/*` |
| Phase inputs | `train/batches`, `eval/steps`, `gen/prompt_len`, `gen/tokens`, `gen/num_samples`, `sft/ignore_every`, `rl/num_samples`, `rl/prompt_len`, `rl/tokens`, `rl/num_passes`, `rl/examples_per_rank`, `rl/reward_rule` |
| Evaluation | `eval/tokens/<b>`, `eval/argmax/<b>` |
| Generation | `gen/prompt`, `gen/ids`, `gen/mask` |
| SFT | `sft/valid_targets`, `sft/zero_weights` |
| RL | `rl/prompt/<r>`, `rl/ids/<r>`, `rl/valid_targets` |

The trace pins the host thread count to one. The CPU reference reductions depend
on the thread count, and a pinned count keeps a re-emission stable.

The schema check runs before any value is compared. It requires all three
sections to exist, requires the three suffix lists to be identical, and requires
the dtype and the shape of every record to match the run.

### 5.2 Re-emission

One command builds the emitter under each precision and writes the single file.

```bash
python3 tools/dump_fixtures.py numerics
```

The generator runs the emitter through `tools/nanochat run t0-cpu --`, prefixes
each record name with its section, and writes the container. `--check`
regenerates into a temporary directory and compares the bytes with the committed
file. The manifest records the size, the SHA-256, the FNV-1a digest, the record
count, the record names, and the producer facts.

A change to the graph, the configuration, the seed, or the tokenizer changes the
fixture. Re-emit it in the same change, and update the manifest and
`tolerances.json` with it. A change that moves only the floating-point results
does not touch this fixture.

## 6. Trace contents and model configuration

### 6.1 Trace contents

The trace drives five optimizer steps, then records the discrete results of the
four phases.

- The header holds the magic string, the seed, and the step count.
- The configuration holds the `Config` values and the parameter count.
- The optimizer and schedule groups hold the reference recipe as data.
- The evaluation section holds the token ids of two batches and their argmax.
- The generation section holds the prompt, the greedy identifiers, and the mask.
- The SFT section holds the valid-target count and the zero-weight count.
- The RL section holds four rollouts and the valid-target count.

### 6.2 Reference configuration

The host is one GTX 1080 Ti with 11 GB of memory. The device is `sm_61`, and
the device has no tensor cores. The sandbox caps memory at 5 GiB high and
6 GiB max.

```text
num_layers = 10
num_heads = 6
num_kv_heads = 3
hidden_dim = 384
seq_len = 256
batch = 4
vocab_size = 512
padded_vocab_size = 512
window_pattern = "SSSL"
value_embedding = true
seed = 42
steps = 5
```

`padded_vocab_size` is a separate field. `src/model.cc` reads it directly. If
the test leaves the field at the default 32768, the classifier stays wide and
the logits use about 268 MB.

Use `fp32` only. Keep the shapes small. The test measures agreement, not speed.

### 6.3 Greedy tie rule

A random 10-layer model can produce near-ties at the argmax. One unit of
roundoff can flip the result. The test records the logit margin and requires the
margin to exceed a threshold.

The implementation measures the margin at the position that predicts each
sampled token. It reads the logits from the training graph over the whole row,
so one forward covers every step of a rollout. The threshold is `1e-5`. The
measured minima of the committed fixture are:

| Section | Evaluation | Generation | RL |
|---|---|---|---|
| fp32 | `3.73e-3` | `5.75e-2` | `3.73e-2` |
| bf16 | `0` | `4.69e-2` | `1.56e-2` |
| fp8 | `0` | `6.25e-2` | `0` |

The fp32 head is peaked, so the rule passes. A bf16 or fp8 build rounds the
logits to 7 mantissa bits, so the top two logits can be exactly equal and the
margin is exactly zero. The test prints the margin for those builds and does not
fail on it. The discrete records still compare byte for byte, and the committed
sections are stable.

### 6.4 Trace inputs

Every input comes from the seed 42 and the committed tokenizer, so the fixture
needs no data file. The trace records no batch and no weight vector.

- The six pretraining batches come from a xoshiro256\*\* source seeded with
  `seed ^ 0x9e3779b97f4a7c15`. Each row draws `seq + 1` ids in `[0, vocab)`, and
  the row is the input and its one-position shift is the target.
- The evaluation batches come from the committed tokenizer
  `tests/data/loader_tokenizer.nctoken` and 128 fixed documents. The loader
  refills its buffer to 64 documents, so the packing does not depend on
  producer timing.
- The generation prompt is 16 ids from a source seeded with
  `seed ^ 0x2545f4914f6cdd1d`. Greedy decoding adds 16 ids and no stop token.
- The RL rollout uses four prompts of 16 ids each, from a source seeded with
  `seed ^ 0x5bf03635ef1b0a0d`. Greedy decoding adds 32 ids per row. The reward
  rule is a weighted mean of the generated ids, mapped to `[0, 1]`. The
  advantage is the reward minus the mean reward over the four rows.
- The SFT section reuses the first evaluation batch. The weight is 1 on the
  second half of each row and 0 on the first half, and every fifth target is the
  ignore index. The recorded normalizer is the valid-target count.
- The RL section takes the weights from the advantages on the sampled positions
  and 0 elsewhere. The scale is `1 / (num_passes * examples_per_rank)`, so the
  normalizer is `num_valid * num_passes * examples_per_rank`.

## 7. Phase plan

Status: the tree holds phases 0 to 5. Phase 6 holds steps 1 and 2. Step 3
stays open. The names below match the tree:
`tests/numerics_trace.{h,cc}` holds the trace and the comparison,
`tests/numerics_trace_emit_main.cc` holds the emitter, and
`tests/numerics_trace_test.cc` holds the replay gate.

Owners follow the ownership map in [AGENTS.md](../AGENTS.md). The Oracle
workstream owns `tests/**` and `tools/dump_*.py`. The Harness workstream owns
`src/**`. The Architect owns the frozen headers, the root build files, and
`docs/**`.

### Phase 0: contract and fixture format

1. The Architect records the contract in [parity.md](parity.md).
2. The Oracle extends `src/container.h` with the `NANONUM1` magic.
3. The Architect freezes the format.

Gate: a reviewer approves the contract and the format.

### Phase 1: trace primitives and the training trace

1. The Oracle writes `tools/dump_fixtures.py numerics`.
2. The Oracle writes the emitter. It builds the model with `Model::Create`,
   calls `InitWeights(seed)`, and drives `ForwardLoss`, `BackwardAccumulate`,
   and `Optimizer::Step` for five steps.
3. The Oracle writes the fixture.
4. The Oracle adds the two targets to `tests/BUILD.bazel`.

Gate: every leg equals its section byte for byte.

### Phase 2: evaluation trace

1. The Oracle adds an evaluation section to the emitter.
2. The emitter calls `EvalBpb` inside a `NoGradGuard`.
3. The emitter calls `ScoreBatch` on a fixed batch.
4. The emitter records the token ids and the greedy argmax identifiers.

Gate: the argmax identifiers agree on all three targets.

### Phase 3: greedy generation spike and trace

1. The Oracle records the greedy `GenerateBatch` identifiers for a fixed prompt
   and a fixed mask.
2. The Oracle measures the logit margin of each step and applies the tie rule.

Gate: the generated identifiers are identical on all three targets, and the
margin exceeds the threshold at each step.

### Phase 4: SFT arithmetic

The SFT arithmetic is a floating-point result, so it is not a trace record.

1. The Oracle adds one masked SFT step to `//tests:sft_step_parity_test`.
2. The test records the post-step loss and the gradient norm in its trajectory.

Gate: the SFT loss and the gradient norm agree within the bound of
[testing.md](testing.md).

### Phase 5: RL arithmetic

The RL arithmetic is a floating-point result, so it is not a trace record.

1. The Oracle records the rollout identifiers and the valid-target count in the
   trace.
2. The Oracle records the reward, the advantage, and the objective in
   `//tests:rl_parity_test`.

Gate: the rollout identifiers are identical, and the RL arithmetic agrees within
its bound.

### Phase 6: wiring and gates

1. The Architect adds `//tests:numerics_trace_test` to the `correctness` suite
   of `tools/nanochat simulate`.
2. The Architect adds the test to the merge-gate GPU budget in
   [AGENTS.md](../AGENTS.md) section 5.5 as a `t2-parity` gate.
3. The Harness adds a `--trace <path>` option to `train_main` and `eval_main`
   if full harness coverage is necessary. This step is optional.

Gate: two commands run every leg. `tools/nanochat simulate --device
gtx1080ti --suite correctness` runs the CPU leg and the simulator leg.
`tools/nanochat test --gpu //tests:numerics_trace_cuda_test` runs the GPU leg.

## 8. Commands and wall-clock budget

```bash
tools/nanochat test //tests:numerics_trace_test
tools/nanochat test --gpu //tests:numerics_trace_cuda_test
tools/nanochat simulate --device gtx1080ti --suite correctness
```

A 10-layer CPU run with `double` GEMM over five steps is the slow part. It takes
about 60 seconds under the `t0-cpu` profile. The GPU leg takes about 10 seconds,
including the repeat run of clause 2.

The CPU target carries the `manual` tag, so it stays out of the default test
loop. `tools/nanochat simulate --suite correctness` runs it, and
`tools/nanochat test //tests:numerics_trace_test` runs it alone. Keep the
per-merge GPU budget in [AGENTS.md](../AGENTS.md) section 5.5.

## 9. Risks

- The discrete results are derived from floats, so a change that flips one
  argmax changes the fixture. That is a real change and a re-emission, not a
  host difference.
- A low-precision build can tie at the argmax. Section 6.3 records the measured
  margins. If a tie becomes unstable, the fixture is not usable for that
  section.
- A cuBLAS version change can change the device floating-point results. The
  device floating-point contract is `//tests:train_parity_cuda_test`, and the
  tolerance record pins the CUDA and cuBLAS versions.
- The fixture depends on the pinned host thread count and on the tokenizer
  artifact of the evaluation phase. Regenerate it after a change to either.
- The trace covers the discrete results only. A pure floating-point drift that
  flips no identifier is caught by the parity tests, not here.
- The 11 GB card and the 5 GiB sandbox cap bound the sequence length and the
  vocabulary. Keep `seq_len` at or below 256.

## 10. Open questions

1. Does clause 3 hold on the Pascal GPU for a 10-layer model? Resolved by
   Phase 3: the identifiers agree on all three targets.
2. Does the operator accept the reduction in section 1? If the operator needs
   bit equality on the device leg, choose Path B.
3. Should the comparison run in continuous integration, or only at merge? The
   GPU tier suggests merge only, and section 5.5 settles it. The `manual` tag
   keeps the CPU target out of the inner loop.
4. Is `vocab_size = 512` acceptable for the parity goal? The production model
   uses 32768.
