# Numerics integration test: CPU, simulator, and Pascal GPU

This document defines an integration test for numeric agreement across three
targets. The three targets are the CPU backend, the CUDA simulator, and the
Pascal GPU. The test uses a 10-layer model. It covers training, evaluation,
supervised fine-tuning (SFT), and reinforcement learning (RL).

The test follows Path A. Path A is a staged precision contract. This document
also records what Path A does not do.

Status: phases 0 to 5 are complete. Phase 6 steps 1 and 2 are complete. Phase 6
step 3 stays open: the optional `--trace` option of the harness. The trace
library, the emitter, the two Bazel targets, the golden file, the simulator
wiring, and the merge-gate entry are present. Section 4 records the measured
GPU maxima.

## 1. The reduction, stated first

Path A does not give bit-exact agreement across all three targets. It gives
four weaker results. State them before the design.

1. The CPU trace and the simulator trace are bit-identical. The two targets
   share one code path. Only `GetCaps()` changes.
2. The CPU target and the simulator target are deterministic across runs. The
   GPU target is reproducible within a tolerance, not bit-identical. The device
   kernels use atomic scatter-add operations.
3. The Pascal GPU trace agrees with the CPU golden trace within a tolerance.
4. Greedy token identifiers agree exactly on all three targets when the argmax
   margin is large.

The status table in [post-training.md](post-training.md) marks the SFT renderer,
the SFT packer, the SFT loop, the `rl_step` binary, and the RL fixture as
**Missing**. Section 6 gives kernel-level coverage only.

## 2. What the recon established

Four facts shape the design.

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

One more fact fixes the architecture. Bazel selects the backend at link time
(see `src/BUILD.bazel`). The CPU backend and the CUDA backend define the same
`nanochat::kernels::*` symbols. One process cannot hold both. The test compares
each backend against one committed golden file.

## 3. The precision contract

The contract has four clauses.

1. The CPU trace equals the golden trace, byte for byte.
2. The simulator trace equals the golden trace, byte for byte.
3. The GPU trace reproduces the golden trace within the tolerance table in
   section 4. The GPU trace must also reproduce itself within the same
   tolerances across two runs.
4. The greedy token identifiers equal the golden identifiers on all three
   targets.

Clause 4 uses parity entry `E1`. [parity.md](parity.md) records greedy
generation as bit-identical. Clause 4 holds only for a peaked model head.
Section 6.4 defines the tie rule.

## 4. Tolerance table

The table holds the applied tolerance and the maxima measured on the GTX 1080
Ti. The measurement is the CPU golden replay and the GPU run of
`//tests:numerics_trace_cuda_test`. The precedent is `tests/oracle_test.cc` for
the tight bounds and `tests/train_parity_test.cc` for the trajectory bounds.

| Stage | Artifact | CPU vs simulator (regression guard) | GPU vs CPU golden | Measured GPU maximum |
|---|---|---|---|---|
| Train | loss | exact | 1e-5 relative | 7.6e-8 relative |
| Train | gradient norm | exact | 1e-5 relative | 9.8e-8 relative |
| Train | parameter L2 after five steps | exact | 1e-4 relative | 2.6e-5 relative |
| Eval | bits per byte, per-batch loss | exact | 1e-5 relative | 0 |
| Eval | greedy argmax identifiers | exact | exact | exact |
| Eval | logit margin | exact | 1e-5 absolute | 1.4e-6 absolute |
| Generate | greedy identifiers and mask | exact | exact | exact |
| Generate | logit margin | exact | 1e-5 absolute | 4.8e-7 absolute |
| SFT | masked weighted loss, gradient norm | exact | 1e-5 relative | 0 |
| RL | greedy generated token identifiers | exact | exact | exact |
| RL | advantage-weighted loss | exact | 1e-5 relative | 1.3e-6 relative |
| RL | gradient norm | exact | 1e-5 relative | 6.9e-8 relative |
| RL | reward and advantage | exact | 1e-5 relative | 0 |
| RL | logit margin | exact | 1e-5 absolute | 2.4e-7 absolute |

The "CPU vs simulator" column is exact by construction. The simulator leg is
the CPU backend under `--config=sim` (section 2.2). The two runs share one code
path.

The column is a regression guard. It catches a future numeric path that starts
to read `GetCaps()`. It is not an independent measurement of a device. The
"GPU vs CPU golden" column is the only cross-target check. The measured maxima
leave headroom above the applied tolerances; the smallest one is the parameter
L2 column at 2.6e-5 measured against 1e-4 applied.

## 5. Test architecture: golden replay

The test follows the existing `oracle_test` pattern. One source file builds as
two Bazel targets. One committed golden file is the reference.

- The **golden file** `tests/data/numerics_golden_10l.bin` comes from the CPU
  emitter. A reviewer checks the file. A developer commits the file. The file
  does not need PyTorch at runtime.
- The **CPU target** `//tests:numerics_trace_test` runs the model, then
  compares the result against the golden in process. The comparison is exact.
- The **simulator leg** runs the same target under `--config=sim`. The
  comparison is exact. This leg is a regression guard, not a third numeric
  implementation.
- The **CUDA target** `//tests:numerics_trace_cuda_test` shares the source and
  carries the tag `gpu`. The comparison uses the tolerance table.
- The **comparison runs in process.** CPU equals golden and simulator equals
  golden implies CPU equals simulator. No second binary and no file passing
  between builds are necessary.

The test reuses `tests/oracle_fixture.h`. That reader already stores named
float tensors and scalars. Phase 0 added one magic string to the reader and no
record sequence, because the flat layout holds the trace.

### 5.1 Trace format

The trace reuses the little-endian record layout of `tests/oracle_fixture.h`
under its own magic `NANONUM1`. The phase, the step, and the parameter name live
in the record name. The container is `[8-byte magic][u32 version][u32 count]`,
then one record per
`[u16 name length][name][u8 dtype][u8 width][u64 shape...][payload]`.

| Record group | Names |
|---|---|
| Identity | `header/version`, `header/magic`, `header/backend`, `header/precision`, `header/build`, `header/seed`, `header/steps` |
| Configuration | `config/layers`, `config/heads`, `config/kv_heads`, `config/hidden`, `config/seq`, `config/batch`, `config/vocab`, `config/padded_vocab`, `config/window_pattern`, `config/value_embedding`, `config/rope_base`, `config/parameters` |
| Optimizer and schedule | `opt/unembedding_lr`, `opt/embedding_lr`, `opt/matrix_lr`, `opt/scalar_lr`, `opt/weight_decay`, `opt/clip`, `opt/adam_eps`, `opt/muon_ns_steps`, `opt/muon_beta2`, `sched/*` |
| Phase inputs | `train/batches`, `eval/steps`, `gen/prompt_len`, `gen/tokens`, `gen/num_samples`, `sft/ignore_every`, `rl/num_samples`, `rl/prompt_len`, `rl/tokens`, `rl/num_passes`, `rl/examples_per_rank`, `rl/reward_rule` |
| Training | `train/step<N>/loss`, `train/step<N>/grad_norm`, `train/step<N>/param_l2/<name>`, `train/step<N>/param_hash/<name>`, `train/step<N>/grad_hash/<name>` |
| Evaluation | `eval/tokens/<b>`, `eval/bpb`, `eval/loss/<b>`, `eval/argmax/<b>`, `eval/logits_hash`, `eval/margin_min` |
| Generation | `gen/prompt`, `gen/ids`, `gen/mask`, `gen/margin/<i>`, `gen/margin_min` |
| SFT | `sft/valid_targets`, `sft/zero_weights`, `sft/loss`, `sft/grad_norm` |
| RL | `rl/prompt/<r>`, `rl/ids/<r>`, `rl/reward/<r>`, `rl/advantage/<r>`, `rl/mean_reward`, `rl/valid_targets`, `rl/loss`, `rl/grad_norm`, `rl/margin_min` |

The `sched/*` group records the AdamW and Muon fields of the reference
configuration. The Anvil rail fields are not recorded.

The trace stores the two hashes as 64-bit integers. The hash is FNV-1a over the
staged host float bytes of one parameter buffer or one gradient buffer. The
backend name, the precision, and the build name identify the producer. The
simulator leg keeps the name `cpu`, because it shares the code path.

The trace pins the host thread count to one. The CPU reference reductions depend
on the thread count, so the pin makes the golden file independent of the sandbox
thread budget.

The CPU target carries `-DNANOCHAT_TRACE_EXACT`. The intent to compare byte for
byte therefore belongs to the target, not to a backend define. One source
builds both targets, and Bazel shares one output path across the `--define`
configurations.

## 6. Trace contents and model configuration

### 6.1 Trace contents

The trace holds the header fields and the record fields.

- The header holds the magic string, the backend identifier, the precision, the
  build configuration name, the `Config` values, and the seed.
- Each record holds the step index, the loss, the gradient norm, and the L2
  norm of each parameter.
- Each record also holds a 64-bit hash of the parameter bytes and a 64-bit hash
  of the gradient bytes.
- The evaluation section adds the bits per byte, the per-batch loss, the logits
  hash, and the greedy argmax identifiers.

The hash is valid for the exact clauses only. The hash is meaningless for the
GPU leg, because the bytes differ by design. The GPU leg checks the scalars by
tolerance and the greedy identifiers exactly.

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

### 6.3 Determinism notes

The grouped-query attention path and the value-embedding path use atomics. Set
`num_kv_heads = num_heads` and `value_embedding = false` for the strictest
reproducibility, if the operator accepts a less representative model. Keep the
representative configuration and use a reproducibility tolerance otherwise.

### 6.4 Greedy tie rule

A random 10-layer model can produce near-ties at the argmax. One unit of
roundoff can flip the result. Phase 3 tests the model head for peakedness. The
test records the logit margin and requires the margin to exceed a threshold.
The test uses a briefly trained fixture otherwise. See the generate-parity
fixture note in `tests/BUILD.bazel`.

The implementation measures the margin at the position that predicts each
sampled token. It reads the logits from the training graph over the whole row,
so one forward covers every step of a rollout. The threshold is `1e-5`. The
smallest recorded margin of the committed fixture is `3.7e-3` for the
evaluation section, `5.8e-2` for generation, and `3.7e-2` for RL. The head is
therefore peaked enough on all three targets, and the identifiers agree exactly.

### 6.5 Trace inputs

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
  is a weighted mean of the generated ids, mapped to `[0, 1]`. The advantage is
  the reward minus the mean reward over the four rows.
- The SFT section reuses the first evaluation batch. The weight is 1 on the
  second half of each row and 0 on the first half, and every fifth target is the
  ignore index. The normalizer is the valid-target count.
- The RL section takes the weights from the advantages on the sampled positions
  and 0 elsewhere. The trace indexes the weight vector like the targets. The
  loss that predicts the first sampled token is therefore 0, because the mask
  applies to the shifted targets. The scale is
  `1 / (num_passes * examples_per_rank)`, so the normalizer is `num_valid *
  num_passes * examples_per_rank`.

## 7. Phase plan

Status: the tree holds phases 0 to 5. Phase 6 holds steps 1 and 2. Step 3
stays open. The names below match the tree:
`tests/numerics_trace.{h,cc}` holds the trace, the hash helper, and the
comparison; `tests/numerics_trace_emit_main.cc` holds the emitter; and
`tests/numerics_trace_test.cc` holds the replay gate.

Owners follow the ownership map in [AGENTS.md](../AGENTS.md). The Oracle
workstream owns `tests/**` and `tools/dump_*.py`. The Harness workstream owns
`src/**`. The Architect owns the frozen headers, the root build files, and
`docs/**`.

### Phase 0: precision contract and trace format

This phase blocks all other phases. The Architect and the Oracle deliver one
approval together.

1. The Architect records the contract in [parity.md](parity.md).
2. The Oracle extends `tests/oracle_fixture.h` with the `NANONUM1` magic. The
   flat record layout holds the trace, so no record sequence is necessary.
3. The Architect freezes the format.

Gate: a reviewer approves the contract and the format. Phase 1 does not start
before this gate.

### Phase 1: trace primitives, training trace, and tolerance calibration

1. The Oracle writes `tools/dump_numerics_golden.py`.
2. The Oracle adds the trace writer and the hash helper.
3. The Oracle writes the training emitter. The emitter builds the model with
   `Model::Create` and calls `InitWeights(seed)`. The emitter then drives
   `ForwardLoss`, `BackwardAccumulate`, and `Optimizer::Step` for five steps.
   The emitter follows `tests/train_parity_test.cc`. It needs no harness change.
4. The Oracle writes the CPU golden file.
5. The Oracle runs the GPU target twice. The Oracle records the observed maxima
   for each scalar beside the applied tolerances in section 4.
6. The Oracle adds the two targets to `tests/BUILD.bazel`.

Gate: the CPU trace and the simulator trace equal the golden byte for byte. The
GPU trace passes the measured tolerance table.

### Phase 2: evaluation trace

1. The Oracle adds an evaluation section to the emitter.
2. The emitter calls `EvalBpb` inside a `NoGradGuard`.
3. The emitter calls `ScoreBatch` on a fixed batch.
4. The emitter records the bits per byte, the per-batch loss, the logits hash,
   and the greedy argmax identifiers.

Gate: the bits per byte and the loss agree within tolerance. The argmax
identifiers agree exactly on all three targets.

### Phase 3: greedy generation spike and trace

This phase supplies the rollout primitive for RL. This phase also resolves the
greedy tie risk. Run this phase before the RL phase.

1. The Oracle records greedy `GenerateBatch` token identifiers for a fixed
   prompt and a fixed mask.
2. The Oracle records the logit margin for each step.
3. The Oracle applies the tie rule from section 6.4.

Gate: the generated token identifiers are identical on all three targets. The
logit margin exceeds the threshold at each step.

### Phase 4: SFT arithmetic trace

This phase covers the SFT arithmetic kernel only. The full SFT loop needs
Phase 1 and Phase 2 of [post-training.md](post-training.md).

1. The Oracle drives the masked weighted cross-entropy with `BackwardWeighted`
   and per-row weights.
2. The Oracle records the masked weighted loss and the gradient norm.

Gate for this phase: the masked weighted loss and the gradient norm agree
within tolerance. Gate for full SFT: the SFT loss curve agrees at the
`t2-parity` tier, after the SFT loop lands.

### Phase 5: RL arithmetic trace

This phase covers the RL arithmetic kernel only. The full RL objective needs
Phase 3 of [post-training.md](post-training.md).

1. The Oracle rolls out with greedy `GenerateBatch`.
2. The Oracle computes the advantage as the reward minus the mean reward.
3. The Oracle normalizes by `num_valid * num_passes * examples_per_rank`.
4. The Oracle calls `BackwardWeighted`.
5. The Oracle records the advantage-weighted loss and the gradient norm.

Gate for this phase: the advantage-weighted loss and the gradient norm agree
within tolerance. The rollout identifiers agree exactly. Gate for full RL:
`//tests:rl_parity_test` passes at the `t2-parity` tier, after `rl_step` lands.

### Phase 6: wiring and gates

1. The Architect adds `//tests:numerics_trace_test` to the `correctness` suite
   of `tools/nanochat simulate`.
2. The Architect adds the test to the merge-gate GPU budget in
   [AGENTS.md](../AGENTS.md) section 5.5 as a `t2-parity` gate.
3. The Harness adds a `--trace <path>` option to `train_main` and `eval_main`
   if full harness coverage is necessary. This step is optional.

Gate: two commands run all three legs. `tools/nanochat simulate --device
gtx1080ti --suite correctness` runs the CPU leg and the simulator leg.
`tools/nanochat test --gpu //tests:numerics_trace_cuda_test` runs the GPU leg.
The test writes the observed delta for each group to its log. The repository
sets `--test_output=errors`, so a passing run keeps the table in
`bazel-testlogs/`.

## 8. Commands and wall-clock budget

```bash
tools/nanochat test //tests:numerics_trace_test
tools/nanochat test --gpu //tests:numerics_trace_cuda_test
tools/nanochat simulate --device gtx1080ti --suite correctness
```

A 10-layer CPU run with `double` GEMM over five steps is the slow part. It takes
about 85 seconds under the `t0-cpu` profile, and the simulator leg repeats it.
The GPU leg takes about 10 seconds, including the repeat run of clause 3.

The CPU target carries the `manual` tag, so it stays out of the default test
loop. `tools/nanochat simulate --suite correctness` runs it, and
`tools/nanochat test //tests:numerics_trace_test` runs it alone. Keep the
per-merge GPU budget in [AGENTS.md](../AGENTS.md) section 5.5.

## 9. Risks

- Path A does not give bit equality on the GPU leg. Path B is the prerequisite
  for that goal. Path B changes the CPU GEMM accumulation and the CUDA GEMM
  kernel.
- The CUDA determinism clause is impossible to guarantee with atomics. The
  contract scopes the determinism clause to the CPU and the simulator.
- The simulator is not an independent implementation. The exact check is a
  regression guard.
- A cuBLAS version change can change the reduction order. `backends/cuda/gemm.cu`
  uses `CUBLAS_GEMM_DEFAULT`, which is a heuristic. Pin the CUDA toolkit version
  and the cuBLAS version in the tolerance record.
- Missing code blocks full SFT and full RL coverage.
- The 11 GB card and the 5 GiB sandbox cap bound the sequence length and the
  vocabulary. Keep `seq_len` at or below 256.
- The golden file depends on the pinned host thread count and on the tokenizer
  artifact of the evaluation phase. Regenerate it with
  `python3 tools/dump_numerics_golden.py` after a change to either one.
- The trace covers fp32 only. An fp16 build must not replay the golden, and the
test stops with that message.

## 10. Open questions

1. Does clause 4 of the contract hold on the Pascal GPU for a 10-layer model?
   Resolved by Phase 3: the smallest recorded margins are 3.7e-3, 5.8e-2, and
   3.7e-2 against the 1e-5 threshold, and the identifiers agree exactly on all
   three targets.
2. Does the operator accept the reduction in section 1? If the operator needs
   bit equality on the GPU leg, choose Path B.
3. Should the comparison run in continuous integration, or only at merge? The
   GPU tier suggests merge only, and section 5.5 settles it. The `manual` tag
   keeps the CPU target out of the inner loop.
4. Is `vocab_size = 512` acceptable for the parity goal? The production model
   uses 32768.
