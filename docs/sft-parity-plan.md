# Supervised fine-tuning reference parity: plan

How to earn the name "reference parity" for the supervised fine-tuning path.
This document also states which part of the claim can close before the native
packed-row reader and `sft_main` land.

Status: the plan, revised after an independent review. Phase A is not started.
The renderer, the mixture, and the Python packer are in the tree. See
[post-training.md](post-training.md) for the design and [parity.md](parity.md)
for the status of record.

## 1. Goal

The path must reproduce the reference `scripts/chat_sft.py`. The label
"reference parity" is a claim in [parity.md](parity.md). A claim needs evidence
and a gate. This plan defines both.

The work splits into four rungs. The rungs have different prerequisites. Only
the first two can close before the native reader and `sft_main` land.

## 2. What the label requires

[docs/parity.md](parity.md) defines a parity claim. Three items are necessary:

1. A row in the parity table with a status. The status is `closed`,
   `equivalent`, `open`, or `out-of-scope`.
2. A committed fixture that the PyTorch reference produced.
3. A gate that compares one backend against that fixture
   ([testing.md](testing.md)).

The current post-training rows are `P1`, `P2`, and `P3`. They cover the
calculator tool, the `rl_step` lag, and the base checkpoint conversion. No row
covers the SFT data, the SFT arithmetic, the SFT loop, or the SFT validation.

`//tests:numerics_trace_test` records an SFT masked loss and an SFT gradient
norm (`tests/numerics_trace.cc` lines 802 to 845). That trace is a cross-target
regression guard. Its golden file comes from the C++ emitter, not from torch
(`tools/dump_numerics_golden.py` lines 8 to 9). It uses a synthetic mask and
takes no optimizer step. It gives no reference evidence for SFT arithmetic.

## 3. The four rungs

| Rung | Claim | Requires | Closeable before the reader and `sft_main`? |
|---|---|---|---|
| R1 | SFT data parity. The renderer and the packer match the reference byte for byte. | Reference ids, masks, packed rows, and mixture order. | Yes |
| R2 | SFT arithmetic parity. Loss, gradient, and an N-step optimizer trajectory match. | The embedded base parameters and moments, the packed rows, the optimizer configuration, and the warm start. | Yes |
| R3 | SFT loop parity. Loss curve and checkpoints match. | The native packed-row reader, `sft_main`, the progress schedule, and the warm start. | No |
| R4 | SFT validation parity. Bits-per-byte and ChatCORE match. | The packed validation rows, the evaluation cadence, and the chat task logic. | No |

Do not put one label on the whole path before R4. Add one parity row per rung.
The map is one to one: `P4` is R1, `P5` is R2, `P6` is R3, and `P7` is R4.

One dependency order applies. R2 consumes the packed rows that R1 produces.
The warm start leans on entry `P3` and on `LoadOptimizerState`
(`src/optim.cc`). No dependency forces R3 work earlier.

## 4. Required fixtures

Add three fixtures. Follow the pattern of `tools/dump_train_fixture.py`.

### 4.1 `tests/data/sft_pack_fixture.bin`

The packer fixture holds:

- A fixed conversation set. Cover a SmolTalk-like conversation, an MMLU
  multiple-choice question, a GSM8K solution with tool parts, a system message,
  and one oversized conversation.
- The reference ids and mask for each conversation.
- The reference packed inputs and targets. Include the BOS padding and the
  `-1` positions.
- The reference packed rows for the validation mixture.
- The mixture index order for seed 42.
- The tokenizer identity: the artifact hash and the special-token ids.

### 4.2 `tests/data/sft_step_fixture.bin`

The step fixture holds:

- The initial base parameters and the base optimizer moments, embedded in the
  file.
- The packed rows for the first `N` steps.
- The optimizer configuration, group by group.
- The per-step loss, gradient norm, parameter L2, learning rate, and Muon
  momentum.

The fixture embeds the parameters and the moments. It must not reference a
checkpoint path, because the runtime test never reads torch
(`docs/testing.md`).

### 4.3 `tests/data/sft_loop_fixture.bin`

The loop fixture holds:

- The `progress`, `epoch`, rate multiplier, and Muon momentum trace.
- The reference validation bits-per-byte at the evaluation cadence.
- The reference ChatCORE value at the evaluation cadence.

## 5. Reference settings to pin

The fixture and the loop must share these values.

### 5.1 Data and shape

- The mixture: SmolTalk train, MMLU `auxiliary_train` times 3, GSM8K `main`
  train times 4.
- The inherited shape: `max_seq_len`, `device_batch_size`, and
  `total_batch_size` from the base checkpoint.
- The gradient accumulation: `total_batch_size / (device_batch_size *
  max_seq_len * world_size)`.
- The precision: fp32.

### 5.2 Schedule and rates

- The initial learning-rate fraction: 0.8.
- The warmup ratio: 0. The warmdown ratio: 0.5. The final rate fraction: 0.
- The Muon momentum: 0.85 to 0.95 over 300 steps, then hold.
- The rate scale for the AdamW groups: `dmodel_lr_scale = (model_dim / 768) **
  -0.5`.
- The AdamW groups keep their fixed decays and betas:
  `unembedding 0.01`, `embedding 0.001`, `value embeddings 0.01`,
  `residual 0.05`, `x0 0.0`, `smear 0.0`. The betas are `(0.8, 0.96)`,
  `(0.8, 0.995)`, `(0.8, 0.995)`, `(0.8, 0.95)`, `(0.96, 0.95)`, and
  `(0.8, 0.95)`. The `eps` is `1e-10` for every group.
- The scalar-group rates: `residual = scalar_lr * 0.01`, `x0 = scalar_lr`,
  `smear = 0.2`, `value embeddings = embedding_lr * 0.5`.
- The Muon group weight decay: 0. The AdamW group weight decay is the fixed
  value above.
- The base matrix rate: 0.02. The base embedding rate: 0.3. The base
  unembedding rate: 0.004.

### 5.3 Warm start and checkpoint

- The warm start: load the base moments, then restore the fresh SFT rates.
  The load must not overwrite the rates.
- The checkpoint: `chatsft_checkpoints/<tag>` with the optimizer state.
- The checkpoint metadata: `model_config` and `user_config`.

### 5.4 Evaluation

- The evaluation interval: 200 steps.
- The evaluation tokens: `40 * 524288`.
- The ChatCORE interval: 200 steps.
- The ChatCORE categorical cap: -1. The generative cap: 24.
- The validation: bits-per-byte on the packed validation rows.

## 6. Phases

| Phase | Owner | Deliverable | Gate |
|---|---|---|---|
| A. Freeze the contract | Architect | Add `P4` to `P7` to `parity.md`. Update `post-training.md` sections 10 and 11. Set the tolerances from the oracle fixtures and the training-parity harness in `docs/testing.md` (loss `1e-5`, parameter `1e-4`, fp32 CUDA about `1e-5`). | Document review and the ASD-STE100 checker |
| B. Fixtures | Oracle | Add `tools/dump_sft_fixture.py`. Commit the three fixtures. | The fixtures load on the CPU with no reference checkout |
| C. R1 data parity | Python surface for `python/tests/`; Oracle for the fixture | Extend `python/tests/sft_data_test.py` against `sft_pack_fixture.bin`. Use the pinned tokenizer. Check the mixture order and the validation rows. | `tools/nanochat test //python:sft_data_parity_test` at T0 |
| D. R2 arithmetic parity | Harness and Oracle | Add `tests/sft_step_parity_test.cc`. Drive an N-step trajectory from the embedded base parameters and moments. Compare loss, gradient norm, and parameter L2 per step. Add the warm start. Extend the numeric trace with one SFT optimizer step, and regenerate `tests/data/numerics_golden_10l.bin` and the tolerance table in `docs/numerics-integration.md` section 4. | T0, then T1 at the tiny shape, then T2 |
| E. R3 schedule fixture | Oracle and Harness | Add `tools/dump_sft_loop_fixture.py`. Record progress, epoch, the rate multiplier, and the momentum. Add a unit test. | T0 |
| F. R3 native reader and `sft_main` | Harness for `src/data.cc` and `src/train.cc`; Python surface for the bridge | Add the packed-row source in `src/data.cc`. Add `sft_main` and `nc.sft.run`. Add the progress schedule. Consume the R2 warm start. | `//tests:sft_parity_test` at T2 |
| G. R3 and R4 loop and validation parity | Integrator | Run both implementations from one base checkpoint. Compare the loss curve, the validation bits-per-byte, ChatCORE, and the final parameter L2. Set `P6` and `P7`. | T2, then the `parity.md` update |

The warm start belongs to R2. Phase F consumes it and does not repeat it.

## 7. Interim label

Use these names until Phase G closes:

- After Phase C: "SFT data parity" (`P4`).
- After Phase D: "SFT arithmetic parity" (`P5`), with the GPU leg.
- Until Phase G: "SFT loop parity" (`P6`) and "SFT validation parity" (`P7`)
  stay `open`.

## 8. Open questions and risks

- Progress schedule. The reference computes the rate from `progress`, because
  the run stops at the dataset end. The C++ `Scheduler` uses a step count.
  Phase E must decide: add a progress input, or fix the step count up front.
- Validation. The reference evaluates bits-per-byte on the packed validation
  mixture. The C++ `Evaluator` reads parquet documents. A direct comparison
  fails. `P7` must define a packed-row evaluation. This is the easiest rung to
  leave undefined.
- Warm start. The reference loads the base moments and then restores the fresh
  SFT rates. The C++ load must not overwrite them.
- Mixture order. The order depends on the `datasets` shuffle with seed 42 and
  on `random.Random(42)`. Both must match `sft_data`. The reference `datasets`
  version can drift.
- Muon momentum indexing. The reference step counter is 0-based. The C++
  `Optimizer::Step` is 1-based. Phase E must pin the offset.
- Rate scale. `setup_optimizer` scales the rates by the model dimension.
  Confirm the C++ groups match.
- Checkpoint metadata. The reference writes `model_config` and `user_config`.
  Confirm the C++ container carries the same fields.
- GPU budget. R2, R3, and R4 are T2 runs. Schedule them at a wave boundary,
  not for every commit.

## 9. Non-goals

- A distributed SFT run.
- A preference-optimization method.
- A native chat template in the C++ model layer.
