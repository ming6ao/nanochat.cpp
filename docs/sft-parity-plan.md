# Supervised fine-tuning reference parity: plan

How to earn the name "reference parity" for the supervised fine-tuning path.
This document is the schedule. [post-training.md](post-training.md) is the
design. [parity.md](parity.md) is the status of record.

Status: Phase A landed. This document pins the data path, and Phase C.0 landed
with it. Phase B is next. Phases C.1, D, E, F, and G remain. Section 9 lists the
state of the tree.

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

The post-training rows are `P4` to `P7`. They cover the SFT data, the SFT
arithmetic, the SFT loop, and the SFT validation. No other row covers those
items.

`//tests:numerics_trace_test` records an SFT masked loss and an SFT gradient
norm (`tests/numerics_trace.cc`). That trace is a cross-target regression
guard. Its golden file comes from the C++ emitter, not from torch. It uses a
synthetic mask and takes no optimizer step. It gives no reference evidence for
SFT arithmetic.

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

## 4. Decisions

These decisions close the open questions of the first revision. Do not reopen
one without a new entry here.

### D1. The schedule takes progress, not a step count

The reference computes the learning-rate multiplier from `progress`, because a
run can stop at the dataset end. Add two items to the frozen interface:

- `SchedulerConfig::warmup_ratio` (default `0.0`).
- `Scheduler::LrMultiplierForProgress(float progress)`.

The body mirrors `scripts/chat_sft.py:310-317`:

- If `progress < warmup_ratio`, return `(progress + 1e-8) / warmup_ratio`.
- Else, if `progress <= 1.0 - warmdown_ratio`, return `1.0`.
- Else, return `1.0 + decay * (final_lr_frac - 1.0)`, where
  `decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio`.

Keep `LrMultiplier(int step)` for pretraining. The owner of
`include/nanochat/scheduler.h` is the Architect. The owner of `src/optim.cc` is
the Optimizer workstream.

### D2. The Muon momentum keeps the 1-based step

`Optimizer::Step` is 1-based and `Scheduler::MuonMomentum` evaluates at
`step - 1`. The reference loop counter `it` is 0-based. The two agree today.
The SFT driver restarts its counter at 1 after the warm start. With
`muon_momentum_peak = muon_momentum_final = 0.95`, the ramp holds at 0.95 after
300 steps, which is the reference behavior. No code change is necessary.

### D3. `init_lr_frac` belongs to the optimizer configuration

The reference multiplies every group rate by `init_lr_frac = 0.8` before the
first step. Add `OptimizerConfig::init_lr_frac` (default `1.0`) to the frozen
header `include/nanochat/optim.h`. `src/optim.cc` multiplies every group rate,
including the smear group, by that value in `Build`. Do not scale the value at
each call site.

### D4. The SFT run does not clip the gradient

The reference SFT loop has no gradient clip. The C++ default `clip` is `1.0`.
The SFT driver sets `clip = 0.0`. This changes no interface.

### D5. The warm start is not a resume

Add two load modes to `sft_main`:

- `--warm-start PATH` loads the weights and the optimizer moments, and keeps
  the fresh SFT rates. The step counter stays at 0. This is the reference
  warm start.
- `--resume PATH` loads the weights, the moments, and the stored step. This
  continues the same run.

Pass `nullptr` for the step to `Checkpointer::LoadModel` in the warm-start
case, because `src/train.cc` keeps the caller's step when the pointer is null.
Do not use `--resume` for a base checkpoint: the stored base step breaks the
SFT schedule.

### D6. The packed validation reports bits-per-byte

The reference `evaluate_bpb` sums the negative log-probability of every valid
target and divides by the number of target bytes:

```
bpb = total_nats / (ln(2) * total_bytes)
```

Add `EvaluatePackedRows` to `src/eval.{h,cc}` with this interface:

- Input: the model, an `NCSFT001` shard, the token-byte table, and a step count.
- Output: `bpb` and the number of valid targets.

The per-batch sum of nats is `mean_loss * valid_targets`, because
`Model::ForwardLoss` returns the mean over valid targets. The token-byte table
is `int` per token id. The validation fixture embeds the table, so the T0 test
needs no artifact.

The evaluator reads one pass of the validation shard. Require
`eval_steps * device_batch <= validation rows`. The pinned settings satisfy
this condition.

### D7. The checkpoint records the recipe

The reference inherits `max_seq_len`, `device_batch_size`, `total_batch_size`,
and the learning rates from the base checkpoint `user_config`. The C++ metadata
block has no such fields. Append a `Recipe` block to `CheckpointMetadata`
(`include/nanochat/checkpoint_metadata.h`), last in `VisitFields`, as rule 1 of
that header requires. The block carries:

- `device_batch_size`, `total_batch_size`.
- `embedding_lr`, `unembedding_lr`, `matrix_lr`, `scalar_lr`, `adam_eps`.
- `init_lr_frac`, `warmup_ratio`, `warmdown_ratio`, `final_lr_frac`.
- `muon_momentum_steps`, `muon_momentum_start`, `muon_momentum_peak`,
  `muon_momentum_final`.
- `mmlu_epochs`, `gsm8k_epochs`.

`Recipe` is the C++ equivalent of the reference `user_config`. The `Config`
block stays the model configuration. This item blocks R3 only.

### D8. The fixture pins the mixture algorithm, not the hub order

The real mixture order needs a network download and a pinned `datasets`
version. Make the gate hermetic instead. The pack fixture records the order of
a synthetic five-task mixture with lengths `7, 3, 3, 2, 2`. The test compares
that order against `sft_data.TaskMixture`, so the test pins the seed-42
shuffle.

The real mixture order is an optional generator mode, `--real-mixture`. It is
not a gate. Section 10 records the risk.

### D9. Two generator kinds cover the fixtures

The merged generator gains two kinds:

- `sft` writes `tests/data/sft_pack_fixture.bin`,
  `tests/data/sft_step_fixture.bin`, `tests/data/sft_rows_fixture.nchsft`,
  and `tests/data/sft_base_fixture.ckpt`.
- `sft_loop` writes `tests/data/sft_loop_fixture.bin`.

The pack fixture is `NANCHAT1`, one version word, and JSON, like
`tests/data/chat_fixture.bin`. The step fixture and the loop fixture use
`tools/fixture_format.py` records, like `tests/data/train_parity.bin`.

### D10. R3 compares two loops over one shard

The reference `chat_sft.py` reads its own mixture, so two loops cannot read the
same rows directly. Use one shard instead:

- The `sft` kind packs the fixture conversations with the reference packer, and
  it writes `tests/data/sft_rows_fixture.nchsft`.
- The `sft` kind writes the reference parameters, moments, and settings to
  `tests/data/sft_base_fixture.ckpt` at step 0.
- The `sft_loop` kind runs the reference model, optimizer, and schedule over
  that shard for 8 steps. It records the trace and the loss curve.
- `sft_main` runs over the same two files. Phase G compares the two runs.

The real mixture is an optional check, not a gate (D8). R1 covers the mixture.

## 5. Pinned settings

The fixture and the loop share these values.

### 5.1 Data and shape

- The training mixture: SmolTalk train, MMLU `auxiliary_train` times 3, GSM8K
  `main` train times 4.
- The validation mixture: SmolTalk test, MMLU test with `stop=5200`, GSM8K
  test with `stop=420`.
- The renderer truncates each conversation to 2048 tokens. The reference calls
  `render_conversation` with the default, and `scripts/chat_sft.py` never
  overrides it.
- The packer buffer holds 100 conversations. It refills the buffer before
  each largest-fit search.
- The packer searches the buffer for the largest conversation that fits the
  row. It pads the remainder with BOS tokens when none fits.
- The row capacity is `max_seq_len + 1`. The inputs are the first
  `max_seq_len` positions. The targets are the last `max_seq_len` positions.
- A masked target and a padding target are `-1`.
- The cursor advances by `world_size`, and it wraps with an epoch increment.
  The packer skips a conversation longer than one row at fetch time.
- The stream ends after the batch in which `consumed >= len(conversations)`.
  This is the reference `last_step`. One list of conversations yields about
  one epoch of rows.
- The inherited shape: `max_seq_len`, `device_batch_size`, and
  `total_batch_size` from the base checkpoint.
- The gradient accumulation:
  `total_batch_size / (device_batch_size * max_seq_len * world_size)`.
- The precision: fp32 for the T0 and T1 gates, the build precision for T2.

### 5.2 Schedule and rates

- The initial learning-rate fraction: 0.8. Apply it to every group.
- The warmup ratio: 0. The warmdown ratio: 0.5. The final rate fraction: 0.
- The learning rate follows D1 and the reference progress value. The loop takes
  the running maximum of `approx_progress`.
- The Muon momentum: 0.85 to 0.95 over 300 steps, then hold at 0.95.
- The rate scale for the AdamW groups: `(hidden_dim / 768) ** -0.5`.
- The gradient clip: 0.
- The per-group settings, before `init_lr_frac`:

| Group | Rate | Betas | Weight decay |
|---|---|---|---|
| unembedding | 0.004 × scale | (0.8, 0.96) | 0.01 |
| embedding | 0.3 × scale | (0.8, 0.995) | 0.001 |
| value embeddings | 0.15 × scale | (0.8, 0.995) | 0.01 |
| residual lambdas | 0.005 | (0.8, 0.95) | 0.05 |
| x0 lambdas | 0.5 | (0.96, 0.95) | 0.0 |
| smear and backout | 0.2 | (0.8, 0.95) | 0.0 |
| Muon matrices | 0.02 | beta2 0.9, ns_steps 5 | 0.0 |

- The `eps` is `1e-10` for every AdamW group. The `scale` column is the
  `hidden_dim` scale of the third bullet.

### 5.3 Warm start and checkpoint

- The warm start follows D5: weights and moments, fresh rates, step 0.
- The checkpoint path is `chatsft_checkpoints/<tag>`. It holds the optimizer
  state and the recipe block (D7).
- The checkpoint stage is `Stage::kSft`.

### 5.4 Evaluation

- The evaluation interval: 200 steps.
- The evaluation tokens: `40 * 524288`.
- The ChatCORE interval: 200 steps.
- The ChatCORE categorical cap: -1. The generative cap: 24.
- The validation metric: bits-per-byte on the packed validation rows, per D6.

### 5.5 Fixture sizes

Keep every fixture small enough for a fast T0 run. Use one tiny model for the
step, loop, and shard fixtures:

- Model: 2 layers, `hidden_dim=32`, `num_heads=2`, `num_kv_heads=2`,
  `vocab_size=256`, `padded_vocab_size=256`, `seq_len=32`,
  `window_pattern="L"`.
- Pack fixture: 8 conversations, `seq_len=32`, `buffer_size=100`.
- Step fixture: 8 steps, `device_batch_size=1`, `grad_accum=2`, `seq_len=32`.
- Loop fixture: 8 steps, `num_iterations=8`, one evaluation at step 8.

## 6. Fixtures

### 6.1 `tests/data/sft_pack_fixture.bin`

The file is `NANCHAT1`, one `uint32` version, and UTF-8 JSON. The object holds:

- `settings`: `seq_len`, `buffer_size`, `render_max_tokens`, `rank`,
  `world_size`, and `stop_rule`.
- `tokenizer`: the artifact name, its SHA-256, the special-token map, and a
  `pieces` table from every fixture text fragment to its reference token ids.
- `conversations`: 8 conversations. Cover a SmolTalk-like conversation, an
  MMLU multiple-choice question, a GSM8K solution with tool parts, a system
  message, and one conversation above 2048 tokens.
- `render`: the reference `ids` and `mask` for each conversation.
- `pack`: the reference `inputs` and `targets`, one list per row.
- `mixture`: the synthetic task lengths and the reference index order (D8).
- `validation_mixture`: the same for the validation task set.

The reference rows come from a transcription of
`scripts/chat_sft.py:178-300`. Run it with the pinned `seq_len`, the pinned
`buffer_size`, `rank=0`, `world_size=1`, and the `last_step` stop rule.

### 6.2 `tests/data/sft_step_fixture.bin`

The file uses `fixture_format.FixtureWriter`. The records are:

- `config/*`: layers, heads, kv_heads, embd, seq, vocab, padded_vocab, and
  window pattern.
- `config/opt/*`: the four rates, `init_lr_frac`, `adam_eps`, and `clip`.
- `config/schedule/*`: the warmup ratio, the warmdown ratio, the final
  fraction, and the four Muon momentum fields.
- `data/step/<k>/{tokens,targets}`: the micro-batches of step `k`, shape
  `[grad_accum, device_batch_size, seq]`.
- `step/<k>/{loss,grad_norm,param_l2,lr,momentum}` for `k` = 1 to 8.
- `step/8/param/<name>`: the final parameters.

The test stages the initial parameters and the initial moments from
`tests/data/sft_base_fixture.ckpt`. The fixture records the reference
trajectory as data, so the runtime test never reads torch.

### 6.3 `tests/data/sft_loop_fixture.bin`

The file uses `fixture_format.FixtureWriter`. The records are:

- `config/num_iterations`, `config/rows`, and `config/eval_every`.
- `loop/<k>/{step,progress,epoch,lr_multiplier,momentum}` for 8 steps.
- `train/loss/<step>` for 8 steps.
- `val/bpb/8` and `chatcore/8` for the final evaluation.

### 6.4 `tests/data/sft_rows_fixture.nchsft` and the base checkpoint

The `sft` kind writes two more files from the same tiny model:

- `tests/data/sft_rows_fixture.nchsft`: the reference packed rows as an
  `NCSFT001` shard, shape `[rows, 32]`. The rows come from the pack fixture
  conversations. The writer is `nanochat_cpp.sft.pack`, so the generator adds
  `python/` to `sys.path`.
- `tests/data/sft_base_fixture.ckpt`: the reference initial parameters and
  optimizer moments in the C++ container, with step 0. Once D7 lands, the
  checkpoint also carries the recipe block.
  Write it through the path of `tools/convert_checkpoint.py`.

The shard and the checkpoint make R2 and R3 hermetic. Both loops consume the
same rows from the same start. The real mixture stays an optional check (D8).

## 7. Phases

| Phase | Owner | State | Gate |
|---|---|---|---|
| A. Freeze the contract | Architect | Done | Document review and the ASD-STE100 checker |
| B. Fixtures | Oracle | Open | The fixtures load on the CPU with no reference checkout |
| C. R1 data parity | Python surface, Oracle | C.0 done, C.1 open | `//python:sft_data_parity_test` at T0 |
| D. R2 arithmetic parity | Harness, Oracle | Open | T0, then T1, then T2 |
| E. R3 schedule | Architect, Optimizer, Harness | Open | `//tests:sft_loop_parity_test` at T0 |
| F. R3 reader and driver | Harness, Python surface | Partial | `//tests:sft_parity_test` at T2 |
| G. R3 and R4 parity | Integrator | Open | T2, then the `parity.md` update |

### Phase A. Freeze the contract (done)

The artifacts are the `P4` to `P7` rows in `docs/parity.md`, the status table in
`docs/post-training.md` section 11, and the tolerance section in
`docs/testing.md`. Keep them in step with this document.

### Phase B. Fixtures

1. Add `sft` and `sft_loop` to `COMMANDS` in `tools/fixture_common.py`, to the
   dispatch of `tools/dump_fixtures.py`, to its module docstring, and to the
   fixture-kind row of `docs/testing.md`.
2. Implement the `sft` kind. It writes the four files of sections 6.1, 6.2,
   and 6.4. Add a `--real-mixture` mode for D8, and mark it as not part of a
   gate.
3. Implement the `sft_loop` kind for section 6.3. It reads the shard and the
   base checkpoint, runs the reference loop, and records the trace and the
   loss curve.
4. Commit the five fixtures in `tests/data/`.
5. Print the checksum of every file, in the style of the `checkpoint` kind.
6. Add `tests/sft_fixture_load_test.cc` and the target
   `//tests:sft_fixture_load_test`. It opens the two record fixtures through
   `tests/oracle_fixture.h`, reads the shard with `LoadSftShard`, and loads the
   checkpoint with `Checkpoint::Load`.

Gate: this command parses the pack fixture, and the new load test reads the
other four files. Both run with no reference checkout and no torch.

```bash
python3 -c 'import json,sys; raw=open(sys.argv[1],"rb").read();
assert raw[:8]==b"NANCHAT1"; json.loads(raw[12:])' \
    tests/data/sft_pack_fixture.bin
tools/nanochat test //tests:sft_fixture_load_test
```

### Phase C. R1 data parity

C.0 (done, this change):

- `python/nanochat_cpp/chat.py`: `render_conversation` truncates at
  `max_tokens=2048`, asserts the alternating roles, and rejects an unknown
  content part.
- `python/nanochat_cpp/sft_data.py`: `PackedRowStream` mirrors the reference
  buffer, cursor, epoch, consumption, progress, and stop rule.
  `PACK_BUFFER_SIZE` is 100. `RENDER_MAX_TOKENS` is 2048.
- `python/tests/sft_data_test.py` compares the stream against a transcription
  of the reference generator over seeds, buffer sizes, batch sizes, and ranks.

C.1:

1. Add `python/tests/sft_data_parity_test.py` and the target
   `//python:sft_data_parity_test` in `python/BUILD.bazel`.
2. The test loads the pack fixture and builds a fixture tokenizer from the
   `pieces` table and the special-token map.
3. The test compares `render_conversation` ids and mask with `render`.
4. The test feeds the recorded ids and mask through `pack_rows` with an
   injected renderer, and compares `pack`.
5. The test compares the synthetic mixture order with `TaskMixture`.
6. The test compares the validation order and the two `stop` values.

Gate: `tools/nanochat test //python:sft_data_parity_test`, then
`tools/nanochat lint`.

### Phase D. R2 arithmetic parity

1. Apply D3, D4, and D5. D3 changes `include/nanochat/optim.h` and
   `src/optim.cc`. D4 and D5 change `src/sft_main.cc`. Add `--grad-accum`,
   `--warm-start`, and `--val-rows`.
2. Add `tests/sft_step_parity_test.cc`. Follow `tests/train_parity_test.cc`:
   read the fixture, stage the parameters and the moments from
   `tests/data/sft_base_fixture.ckpt`, drive
   `ForwardLoss`/`BackwardWeighted`/`Optimizer::Step` for the recorded steps,
   and compare `loss`, `grad_norm`, `param_l2`, `lr`, and `momentum`.
3. Add the three targets to `tests/BUILD.bazel`, named as follows:
   `//tests:sft_step_parity_test` (T0, CPU), `//tests:sft_step_parity_gpu_test`
   (T1, `gpu` and `manual` tags), and `//tests:sft_step_parity` (T2,
   standalone binary).
4. Extend `tests/numerics_trace.cc` with one SFT optimizer step, and record
   the post-step loss and gradient norm. Regenerate
   `tests/data/numerics_golden_10l.bin` and
   `tests/data/numerics_golden_10l_fp8.bin` with
   `//tests:numerics_trace_emit`. Update the tolerance table in
   `docs/numerics-integration.md` section 4.

Gate: `tools/nanochat test //tests:sft_step_parity_test` at T0, then
`tools/nanochat test --gpu //tests:sft_step_parity_gpu_test` at T1, then
`tools/nanochat test --gpu //tests:numerics_trace_cuda_test` at T2.

### Phase E. R3 schedule

1. Apply D1. D2 needs no code change; add a comment at the step counter of
   `src/sft_main.cc`.
2. Add `tests/sft_loop_parity_test.cc` and the target
   `//tests:sft_loop_parity_test`. Replay the loop fixture: compute the rate
   multiplier from `progress` and the momentum from the step, and compare both
   per step.
3. Add a unit test for the three branches of `LrMultiplierForProgress` in
   `src/optim_test.cc`.

Gate: `tools/nanochat test //tests:sft_loop_parity_test //src:optim_test`.

### Phase F. R3 reader and driver

The reader and the bridge landed as `src/sft_shard.{h,cc}` and
`python/nanochat_cpp/sft.py`. The next steps are:

1. Apply D7 to `include/nanochat/checkpoint_metadata.h`, `src/fields.cc`, and
   `src/checkpoint_metadata.cc`. Add a round-trip test to
   `//src:fields_test`.
2. Replace the step schedule in `src/sft_main.cc` with the progress stream.
   Read the recipe from the warm-start checkpoint. Use the accumulation of
   section 5.1. Evaluate at the cadence of section 5.4.
3. Add `--warm-start`, `--val-rows`, `--eval-every`, `--eval-tokens`, and
   `--chatcore-every` to `python/nanochat_cpp/sft.py:run`.
4. Implement `EvaluatePackedRows` per D6. Add its test to
   `src/eval_test.cc`.
5. Add `tests/sft_parity_test.cc` and `//tests:sft_parity_test`. It runs the
   driver over `tests/data/sft_rows_fixture.nchsft` from
   `tests/data/sft_base_fixture.ckpt`, and compares the loss curve, the
   bits-per-byte, and the final parameter L2.

Gate: `tools/nanochat test` at T0, then
`tools/nanochat test --gpu //tests:sft_parity_test` at T2, run at a wave
boundary.

### Phase G. R3 and R4 parity

1. Run the reference loop and `sft_main` over the same shard and the same base
   checkpoint, with the pinned settings.
2. Compare the loss curve, the validation bits-per-byte, ChatCORE, and the
   final parameter L2. Use the tolerances of `docs/testing.md`: loss `1e-5`,
   parameter L2 `1e-4`, fp32 CUDA about `1e-5`.
3. Set `P6` and `P7` in `docs/parity.md` and `docs/post-training.md`, and
   record the evidence.

Gate: T2 at a wave boundary, then the `parity.md` update.

The warm start belongs to R2. Phase F consumes it and does not repeat it.

## 8. Interim label

Use these names until Phase G closes:

- After Phase C: "SFT data parity" (`P4`).
- After Phase D: "SFT arithmetic parity" (`P5`), with the GPU leg.
- Until Phase G: "SFT loop parity" (`P6`) and "SFT validation parity" (`P7`)
  stay `open`.

## 9. State of the tree

| Item | State | Location |
|---|---|---|
| SFT renderer | Done, pinned | `python/nanochat_cpp/chat.py` |
| SFT mixture and packer | Done, pinned | `python/nanochat_cpp/sft_data.py` |
| Packer parity against a reference transcription | Done | `python/tests/sft_data_test.py` |
| Native packed-row reader | Done | `src/sft_shard.{h,cc}`, used by `src/mesh_train.cc` |
| SFT driver | Partial | `src/sft_main.cc`: no progress schedule, no warm start, no accumulation, no evaluation |
| SFT bridge | Partial | `python/nanochat_cpp/sft.py` |
| Packed-row validation | Missing | `src/eval.cc` reads parquet documents only |
| Recipe metadata | Missing | `include/nanochat/checkpoint_metadata.h` |
| SFT fixtures | Missing | `tests/data/sft_*_fixture.bin` |
| Data parity gate | Missing | `//python:sft_data_parity_test` |
| Arithmetic parity gate | Missing | `//tests:sft_step_parity_test` |
| Schedule parity gate | Missing | `//tests:sft_loop_parity_test` |
| Loop parity gate | Missing | `//tests:sft_parity_test` |

## 10. Risks

- Mixture order. The real hub order needs a network download and a pinned
  `datasets` version. D8 keeps the gate hermetic. Verify the real order by
  hand once, with `--real-mixture`.
- Tokenizer drift. The pack fixture pins the tokenizer hash and the text
  pieces. A new tokenizer artifact invalidates the fixture. Regenerate it in
  the same change.
- Progress source. The C++ stream must report the same progress as the
  reference, including the monotone maximum. The loop fixture pins 8 steps
  only. A long run can still differ at the tail.
- Validation length. The evaluator reads one pass. A validation shard shorter
  than `eval_steps * device_batch` needs a restart rule. Section 5.4 forbids
  the condition for the pinned run.
- GPU budget. R2, R3, and R4 are T2 runs. Schedule them at a wave boundary,
  not for every commit.
- Environment. The reference environment has torch and pyarrow. The optional
  `--real-mixture` mode also needs `datasets`, which is absent today.

## 11. Non-goals

- A distributed SFT run.
- A preference-optimization method.
- A native chat template in the C++ model layer.
