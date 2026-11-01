# The pipeline scale-up plan: a bigger chain and two Modal legs

Status: ready to execute. An oracle review corrected six defects in an earlier
draft, and this version carries the corrections. The corrections are the eval
batch, the eval count, the Modal function names, the build flags, the roofline
target, and the hardware-check claim. Section 9 lists the work.

This plan extends [pipeline-plan.md](pipeline-plan.md), which delivers the chain
itself: base training, supervised fine-tuning (SFT), then reinforcement learning
(RL), and the `deliver/` artifact contract. This document covers four changes:

- a bigger model at a longer schedule;
- one new stage, evaluation;
- a second Modal leg, on an H100 at bf16;
- a performance check for each leg.

Read [post-training.md](post-training.md) for the RL design,
[host-portability.md](host-portability.md) for the host contract, and
[fp8-training.md](fp8-training.md) for the 8-bit tier that this plan does not
use.

## 1. Goal

One job file runs the whole chain, and the chain ends with a measured result. On
a local CPU the chain stays small, so a person can watch it. On Modal it runs two
legs: a T4 at fp32, then an H100 at bf16. Each leg writes one report. The entry
point compares the two reports and fails when a gate fails.

The H100 leg runs at bf16 only. This plan writes no fp8 training path. Section 4
gives the reasons.

### 1.1 Where the work runs

The chain does not exist on `main` yet. It lives on branch `feat/pipeline-plan`,
in the worktree `.sliceme/worktrees/campaign-feat--pipeline-plan`.

- Rows 3 to 7 run in that worktree, because rows 3 and 5 edit
  `python/nanochat_cpp/pipeline.py`, which only that branch holds.
- This document travels with the chain. Commit it in that worktree, rebased on
  the branch that holds the Modal app.
- Row 1 and row 2 touch files that `main` holds. Land them first, or rebase the
  chain branch on them.

## 2. What the tree holds today

| Item | Today | Path |
|---|---|---|
| Chain module | base, then SFT, then RL; no eval | `python/nanochat_cpp/pipeline.py` |
| Shape | `demo`: 2 layers, hidden 64, sequence 64 | `PROFILES` |
| Schedule | 100, 100, and 5 steps | `Job` |
| Corpus | committed fixture, 10 documents, 109 bytes | `build_corpus` |
| Base batch | one row, because one epoch holds 119 tokens | `BASE_BATCH` |
| Modal | one image, T4, `sm_75`, fp32 | `tools/modal/app.py` |
| Build precisions | `fp32` and `bf16` | `python/nanochat_cpp/_build.py` |
| Device table | the `h100` row declares bf16 and fp8 | `backends/cuda/caps.json` |
| Measured chain | about 42 ms per step on a T4, for the `demo` shape | section 14 |

Four facts come from `tools/nanochat roofline`:

- The demo shape has a 0.366 ms step bound on a T4 at fp32.
- The standard shape has a 3.600 ms step bound on a T4 at fp32.
- The standard shape has a 0.764 ms step bound on an H100 at bf16.
- Every row of those reports reads `limit: launch`. The chain is bound by
  launches and by the host program, not by arithmetic.

## 3. The three traps

1. `--config=h100` in `.bazelrc` is the simulator config. It sets
   `--define=simulator=1` and `--define=collective=native`.
2. The build helper has no `fp8` name in `SUPPORTED_PRECISIONS`. A build with
   `NANOCHAT_CPP_PRECISION=fp8` fails before Bazel starts.
3. `bindings/nanochat_capi.cc` prints the text `precision (fp32)` in its error
   message, whatever the build precision is. A bf16 failure therefore reports
   the wrong precision.

## 4. Decisions

| # | Decision |
|---|---|
| P1 | Add a second profile, `standard`. Keep `demo` for the CPU test loop. `Job()` defaults to `standard`. |
| P2 | Keep the sequence length at 64 and the base batch at one row. The committed corpus fills one 65-token row per epoch. |
| P3 | The H100 leg runs at bf16. No leg runs an fp8 training path. |
| P4 | Add the eval stage in process, through `api.evaluate`, `api.Model.load`, and `api.Tokenizer`. Add no third stage executable. |
| P5 | The eval stage reads the base checkpoint and the RL checkpoint, so the report holds both numbers. |
| P6 | Run T4 first, then H100, as four functions over two images. One library per image keeps the architecture and the precision honest. |
| P7 | Add one named config, `hopper`, for a hand run only. The build helper keeps its explicit flag triple. |
| P8 | Gate the H100 leg on evidence that bf16 ran, and on a comparison with the T4 leg. Do not gate on an fp8 speedup, and do not gate on MFU. |
| P9 | Change no frozen header. `caps.json` already declares bf16 for the H100. |

### 4.1 Why the H100 leg runs at bf16

The T4 has no bf16 and no fp8, so the T4 leg stays at fp32. The H100 declares
bf16 and fp8. This plan takes bf16 for three reasons:

- The build helper supports bf16 today. The H100 leg therefore adds no new build
  precision, and no risk to the build path.
- The chain is bound by launches at this shape. Section 14 gives a 0.76 ms device
  bound inside a step of tens of milliseconds, so the precision does not change
  the measured time.
- The bf16 leg on an H100 still runs the fp8 operators. The startup hardware
  check keys on the card, not on the build. So the leg proves the fp8
  conversion, the scaled fp8 GEMM, and the attention tile on Hopper silicon.

What the leg does not prove is the fp8 training graph and the fp8 planner rule.
Those belong to a later row, and the row needs a Hopper benchmark first.

## 5. The shape and the schedule

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

## 6. The eval stage

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

## 7. The Modal legs: T4 first, then H100 at bf16

`tools/modal/app.py` gains one record, `Leg`, with these fields: name, card
type, card count, architecture, precision, and build configuration.

| Leg | Card | Architecture | Precision | Build command |
|---|---|---|---|---|
| `t4` | `T4` | `sm_75` | `fp32` | `--config=cuda --define=precision=fp32 --@rules_cuda//cuda:archs=sm_75` |
| `h100` | `H100` | `sm_90` | `bf16` | `--config=cuda --define=precision=bf16 --@rules_cuda//cuda:archs=sm_90` |

The build helper emits the flag triple of the table today
(`_build._build_command`). It never emits a named config. The `hopper` config of
P7 is a shorthand for a hand run only.

Both legs keep the default collective, `nccl`, because the chain runs
`world_size=1`. The `h100` simulator config would change the collective to
native; the hardware leg must not use it.

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
- The image build adds `//src:roofline_main`. Gate G3 runs the tool inside the
  container, and the tool must not start a second Bazel build there.

`notebooks/target.py` gains a `leg` argument on `ModalTarget`, so a notebook
selects one leg.

The T4 leg runs first. The T4 artifact therefore survives an H100 failure, and a
reviewer reads the cheap leg while the expensive leg runs.

## 8. The performance check

Decision P8 puts the check in two places. The H100 function records the
prediction. The entry point gates on the comparison.

| Gate | Claim | Check |
|---|---|---|
| G1 | bf16 ran on the H100 | `host.precision` is `bf16`, the RL checkpoint metadata records `bf16`, and `host.arch` is `sm_90` |
| G2 | the H100 leg did not regress | the H100 base `step_ms` is at most 1.25 times the T4 base `step_ms` for the same job, with the same `cpu=` value |
| G3 | the run matches the model | the H100 container runs `roofline --device h100 --precision bf16 --json` for the job shape and records `predicted_step_ms` and `limit`; the measured ratio must be inside the recorded band |

G2 is a smoke check, not a throughput claim. Both legs are bound by the host
program, so the gate catches a large regression and a broken device path. It does
not measure the card.

G3 starts with the band `[1, 200]`. The first two-leg run records the measured
ratio in section 14. The row then narrows the band to twice the measured value.

The report records two numbers, and no gate reads them:

- MFU. At 256 tokens per step, the device bound is about 1 ms inside a step of
  tens of milliseconds. MFU therefore reads near 0.01 percent. The step time of
  section 14 is an assumption until row 9 measures it.
- The cross-leg speed ratio. The chain is bound by launches at this shape, so a
  ratio below one is correct and a ratio above one is also correct.

The two legs exercise different branches of the startup hardware check. The
check keys on the device capability (`src/hw_gate.cc`), so the T4 leg reports a
skip and the H100 leg runs the fp8 conversion, the scaled fp8 GEMM, and the
attention cases.

## 9. Work rows

Each row ends with `tools/nanochat check` green. Rows 1 and 2 are independent,
and both can land before the chain branch.

Row 0 is not part of the change. The lint gate fails on the branch today, with
448 non-ASCII characters in 20 tracked C++ files. The characters are em dashes
in comments, and `tools/nanochat lint` reports them. The merge gate of
`AGENTS.md` section 5.5 runs that command, so a merge needs row 0 first. Row 0
changes comments only and moves no arithmetic.

| # | Work | Files | Gate | Size |
|---|---|---|---|---|
| 0 | Replace the non-ASCII characters in the tracked C++ comments with ASCII | 20 files, comments only | `tools/nanochat lint` | small |
| 1 | Add the `hopper` config for a hand run | `.bazelrc`, `python/tests/build_test.py` | `//python:build_test`, then an `sm_90` bf16 build on this host | small |
| 2 | Build the precision text from `PrecisionName(kBuildPrecision)` | `bindings/nanochat_capi.cc`, `bindings/nanochat_capi_test.cc` | `//bindings:nanochat_capi_test` | small |
| 3 | Add the `standard` profile, the new `Job` fields, report version 2, and the example job | `python/nanochat_cpp/pipeline.py`, `notebooks/pipeline_job.json`, `tools/modal/app.py` | `//python:pipeline_test` | small |
| 4 | Record `step_ms`, `tok/s`, `mfu`, and the hardware-check facts of the base stage | `python/nanochat_cpp/pipeline.py` | `//python:pipeline_test` | small |
| 5 | Add the eval stage and assertions A10 to A13 | `python/nanochat_cpp/pipeline.py`, `python/tests/pipeline_test.py` | `//python:pipeline_test` | medium |
| 6 | Add the two legs, the two images, the four functions, the comparison, and gates G1 to G3 | `tools/modal/app.py`, `tools/modal/modal_app_test.py` | `//tools/modal:all` | medium |
| 7 | Add the client leg argument | `notebooks/target.py`, `notebooks/target_test.py` | `//notebooks:all` | small |
| 8 | Update the documents of section 11 | `docs/**` | `tools/nanochat lint` | small |
| 9 | Run both legs, record the numbers, and narrow the G3 band | `docs/pipeline-plan.md`, this document | `modal run tools/modal/app.py::pipeline_demo` | manual |

Row 1 carries one unproven step. This host has no `sm_90` card, so the row proves
the build only. Row 9 is the first run on a real H100.

## 10. New assertions

The existing gate keeps A1 to A9 on the `demo` profile.

| # | Assertion | Catches |
|---|---|---|
| A10 | The eval stage ran, both `bpb` values are finite and inside `[0, 8]`, and each sample is non-empty text | A silent eval, and a wrong `eval_batch` |
| A11 | The RL checkpoint digest is the same before and after the eval stage | An eval that writes |
| A12 | The report names the evaluated checkpoint, its digest, the corpus path, the batch size, and the batch count | An unreviewable eval |
| A13 | `deliver/chatrl.ckpt` holds the RL weights, and its metadata reports stage `rl` | A delivery that follows the wrong stage |

A13 guards the decision in section 6.

## 11. Documents to update

- `pipeline-plan.md`: D3 becomes a sequence of two legs. Add the `standard`
  profile, the eval stage, the two legs, and gates G1 to G3 to sections 1, 5, 6,
  7, 8, and 11.
- `host-portability.md`: the Modal host block names two legs, two images, and
  two precisions.
- `build.md`: the `hopper` config, printed beside the `h100` simulator config,
  with one sentence on the difference.
- `fp8-training.md`: the status line says "the tree holds no fp8 arithmetic".
  That claim is false today, because `include/nanochat/fp8.h`,
  `backends/cuda/kernels/fp8.cu`, and three fp8 tests exist. Correct the line.
  Add one sentence on the Hopper hardware check, which runs the fp8 operators
  under a bf16 build.
- `precision.md`: name the chain as the integration leg of the bf16 tier.
- `performance-model.md`: add the chain numbers of section 14, and the sentence
  that this shape is bound by launches.
- `README.md`: one row for the two-leg run.
- `README.md` of `docs/`: one index row for this document.

## 12. Risks

- No row proves the `sm_90` bf16 build until row 1 runs here and row 9 runs on
  Modal. Row 1 catches a compile error on this host.
- The H100 image adds a second build and a second cached image. The first deploy
  is slow, and later deploys use the cache.
- Two legs double the run time. Each chain is under one minute, so the cost is
  the container start, not the training.
- The two legs are not bit-identical. Compare `bpb` and step times with a
  tolerance. Never compare digests across legs.
- The eval is thin. One batch of 65 tokens is the most the corpus allows, so a
  `bpb` difference between base and RL is small.
- The eval stage loads a second model in the chain process. On the 11 GB local
  card this is small, but the process holds two model arenas.
- The corpus limits the sequence length, the batch, and the eval count. A larger
  corpus needs a producer and one extra image dependency.
- The hardware check runs on the H100 with `hw_check=auto`. A failure there is a
  finding about the card or the build, not a defect of the chain.
- A bf16 leg on an H100 proves the Hopper path and the fp8 operators. It does not
  prove the fp8 training graph.
- The merge gate fails on the branch today, for reasons that predate this plan.
  Row 0 clears it, and row 0 touches comments only.

## 13. Non-goals

- The fp8 training graph, the fp8 planner rule, and an fp8 leg.
- A Hub corpus, a held-out validation split, and a bigger eval count.
- Chat accuracy through `ChatEvaluator`.
- A third stage executable for eval.
- Distributed training of any stage.
- A gate on the fp8 speedup or on MFU.

## 14. The numbers and how the writer measured them

The writer ran `tools/nanochat roofline` on the tags of this plan. The command
for one row:

```
tools/nanochat roofline --device h100 --precision bf16 \
    --layers 8 --heads 8 --kv-heads 4 --hidden 512 --seq 64 \
    --vocab 487 --padded-vocab 512 --window-pattern L --batch 1 --json
```

| Shape | Card | Precision | Step bound | Limit |
|---|---|---|---|---|
| `demo` 2/64 | T4 | fp32 | 0.366 ms | launch |
| `demo` 2/64 | H100 | bf16 | 0.173 ms | launch |
| `standard` 8/512 | T4 | fp32 | 3.600 ms | launch |
| `standard` 8/512 | H100 | bf16 | 0.764 ms | launch |

The measured chain of section 2 comes from the two-host harness of
`pipeline-plan.md` section 2. That run did 205 steps in 8.60 seconds on a T4,
or about 42 ms per step, at the `demo` shape.

Row 9 supplies the measured step time for the `standard` shape on both legs. This
document holds no fitted value.
