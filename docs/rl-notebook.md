# Reinforcement learning in a notebook

Status: partly delivered. The hand-rolled loop, the one-call facade, the
persistent worker, and the three-stage chain of section 9 exist. The
multi-prompt C ABI binding and the tool-forcing driver are not wired yet.

This document gives the plan to run reinforcement learning (RL) in a notebook.
`docs/post-training.md` section 12 and `docs/training-seam.md` section 8 hold the
same plan from the design side. This document is the notebook view. Section 9
gives the one-job chain that runs this call last, after base training and SFT.

## 1. Purpose

A notebook user must write the data policy and the reward policy. The user must
not write the training mechanism. One call must start an RL run. The C++ driver
must hold the model and the optimizer.

## 2. Today: compose it by hand

A user can write an RL loop today with the public API. The loop is long.

```python
import nanochat_cpp as nc
from nanochat_cpp import chat

model = nc.Model(nc.Config(...))
opt = nc.Optimizer(model, num_iterations=steps)
task = chat.build_task("gsm8k")
stop = tokenizer.encode_special("<|assistant_end|>")

for step in range(1, steps + 1):
    ids, targets, weights = [], [], []
    for conv in prompts:
        prompt = render_for_completion(conv)
        rows = model.generate(prompt, num_samples=4, temperature=1.0,
                              stop_ids=[stop])
        rew = [task.reward(conv, tokenizer.decode(r.tokens)) for r in rows]
        mean = sum(rew) / len(rew)
        for row, r in zip(rows, rew):
            # Build tokens = prompt + row tokens. Pad to seq_len.
            # Set targets = -1 on the prompt. Set weight = r - mean.
            ...
    model.forward_loss(ids, targets)
    model.zero_grad()
    model.backward_weighted(weights, scale=1.0 / valid_tokens)
    opt.step(step)
```

The loop has six problems.

1. The user writes the mechanism and duplicates `TrainLoop`.
2. `render_for_completion` is private (`python/nanochat_cpp/chat.py`).
3. Only GSM8K has a `reward` method (`python/nanochat_cpp/tasks.py`).
4. `generate` takes one prompt and many samples. It does not do multi-prompt
   rollout.
5. The loop crosses the application binary interface (ABI) every step.
6. The divisor lives in Python, not C++. This breaks the single-source rule.

## 3. The target: one call

The plan gives the notebook one facade call. The call follows the proposed
supervised fine-tuning call in `docs/training-seam.md` section 6.5.

```python
nc.rl.run(
    task="gsm8k",
    prompts=conversations,
    num_samples=4,
    steps=1000,
    config=shape,                       # the model shape, as model flags
    warm_start="chatsft.nchkpt01",      # the load path
    checkpoint="chatrl.nchkpt01",       # the save path
    checkpoint_interval=100,
)
```

The user writes prompts and selects a task. The user does not write a loop.
Python still owns the reward, so Python stays in the control loop. The mechanism
moves into a persistent C++ worker.

Three arguments carry three separate facts, and the second is why the call is
safe to make at all:

- `warm_start` is the container the worker loads the weights and the optimizer
  moments from. It reaches the worker as `--checkpoint`.
- `checkpoint` is the container the worker saves to, and nothing else. A save
  path never becomes a load path, so a stage that warm starts from the SFT
  checkpoint and saves to its own file keeps its parent intact (blocker B3 of
  `docs/pipeline-plan.md`). The legacy rule that derived the load path from
  `checkpoint` is removed, and `docs/python.md` section 3.4 records that as a
  breaking change.
- `config` is the model shape. It reaches the worker through
  `nanochat_cpp.api.model_flags`, the one model flag list on the Python side, so
  the worker rebuilds the shape of the run it warm starts from and the two
  stages cannot drift (B1 and B2 of the same plan).

## 4. The worker

A long-lived process holds the model and the optimizer. It speaks a pipe
protocol with the Python bridge. One step has seven parts.

1. Python renders a batch of prompts.
2. Python sends a `rollout` message.
3. The worker generates rows and returns them.
4. Python scores the rows and computes the advantage.
5. Python sends an `advantage` message with weights.
6. The worker runs one RL step. The step is `ForwardLoss`, then
   `BackwardWeighted`, then `Optimizer::Step`.
7. The worker returns the loss and the gradient norm.

A `rl_step` binary shares the step logic. It reads a fixture. It is the
correctness gate. The project trusts the worker only after
`//tests:rl_parity_test` passes at tier T2.

## 5. The phases, and what shipped

The phases come from `docs/training-seam.md` section 8. The owners follow
`AGENTS.md` section 1. Phases 1 to 4 are delivered. Phase 5 is delivered in
C++ but its C ABI binding is not. Phase 6 is delivered in part.

1. **Add `nanochat_rl_step` to the C ABI.** Add the call to
   `include/nanochat/capi.h` and the binding. It maps to `BackwardWeighted` and
   the divisor. The divisor moves into C++ per `docs/training-seam.md` section
   5.7. Owner: architect for the header, Python surface for the binding.
2. **Add the `rl_step` binary and the parity fixture.** Add the file-driven
   binary. Add `tools/dump_fixtures.py rl`. Gate: `//tests:rl_parity_test` at tier
   T2. Owner: harness for the step, oracle for the fixture.
3. **Add the persistent worker.** Add the worker binary and the pipe protocol.
   Gate: an RL smoke run at tier T2. Owner: harness.
4. **Add the Python bridge.** Add `nc.rl` next to `nc.sft`. Add the RL command
   to the command table in `python/nanochat_cpp/_entry.py` and `toolchain.py`.
   Owner: Python surface.
5. **Add multi-prompt rollout.** Extend the C ABI generation to many prompts
   with an independent stop for each row. The compute and the header are in
   place. The C ABI shim in `bindings/` is still owed. Owner: architect for
   `capi.h`, and the generation owner for the compute.
6. **Add tool forcing.** Stream the `chat_engine` protocol. Force the
   tool-output tokens in the generation loop. The C++ hook and the Python
   driver exist. The driver is not wired to a tested path, so parity item P1
   stays open in `docs/parity.md`. Owner: workflow and Python surface. This
   step is optional.

## 6. Notebook concerns

- **Kaggle.** The notebook sets `NANOCHAT_SANDBOX_BACKEND=none`
  (`docs/python.md` section 3.1). In this mode the facade must start the worker
  directly. It must not call `tools/nanochat`, because Kaggle has no systemd
  user manager.
- **Workstation.** The facade calls `tools/nanochat` with a new profile and
  command, for example `tools/nanochat train -- rl_worker ...`. The GPU queue
  and the sandbox apply.
- **Checkpoints.** The `rl` path already resolves to `chatrl_checkpoints`
  (`python/nanochat_cpp/checkpoint.py`). A warm start and the optimizer state
  use the checkpoint container.
- **Sizes.** A notebook GPU is small. Keep the rollout batch and `num_samples`
  small. The plan caps the shapes at tier T1 and tier T2.

## 7. Convenience levels

| Level | What the user writes | Available |
|---|---|---|
| Hand-rolled | The full loop | Yes, today |
| Facade | `nc.rl.run(...)`, no loop | Delivered |
| Batched and fast | The same call, many prompts | Delivered in C++; the C ABI shim is not |
| Tool-forced parity | The same call | The mechanism exists; not yet wired |

## 8. Open items

Two items stay open.

1. The multi-prompt C ABI call `nanochat_generate_multi` has no binding, so a
   Python caller cannot reach the batched rollout through the C ABI.
2. The tool-forcing driver (`ChatEngine` and `use_calculator`) is not wired to
   a caller or a test, so `docs/parity.md` still records parity item P1 as
   open.

The objective is already golden-gated. The fixture is
`tests/data/numerics_golden_10l.bin`. The gate is `//tests:numerics_trace_test`.
The correctness risk is therefore low. The remaining work is the interface and
the orchestration.

## 9. The chain: one job file, three stages

`nc.rl.run` is the last of three stages, and one job file drives all three on
any host. The chain is `python/nanochat_cpp/pipeline.py`, and it holds no
training logic, no model flag list of its own, and no host name (design decision
D1 of `docs/pipeline-plan.md`). Each stage calls one facade and reads the
checkpoint of the stage before it:

| Stage | Facade |
|---|---|
| base | `runtime.MeshTrainConfig` |
| SFT | `sft.pack`, then `sft.run` |
| RL | `rl.run` |

```python
# base: the mesh driver at world_size = 1, over the fixture corpus.
runtime.MeshTrainConfig(config(job), steps=job.base_steps, ...)
# SFT: one packed shard, the base checkpoint as the load path.
sft.run(shard, ckpt, config=config(job), warm_start=base.checkpoint,
        num_iterations=job.sft_steps)
# RL: the SFT checkpoint as the load path, its own file as the save path.
rl.run(reward, prompts, num_samples=job.num_samples, steps=job.rl_steps,
       config=config(job), warm_start=sft.checkpoint, checkpoint=rl_ckpt,
       max_tokens=job.max_tokens)
```

`config(job)` is `pipeline.PROFILES[job.profile]`, so the shape has one home;
`PROFILES` holds the `demo` profile of decision D5, and every stage receives it
as an `nanochat_cpp.api.Config`, which `api.model_flags` turns into the flags
the stage binaries parse. The load path and the save path stay separate at every
stage (B3), so each checkpoint survives the stage that reads it.

A job is plain data with a JSON round trip: `run_id`, `profile`, `base_steps`,
`sft_steps`, `rl_steps`, `num_prompts`, `num_samples`, `max_tokens`, `seed`,
`out_root`, `corpus`, `deliver`, `temperature`, and `top_k`. `out_root` is empty
in a committed job, because the host that owns a run root fills it: the Colab
notebook sets `NANOCHAT_RUNS_DIR`, and the Modal app rewrites it to
`/runs/<run_id>`.

The reward is a class with a `reward` method, which `rl.run` accepts directly,
so no Hub dataset loads. It is graded: 1.0 for the right number, 0.5 for the
`####` marker, and otherwise the character overlap with the reference. A graded
reward keeps the advantages of a group from collapsing into zeros, and the seed
mix of B4 makes the four samples of a group differ in the first place, so the
advantage vector is not zero either.

`pipeline.run(job)` runs the three stages and returns a `Report`, and
`python -m nanochat_cpp.pipeline <job>` is the same call from a command line.
`deliver()` writes the two artifacts, and `//python:pipeline_test` asserts A1 to
A9 of `docs/pipeline-plan.md` section 7 over the `demo` profile, so the
assertions and not the loss values are the contract.

### 9.1 The delivered artifacts

`deliver()` writes the contract of decision D6. Five paths name it:

```
<run_root>/<run_id>/base/checkpoint.ckpt
<run_root>/<run_id>/sft/checkpoint.ckpt
<run_root>/<run_id>/rl/checkpoint.ckpt
<run_root>/<run_id>/deliver/chatrl.ckpt
<run_root>/<run_id>/deliver/pipeline_report.json
```

`deliver/chatrl.ckpt` is the RL checkpoint copied byte for byte, and the report
is the review artifact: the stage order and the step counts, the loss traces and
the mean rewards, the checkpoint digests and the metadata blocks, the seconds
and the seeds, the model shape and the tokenizer digest, and the host block of
`backend`, `arch`, `precision`, and `revision`. A reviewer reads the report and
the checkpoint, not the run.

The copy is verified against the digest the stage recorded. One field is empty
by construction: `nc.sft.run` returns an exit status and not a loss trace, so
the SFT stage says so in `extra` and the report carries an empty trace for it.
The RL stage reads its rewards through `on_step`, which receives the per-rollout
rewards, so it records the mean reward and the count of groups whose advantages
are all zero.

### 9.2 The two entry forms

The entry is always the tool, never a bare module (decision D10), because both
stage executables call `RequireSandboxOrDie`, and a notebook's
`NANOCHAT_SANDBOX_BACKEND=none` satisfies the library only. Two forms exist:

```
# The host form: one `train` scope, or the `none` backend of a container.
tools/nanochat train -- python -m nanochat_cpp.pipeline <job>

# The container form: the owner routes the work through the GPU queue.
tools/nanochat client --profile train -- python -m nanochat_cpp.pipeline <job>
```

The first form is the one the Colab notebook and a workstation use; the second
is the one the Modal `pipeline` function uses. `docs/host-portability.md`
section 9 gives the three hosts and what each one records.
