# Reinforcement learning in a notebook

Status: proposed. The mechanism exists in parts. The notebook path does not.

This document gives the plan to run reinforcement learning (RL) in a notebook.
`docs/post-training.md` section 12 and `docs/training-seam.md` section 8 hold the
same plan from the design side. This document is the notebook view.

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
    checkpoint="chatrl.nchkpt01",
    checkpoint_interval=100,
)
```

The user writes prompts and selects a task. The user does not write a loop.
Python still owns the reward, so Python stays in the control loop. The mechanism
moves into a persistent C++ worker.

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

## 5. Work to build, in order

The phases come from `docs/training-seam.md` section 8. The owners follow
`AGENTS.md` section 1.

1. **Add `nanochat_rl_step` to the C ABI.** Add the call to
   `include/nanochat/capi.h` and the binding. It maps to `BackwardWeighted` and
   the divisor. The divisor moves into C++ per `docs/training-seam.md` section
   5.7. Owner: architect for the header, Python surface for the binding.
2. **Add the `rl_step` binary and the parity fixture.** Add the file-driven
   binary. Add `tools/dump_rl_fixture.py`. Gate: `//tests:rl_parity_test` at tier
   T2. Owner: harness for the step, oracle for the fixture.
3. **Add the persistent worker.** Add the worker binary and the pipe protocol.
   Gate: an RL smoke run at tier T2. Owner: harness.
4. **Add the Python bridge.** Add `nc.rl` next to `nc.sft`. Add the RL command
   to the command table in `python/nanochat_cpp/_entry.py` and `toolchain.py`.
   Owner: Python surface.
5. **Add multi-prompt rollout.** Extend the C ABI generation to many prompts
   with an independent stop for each row. The current call is one prompt with
   `num_samples` rows. Owner: architect for `capi.h`, and the generation owner
   for the compute.
6. **Add tool forcing.** Stream the `chat_engine` protocol. Force the
   tool-output tokens in the generation loop. This step closes parity item P1.
   Owner: workflow and Python surface. This step is optional.

## 6. Notebook concerns

- **Kaggle.** The notebook sets `NANOCHAT_SANDBOX_BACKEND=none`
  (`docs/python.md` section 3.1). In this mode the facade must start the worker
  directly. It must not call `tools/nanochat`, because Kaggle has no systemd
  user manager.
- **Workstation.** The facade calls `tools/nanochat` with a new profile and
  command, for example `tools/nanochat train -- rl_worker ...`. The GPU broker
  and the sandbox apply.
- **Checkpoints.** The `rl` path already resolves to `chatrl_checkpoints`
  (`python/nanochat_cpp/checkpoint.py`). A warm start and the optimizer state
  use `NCHKPT01`.
- **Sizes.** A notebook GPU is small. Keep the rollout batch and `num_samples`
  small. The plan caps the shapes at tier T1 and tier T2.

## 7. Convenience levels

| Level | What the user writes | Available |
|---|---|---|
| Hand-rolled | The full loop | Yes, today |
| Facade | `nc.rl.run(...)`, no loop | Needs phases 1 to 4 |
| Batched and fast | The same call, many prompts | Needs phase 5 |
| Tool-forced parity | The same call | Needs phase 6 |

## 8. Current blockers

Three items block a convenient notebook today.

1. There is no `rl_step` binary and no `nc.rl` facade.
2. There is no persistent worker, so the loop crosses the ABI every step.
3. `generate` is single-prompt only, so rollout throughput is low.

The objective is already golden-gated. The fixture is
`tests/data/numerics_golden_10l.bin`. The gate is `//tests:numerics_trace_test`.
The correctness risk is therefore low. The remaining work is the interface and
the orchestration.
