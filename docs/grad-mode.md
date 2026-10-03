# Grad mode: a forward-only path for evaluation

## 1. Purpose

`EvalBpb` must measure bits per byte with a forward pass only. The reference
`nanochat/loss_eval.py::evaluate_bpb` uses `@torch.no_grad()`. The current C++
evaluation path does not. This document describes grad mode, the equivalent
mechanism.

Grad mode lets the model skip the buffers that the backward pass needs. The
evaluation path then fits at the reference batch size.

This document describes the design. The implementation is in `src/model.cc`,
`src/eval.cc`, and `src/grad_mode_test.cc`. See [model.md](model.md) for the
graph inventory and [eval.md](eval.md) for the evaluation contract.

## 2. Problem

`EvalBpb` (`src/eval.cc`) calls `model->ForwardLoss(...)`. `ForwardLoss`
(`src/model.cc`) is the training forward. It saves each activation that
`Backward()` reads.

`BuildWorkspace` (`src/model.cc`) reserves one arena. The arena holds the whole
training graph. `Workspace::Reserve` (`src/workspace.h`) calls `cudaMalloc` one
time. The allocation fails as a whole before the compute starts.

The arena contains three groups:

| Group | Buffers | Size |
|---|---|---|
| Layer activations | `lacts_[0..L-1]` | `L * 20 * R * hidden` |
| Classifier | `raw_logits_`, `dlogits_` | `2 * R * padded_vocab` |
| Backward and global scratch | `dx_*`, `g_a_`, `g_b_`, `block_scratch_` | `33 * R * hidden` |

Here `R` is `batch * seq`. For `R = 32768`, `hidden = 768`, `L = 12`, and
`padded_vocab = 32768`, the total is about 36 GB in fp32. The fp16 build halves
that value. The 11 GB card cannot hold either value.

## 3. Mechanism

Grad mode is a boolean state on the model. Grad mode starts in the enabled
state. Training runs with grad mode enabled. Evaluation disables grad mode for
the duration of the measurement.

Two model methods control the state. A guard object restores the previous state.

```cpp
class Model {
 public:
  virtual void SetGradEnabled(bool enabled) = 0;
  virtual bool grad_enabled() const = 0;
};
```

The guard mirrors `torch.no_grad()`:

```cpp
class NoGradGuard {
 public:
  explicit NoGradGuard(Model* model);
  ~NoGradGuard();
};
```

`EvalBpb` creates one guard at the top. The guard restores the previous mode
after the measurement. The caller needs no other change.

## 4. Workspace split

The model keeps two arenas. `workspace_` serves the training graph.
`eval_workspace_` serves the evaluation graph. Two arenas avoid a rebuild when
training and evaluation alternate. `Workspace::Reserve` reallocates the whole
arena and invalidates each pointer. A rebuild at each evaluation step would cost
time.

`ForwardLoss` selects the arena from the grad-mode state. `BuildEvalWorkspace`
allocates a smaller set:

- one block of activations, not `L` blocks;
- no classifier gradient buffer `dlogits_`;
- no backward scratch;
- a logits buffer of `chunk_rows * padded_vocab`, not `rows * padded_vocab`.

The model caches each arena on `(batch, seq)`. The model rebuilds an arena only
when the shape changes or the mode changes.

## 5. Forward core

One private method holds the shared topology:

```cpp
void TrainModel::RunForward(const int* tokens, const int* targets, int batch,
                            int seq, bool save_for_backward);
```

The method runs the same sequence in both modes:

1. `EmbeddingForward` and `RmsNormForward`.
2. `SmearForward`.
3. The `BlockForward` loop.
4. `BackoutForward` and `RmsNormForward`.
5. `LinearForward` and `ClassifierForward`.

In training mode, the loop writes to `lacts_[i]` for each layer. It keeps every
layer because `Backward()` reads them in reverse order.

In evaluation mode, the loop writes to one `eval_block_` set. Only the current
layer stays alive. The residual stream ping-pongs between two buffers. The
forward-only caller leaves `q_pre` and `k_pre` null. `ops.h` already permits
this.

`ForwardLoss` calls `RunForward` with `save_for_backward = grad_enabled_`.

## 6. Classifier tiling

The classifier is the second largest term. `raw_logits_` has `rows *
padded_vocab` elements. The cross-entropy term is per row. No row depends on
another row. The evaluation path can therefore process the rows in chunks.

The chunk size is a fixed budget:

```cpp
const std::int64_t budget = 512ull << 20;
const std::int64_t chunk =
    std::max<std::int64_t>(1, budget / (padded_vocab * sizeof(ComputeType)));
```

For each chunk, the code runs `LinearForward` into `eval_logits_`. The code then
runs `ClassifierForward` into `losses_ + offset`. The host sums the chunks in
`double`. The sum order differs from the training path. The difference is small
and within the parity tolerance.

This change removes the `R * padded_vocab` term. The reference default batch
size then fits.

## 7. Invalid backward calls

A forward pass with grad mode disabled does not save activations. A later
`Backward()` call cannot run. `Backward`, `BackwardAccumulate`, and `TrainStep`
must stop with a clear message. The message mirrors the PyTorch error.

```
nanochat: Backward() called with grad mode disabled; the forward did not
save activations. Enable grad mode before the forward.
```

`ZeroGrad` stays valid. That method touches parameter gradients only.

## 8. Callers

`EvalBpb` gets one guard at the top. The rest of the function stays as is.

`train.cc` calls `EvalBpb` for the online validation metric. That call gains the
same benefit with no change.

`ScoreBatch` in [eval.md](eval.md) can use the same forward core later.

## 9. Ownership

The change spans three workstreams from [AGENTS.md](../AGENTS.md):

| Workstream | Files |
|---|---|
| Architect | `include/nanochat/model.h` |
| Workflow | `src/model.cc`, `src/ops.h` |
| Harness | `src/eval.cc`, `src/train.cc` |

The two new methods are additive to a frozen header. Ask the architect first.

## 10. Tests and gates

- T0 parity: run `EvalBpb` and a manual training forward on the same data.
  Compare the two bits-per-byte values within `1e-5`.
- T0 memory: compare `workspace_bytes()` in the two modes. The evaluation arena
  must be at least five times smaller.
- T1 GPU: run evaluation at `batch = 32` and `seq = 1024`. The run must pass
  where the training path runs out of memory.
- T2 parity: run the base evaluation gate at the reference batch size. Drop the
  `--device-batch-size 4` workaround.

## 11. Result

At `batch = 32`, `seq = 1024`, 12 layers, fp32:

| Mode | Reserved memory |
|---|---|
| Training workspace | about 36 GB |
| Grad mode, no tiling | about 7 GB |
| Grad mode with tiling | about 3 GB |

The last row fits the 11 GB card. The base evaluation gate then matches the
reference batch size.

## 12. Reproduction

The `src/eval_bench.cc` binary reproduces the last row. Build it and run it
through the entry point:

```bash
tools/nanochat build --config=cuda //src:eval_bench
tools/nanochat bench -- bazel-bin/src/eval_bench --batch 32 --seq 1024
```

The binary prints the eval arena bytes and the token rate. It uses random
tokens, so it needs no data shard. It builds as a plain `cc_binary`, so the same
source runs the CPU reference backend or the CUDA backend.

## 13. Risks

- The `raw_logits()` accessor returns stale data after an evaluation forward.
  The tests in `model_oracle_test.cc` use training mode and stay correct. Check
  each caller before the merge.
- Chunking changes the summation order. Keep the host sum in `double`.
- Two arenas use more total memory than one arena. The model holds both only
  when the caller uses both modes.
- The fp16 build needs no change. The mode and the chunk logic use no dtype.
