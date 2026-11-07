# The model, its graphs, and the optimizer

The model is a fixed set of five graphs over a described architecture. There is
no autograd engine and no runtime topology: `src/model.cc` holds the train
forward and backward, and `src/generate.cc` holds prefill, decode, and generate.
This page states what each graph is, how the description and the workspace work,
and how to drive the model from C++ and from the command line.

Use it through `Model::Create` for a graph, `GenerateBatch` for sampling, and
`Optimizer::Step` for training. The Python surface is [python.md](python.md).

## Fixed graphs

1. **Train step** — embed -> N blocks -> final norm -> lm_head -> softcap -> CE
   -> backward -> optimizer.
2. **Prefill** — same trunk, no loss, fills the KV cache.
3. **Decode** — one token, KV-cache append + attend.
4. **Eval** — forward-only bits-per-byte.
5. **Generate** — prefill once, clone the KV cache per sample, decode in
   lockstep, stop each row independently.

The topology lives in exactly two files: `model.cc` (train forward + backward)
and `generate.cc` (prefill + decode + generate). No other file knows the
architecture.

### No autograd engine

There is no autograd engine: no tape, no source-to-source automatic
differentiation pass, and no reverse-mode template metaprogramming.

- The graphs are fixed, so a backward pass is a fixed program that an author
  writes.
- A runtime engine would add the per-operation bookkeeping that a fixed topology
  does not need.
- A fused kernel must have its backward derived as one unit, and a
  per-operation engine cannot differentiate through the fusion.

How the tree works without an engine:

- **Save-for-backward workspace.** Each forward kernel writes exactly the
  tensors that its backward reads into a named workspace slot: `rstd` for the
  norm, the softmax statistics `(m, l)` for attention, and the pre-activation
  for the pointwise family. The slot set is static. The slots are in
  [Workspace and memory](#workspace-and-memory) and
  [../spec/block_kinds.md](../spec/block_kinds.md).
- **`Model::Backward()`** is a hand-written sequence of kernel backward calls in
  reverse order. The graph is the code.
- **A macro or a template covers a trivial pointwise forward and backward pair
  only.** Do not generalize it further.
- **Validation** is a finite-difference gradient check per kernel in
  `backends/cuda/kernels/`, plus the torch oracle in `tests/`
  ([testing.md](testing.md)).

## The model description

The architecture is data. `models/<name>/spec.json` is the description, parsed by
`include/nanochat/model_spec.h` into a `ModelSpec`, and `ModelSpecToConfig`
derives the runtime `Config` from it. `Config` therefore stays the *derived*
struct: it keeps `head_dim()`, `mlp_dim()`, `window_left()`, and `Validate()`, and
the spec never duplicates them. The one field the spec drives is
`window_pattern`, built from the ordered block list — `'S'` for a sliding-window
block, `'L'` for a full-context one — and reduced to its shortest repeating
unit, so the twelve-block list of `models/nanochat-default/spec.json` derives the
historical `"SSSL"`. The block list and the mask the graph runs cannot disagree.

`models/nanochat-default/spec.json` is the only package in the tree. Its block
list is twelve blocks, nine sliding-window and three full-context, in the order
`SSSL SSSL SSSL`; `requires` names `rope`, `sliding_window`, `value_gate`, and
`primitive_tier_complete`; and its `dimensions`, `embeddings`, `head`, `loss`,
and `policy` groups reproduce today's `Config` defaults and the softcap the
kernel defaults to.

`Model::Create(config, spec_name)` selects a package by name, and a null name
selects the built-in default, which is the spec that reproduces `Config`'s own
defaults. The name resolves through `LookupModelSpec`, the spec is validated,
and its derived `Config` must agree with the `Config` the caller passed; a
mismatch returns null with the reason. The selector reads the host filesystem
and nothing else.

### Block kinds and their saved slots

The ordered block list is the topological part of the description. A **block
kind is a named meaning, not a code path**: it says what a position computes and
nothing about which function runs there. The names, the arithmetic, the closed
list, and the sharing rules are [../spec/block_kinds.md](../spec/block_kinds.md),
which is the authority and is not restated here. What follows is the inventory
the tree actually parses (`src/model_spec.cc` holds the accepted strings) with
the slots each backward reads, because the slot set is what the workspace is
sized from.

| Field and value | In the graph | Saved slots its backward reads |
|---|---|---|
| `norm` = `rms_norm` | before attention and before the MLP | the input rows (`xr`, `x_mid`) and the per-row `rstd` (`rstd1`, `rstd2`) |
| `attention` = `causal` or `sliding_window`, with `attention_variant` = `mha` or `gqa` | the pairwise operator over the sequence | `q_final`, `k_final`, `v_final`, `attn_stats` |
| `position` = `rope` (or `none`) | position, on q and k, with the qk scale | `q_pre`, `k_pre`, plus the `cos`/`sin` tables (constants, not slots) |
| `mlp` = `relu_square` | the feed-forward transform | `pre_act`, and deliberately not `act` |
| `residual` = `blend` (the ResFormer residual) or `plain` | how a sublayer output rejoins the stream | `plain`: none. `blend`: `ve_values`, `ve_gate`, and the hidden rows `h` that fed the gate |

The mask (`attention`) and the head mapping (`attention_variant`) are fields of
the entry and not kinds: the two masks differ only in which keys a query row
sees, so the arithmetic, the head mapping, the saved slots, and the tolerance are
identical, and `mha` is `gqa` with one key/value head per query head.

Every block in `models/nanochat-default/spec.json` names `blend`, and the value
gate is the ResFormer mechanism that comes with it: `Config::has_value_embedding(layer)`
decides which layers carry a gate (alternating, and always the last), so the
gate's three slots are declared for the kind and are live only on those layers.

The slot names above are the `BlockActivations` fields in `src/ops.h`, so the
declaration and the workspace are the same list rather than two lists that must
agree. A composition reads only the slots its kinds declare, which is what lets
a fused override and a composition share one workspace
([block_kinds.md](../spec/block_kinds.md) §3, and §7.6 condition 1 of
[model-and-device-bringup.md](model-and-device-bringup.md)).

`Config` never learns which kind a block names, and the graph never learns which
device runs it: the block list says what the layers are, and the plan says how
each named piece runs ([model-and-device-bringup.md](model-and-device-bringup.md)
§9).

### One definition of semantics

The rule from [model-and-device-bringup.md §8.1](model-and-device-bringup.md):

> One definition of semantics. N implementations. One golden trace.

The composition in `src/ops.cc` defines what a kind computes. A fused family in
a backend is an optimization of that definition and must match its trace within
the committed tolerance, so a fast path is always an optimization of a known
reference and never a second definition. Two implementations of one meaning are
allowed; two meanings behind one name are not — a model that needs different
behaviour needs a new kind.

## Batched generation

`GenerateBatch` is the shared sampling primitive for evaluation samples,
generative chat evaluation, and RL rollouts ([eval.md](eval.md),
[post-training.md](post-training.md)).

```cpp
struct GenerateParams {
  int num_samples = 1;
  int max_tokens = 256;
  float temperature = 1.0f;
  int top_k = 0;
  std::uint64_t seed = 42;
  int stop_id = -1;  // e.g. <|assistant_end|>
  int bos_id = -1;   // e.g. <|bos|>
};

struct GeneratedSequence {
  std::vector<int> tokens;         // prompt + generated, no terminal token
  std::vector<std::uint8_t> mask;  // 1 sampled, 0 prompt/forced
};

void GenerateBatch(Model* model, const int* prompt, int prompt_len,
                   const GenerateParams& params,
                   std::vector<GeneratedSequence>* out);
```

Design points:

- **One prefill, many samples.** The prompt is prefilled once (batch 1); the
  resulting cache is cloned into `num_samples` rows, as the reference
  `KVCache.prefill` does. The decode graph then runs all rows in lockstep.
- **Per-row stop.** A row stops when it emits `stop_id` or `bos_id` or reaches
  `max_tokens`; other rows keep decoding. Terminal tokens are not included in
  the returned sequence.
- **Masks.** The mask records provenance, not correctness: `0` for the prompt
  and any forced token, `1` for sampled tokens. Post-training uses it to decide
  which positions receive loss.
- **Tokenizer-agnostic.** The engine never decodes text, so the terminal ids are
  passed in. The calculator tool (which needs to decode a python expression)
  therefore cannot live here; see [post-training.md](post-training.md).

## Workspace and memory

- One hand-written struct of named buffers, with sizes computed on the host from
  the config and `(batch, seq)`. One `Alloc`, bump-pointer assignment. This is
  the llm.c `ActivationTensors` pattern.
- Norm statistics (`rstd`), optimizer moments, and master weights stay `float`.
- `EmbeddingBackward` uses a **persistent** dense gradient buffer and zeroes only
  the touched rows (targets the largest measured hotspot).

The save-for-backward workspace described here replaces an autograd engine; see
[No autograd engine](#no-autograd-engine).

### Recompute (designed, not implemented)

The training arena is about 36 GB at `d12_s1024`, batch 32, fp32, and the
smallest supported card holds 16 GB. The layer activations are about 24 GB (`L * 20 * R * hidden`),
the classifier buffers about 8.6 GB, and the scratch about 3.3 GB.

`BuildTrainWorkspace` gives every layer its own `ops::BlockActivations` set, and
`RunForward` keeps all of them because `Backward()` reads them in reverse order.
One set is 20 units of `R * hidden`. A compile-time level in
`NANOCHAT_RECOMPUTE_LEVEL` gives the slot back: a recomputed slot becomes one
buffer for the whole model, `BlockForward` writes it and uses it at once, and
`BlockBackward` rebuilds it into the same buffer at the start of the layer.

| Level | Rebuild | Rebuild from | Slots removed |
|---|---|---|---|
| 0 | nothing | — | the current behaviour |
| 1 | `act` via `kReluSquare` | `pre_act` | 4 of 20, 20% |
| 2 | `h` and `h2` | `xr`, `x_mid`, `rstd1`, `rstd2` | 6 of 20, 30% |
| 3 | `xr` via `kScaleAdd`, and `ve_values` | `x_in`, `x0`, the blend scalars, the tokens | 8 of 20, 40% |

Nine slots stay saved per layer: `q_pre`, `k_pre`, `q_final`, `k_final`,
`v_final`, `attn_out`, `x_mid`, `pre_act`, and `x_out`. A rebuild of the first
five repeats a matrix product or loses data that `QkPrepBackward` overwrites in
place, and `x_out` is the input of the next layer.

At level 2 the layer term falls from about 24 GB to about 16.8 GB. The arena
still does not fit 16 GB at this batch; the win is a larger batch and a smaller
peak. No kernel is new, the frozen seam does not change, and the model already
passes each block's input to `BlockBackward`, so the API does not change either.

`tests/train_parity_test.cc` is the gate, because a rebuild reproduces the same
value from the same input, so the trajectory must not move. The
finite-difference test, the arena comparison in `src/grad_mode_test.cc`, and the
`kReluSquare` and RMSNorm kernel tests follow it.

```bash
tools/nanochat build --copt=-DNANOCHAT_RECOMPUTE_LEVEL=2
tools/nanochat test
```

## Public API

```cpp
namespace nanochat {

struct Config {
  int num_layers = 12;
  int num_heads = 6;
  int num_kv_heads = 6;
  int hidden_dim = 768;
  int seq_len = 2048;
  int vocab_size = 32768;
  int padded_vocab_size = 32768;
  float rope_base = 100000.0f;
  std::string window_pattern = "SSSL";
  bool value_embedding = true;
};

class Model {
 public:
  static std::unique_ptr<Model> Create(const Config& config,
                                       std::string* error = nullptr);
  // The spec selector: a null name selects the built-in default.
  static std::unique_ptr<Model> Create(const Config& config,
                                       const char* spec_name,
                                       std::string* error = nullptr);
  void InitWeights(uint64_t seed);

  float ForwardLoss(const int* tokens, const int* targets, int batch, int seq);
  void  Backward();
  float TrainStep(const int* tokens, const int* targets, int batch, int seq,
                  Optimizer* optimizer);

  void Save(const std::string& path) const;
  void Load(const std::string& path);
};

class KvCache;
void Prefill(Model* model, const int* tokens, int num_tokens, KvCache* kv);
int  Decode(Model* model, int token, KvCache* kv, const SampleParams& params);
void GenerateBatch(Model* model, const int* prompt, int prompt_len,
                   const GenerateParams& params,
                   std::vector<GeneratedSequence>* out);
float EvalBpb(Model* model, DataLoader* loader, int steps);

}  // namespace nanochat
```

`Config` and `Model` are frozen headers; treat them as stable by default and
prefer additive changes. See [§18 of
model-and-device-bringup.md](model-and-device-bringup.md). The spec selector is
such an addition: it is a sibling overload, so `Create(config)` and
`Create(config, &error)` keep resolving to the original declaration and every
existing call site compiles unchanged.

## Optimizer

`Optimizer` holds the exact parameter grouping of nanochat's `setup_optimizer`:

- **AdamW groups**: lm_head, token embedding, value embeddings, resid/x0/smear/
  backout scalars.
- **Muon groups**: matrix parameters, grouped by shape and stacked.

Schedules (LR multiplier, Muon momentum, weight decay) live in `Scheduler`.
`AdamWUpdate` and `MuonUpdate` are the kernels; Muon's Polar Express uses
batched cuBLAS GEMMs. A fused variant runs the whole stage as one
phase-parameterised kernel behind the same entry point and is selected for no
shape, so the batched cuBLAS path is what runs
([kernels.md](kernels.md) Part 1).

The kernels themselves are declared in [kernels.md](kernels.md).

### The Muon step

The twenty-layer shape is `PROFILES["big"]` of
`python/nanochat_cpp/pipeline.py`: 20 layers, hidden 1280, 10 heads, 5 key/value
heads, sequence 1024. The Muon groups are the ones `muon_benchmark` builds:

| Group | Batch | Elements |
|---|---:|---:|
| 5x12, the value gate | 10 | 600 |
| 640x1280 | 40 | 32,768,000 |
| 1280x1280 | 40 | 65,536,000 |
| 1280x5120 | 20 | 131,072,000 |
| 5120x1280 | 20 | 131,072,000 |
| **Total** | | **360,448,600** |

One step runs these operations, per group:

| Stage | Operations |
|---|---:|
| Momentum and the Nesterov combination | 1 |
| Frobenius norm: partial pass, finish | 2 |
| Row equilibration, one block per row | 1 |
| Equilibration divisor: partial pass, finish, scale | 3 |
| Polar Express, five iterations | 15 GEMMs and 10 elementwise passes |
| Muon+ renormalisation: partial pass, finish, scale | 3 |
| NorMuon mean squares, one per row or column | 1 |
| Second moment and the per-index step | 1 |
| NorMuon scale | 1 |
| Cautious weight decay and the parameter write | 1 |
| **Total** | **39** |

The tensor workspace is `2·total + 2·min_total` elements, and
`EnsureMuonWorkspace` in `backends/cuda/kernels/muon.cu` sizes it once per device
thread. A steady-state step allocates nothing.

### The cuBLAS seam

`kernels::Gemm` and `kernels::GemmWgrad` in `backends/cuda/gemm.cu` are the only
GEMM entry points. Both run one universal call,
`C = alpha*op(A)*op(B) + beta*C`, with `op(X) = Xᵀ` when the transpose flag is
set.

| Case | Call |
|---|---|
| Batch 1, fp32 | `cublasSgemm` |
| Batch 1, any type | `cublasGemmEx` with `CUBLAS_COMPUTE_32F` |
| Batch above 1 | `cublasLtMatmul` with a cached plan |
| No plan for the shape | `cublasGemmStridedBatchedEx` |

Every Muon product takes the third row, because a group is a batched call. The
plan is the part that matters:

- `PlanKey` holds the three extents, the batch count, three leading dimensions,
  three batch strides, two transpose flags, and the output-type flag. One plan
  serves one shape for the life of the thread.
- `LtPlan` holds one matmul descriptor, three matrix layouts, one algorithm, and
  one workspace.
- The batch rides on the layouts, through
  `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` and
  `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`.
- The workspace request is 32 MiB. The heuristic returns 8 candidates. The first
  call of a shape times all of them and keeps the fastest.
- Row-major becomes column-major by swapping the operands. The matmul's A is the
  stored B, so `CUBLASLT_MATMUL_DESC_TRANSA` receives `transpose_b`.

The operand type is `CUDA_R_16BF` when the build is bf16, else `CUDA_R_32F`. The
handle and the plan cache are per device thread, so two device threads never
share one plan.

### The cost on the H100

`tools/nanochat roofline` bounds one step of the twenty-layer shape at batch 16:

```bash
tools/nanochat roofline --devices 1 --device h100 --precision bf16 \
  --layers 20 --heads 10 --kv-heads 5 --hidden 1280 --seq 1024 \
  --vocab 32768 --window-pattern SSL --batch 16 --grad-accum 1 \
  --muon-ns-steps 5
```

| Row | bf16 | fp32 |
|---|---:|---:|
| Forward | 17.7 ms | 215.9 ms |
| Backward | 32.5 ms | 426.6 ms |
| Optimizer phase | 13.8 ms | 161.4 ms |
| — the optimizer state pass | 3.1 ms | 3.1 ms |
| — the Muon orthogonalisation | 10.7 ms | 158.3 ms |
| Step bound | 64.1 ms | 804.0 ms |

The optimizer phase is 21.6 percent of the bf16 step bound, and the
orthogonalisation alone is 16.7 percent. The two optimizer rows do not move with
the sequence length, because the Muon arithmetic reads the parameter stack
alone.

One measured run of `--profile big --muon` on one H100 SXM card at bf16,
revision `e47f71c`:

| Row | Per step |
|---|---:|
| `polar:big:step`, the five Polar Express products | 15.18 ms |
| `muon:big:step`, ns=5, the shipped stage | 29.09 ms |
| `muon:big:step`, ns=1 | 12.34 ms |
| `muon:big:step`, ns=0, every operation but the iterations | 8.12 ms |

The products run at 70.6 percent of the bf16 peak and the stage at 36.8 percent.
The non-product part of the stage is 13.91 ms. The `ns=0` control says where it
comes from: 8.12 ms is the fixed part of the update, and each Polar Express
iteration adds 4.19 ms, of which 3.04 ms is its products and 1.16 ms is the
elementwise traffic and the launches beside them.

### The decisions that fix the step

| Decision | Rule | Reason |
|---|---|---|
| D2 | Keep `kernels::Gemm`, on every device and in every build | The seam already reaches a tuned batched plan, and the measured hand GEMM reached 0.68 to 0.72 of the cuBLAS rate ([kernels.md](kernels.md) Part 1) |
| D3 | Keep `kMuonFusedMaxElements` at 0 | The fused schedule is fp32 and loads `float4` quads, so it cannot run on the bf16 leg |
| D4 | Remove the launch gaps with one CUDA Graph per `(group, ns_steps)` | It keeps cuBLAS and removes 0.70 ms of launch cost with no arithmetic change |
| D5 | Merge stages inside a family, never across a whole-matrix statistic | One block size for sixteen phases costs 6 to 10 percent |
| D6 | Move bytes, not arithmetic: every merge stays bit exact | The seam tolerance is 1e-5 against `testing/optim_ref.h` |
| D9 | fp32 master and fp32 momentum, with `ComputeType` parameters | A merge never changes a stored value |

The graph of D4 is capturable because no host read sits inside the step: the
first call of a shape builds the plan, queries its 8 candidates, times them, and
allocates the workspace, so `MuonUpdate` warms the group once and then captures.
A later caller that allocates inside a captured region fails the capture, and
the fallback is the multi-launch path with one log line.

```bash
tools/nanochat bench -- ./bazel-bin/src/nanochat_cli train --profile big --muon
```

The `muon:` rows carry `ns=` in the shape column and `polar:` carries the
products alone, so the elementwise share is one subtraction
([performance.md](performance.md) section 2.1).

### Optimizer state

Every parameter carries a fp32 master, a `ComputeType` compute copy, and one
fp32 gradient. In the fp32 build the master aliases the compute copy; in a bf16
build they are two buffers.

| Item | Bytes per parameter |
|---|---|
| master and compute copy | 4 (fp32 build), 6 (bf16 build) |
| parameter gradient | 4 |
| AdamW first and second moment | 8, AdamW groups only |
| Muon momentum and factored second moment | 4, Muon groups only |

Gradient clipping is global over every parameter, but it no longer needs a copy
of every gradient. `Step` reduces the per-parameter gradients in one streaming
pass (`kernels::GradNorm`) and folds the clip scale into each update through
`AdamWParams::grad_scale` and `MuonParams::grad_scale`, so the decoupled weight
decay stays unscaled. The Muon groups still stack, because the kernel's operand
is `[num_params, rows, cols]`: one reusable buffer sized for the widest group is
refilled immediately before that group's update.

At the default `d12` shape (84.9 M Muon parameters, 201.3 M AdamW parameters)
the block is about 4.1 GiB in the fp32 build and 4.6 GiB in the bf16 build,
about 1 GiB below the former design, which staged a second fp32 copy of every
gradient in one flat buffer.

## Data, checkpoints, and the oracle

- **Data**: parquet documents, tokenized during the run by the native reader
  (`--train-parquet` and `--tokenizer`; [data.md](data.md)). No `pyarrow` and no
  Python at runtime.
- **Checkpoint**: a self-describing tensor container (name, storage dtype,
  shape, attributes) with one run metadata block (stage, step, world size,
  precision, config, tokenizer id, revision, notes). Version 1; the reader also
  accepts a version 0 file, which has no metadata block. Not a framework
  `state_dict`. The project writes only this container;
  `Checkpointer::LoadModel` still reads a legacy `NCMDL001` file.
- **Oracle**: `debug_state.bin`, produced offline by nanochat's `gpt.py`.
  `tests/oracle_test.cc` compares logits, loss, and gradients, then runs a few
  optimizer steps and matches losses. Per-backend tolerance: fp32 CUDA ~1e-5).

The oracle fixture is consumed by the test harness described in
[testing.md](testing.md).
