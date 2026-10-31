# Model and the graphs

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
- Recompute is not implemented. The proposals are in
  [optimizations.md](optimizations.md).
- Norm statistics (`rstd`), optimizer moments, and master weights stay `float`.
- `EmbeddingBackward` uses a **persistent** dense gradient buffer and zeroes only
  the touched rows (targets the largest measured hotspot).

The save-for-backward workspace described here is what replaces an autograd
engine; see [DESIGN.md §5](../DESIGN.md).

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
prefer additive changes. See [DESIGN.md §7](../DESIGN.md). The spec selector is
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
batched cuBLAS GEMMs.

The kernels themselves are declared in [kernels.md](kernels.md).

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
  (`--train-parquet` and `--tokenizer`; [parquet-native.md](parquet-native.md)).
  No `pyarrow` and no Python at runtime.
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
