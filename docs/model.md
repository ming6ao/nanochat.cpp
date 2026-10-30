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
  static std::unique_ptr<Model> Create(const Config& config);
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
prefer additive changes. See [DESIGN.md §7](../DESIGN.md).

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
