# Notebook lifecycle and checkpoint container design

Status: Part A is implemented. Part B is not. The container is version 0 in the
tree, with the fp32, bf16, and reserved int8 codes and the attribute block. The
C++ writer stores the true tensor shape. The byte layout lives only in
`src/data.cc`, and a host tool reaches it through the `nanochat_checkpoint_*` C
surface. The parameter-only read is in the tree: the flag on
`Checkpoint::Load`, the `Checkpointer::LoadModel` choice, the
`nanochat_runtime_load_weights` C entry point, and the Python
`MeshRuntime.load(path, optimizer=)` argument. Part B is not in the tree.

The document has two parts. Part A specifies the checkpoint container. Part B
specifies the three notebooks that use it. The container format is the shared
contract, so Part A comes first.

Owners: the Notebooks workstream owns Part B and this file. Part A touches the
training harness and adds one C entry point. Promote Part A to
`docs/checkpoint-format.md` after architect review.

## 1. Goals

- Split the single two-T4 notebook into three notebooks: pretraining,
  evaluation, and post-training.
- Give each notebook the full session setup, so any one notebook runs alone.
- Use one checkpoint container for every stage.
- Let a reader skip the optimizer state when it does not need it.
- Store every checkpoint on the Hugging Face Hub.
- Publish the container schema in one place.

## 2. Decisions

| Topic | Decision |
|---|---|
| File set | `setup.ipynb`, `pretrain.ipynb`, `eval.ipynb`, `posttrain.ipynb` |
| Name prefix | No `nanochat-cpp-` prefix on any notebook |
| Container | One format. Magic `NCHKPT01`. Version 0. |
| Storage | fp32 for the canonical artifact. bf16 for a compact artifact. |
| File suffix | `.ckpt` |
| Writer | The notebooks save through the training runtime |
| Reader | A parameter-only mode reads the parameter section and stops |
| Repository | One repository for the model family. One folder per stage. |
| Namespace | Derived from `HfApi.whoami()`, with an `HF_USERNAME` override |
| Visibility | Public by default |
| Schema | `notebooks/checkpoint.md` part A. The layout lives in `src/data.cc`. |
| Fine-tuning | The post-training notebook runs a one-device `MeshRuntime` loop |

The `.ckpt` suffix names the `NCHKPT01` container. The suffix is cosmetic:
the reader detects the container by the eight magic bytes. `nc.Model.save`
writes a parameter-only `NCHKPT01` file, so every writer uses one format. See
section 4.

---

# Part A: the checkpoint container

## 3. File naming

The canonical name is `checkpoint.ckpt`. The magic is independent of the file
name. A reader detects the container by the eight magic bytes, not by the
suffix.

`NCHKPT01` is the only container the project writes. Reads still accept the
legacy `NCMDL001` container, and no writer produces it.

## 4. Writers

The notebooks save through the training runtime. The runtime writes
`NCHKPT01`.

| Writer | Container | Optimizer section | Step record |
|---|---|---|---|
| `MeshRuntime.save` (`nc.MeshRuntime.save`) | `NCHKPT01` | AdamW, Muon, ANVIL | yes |
| `Checkpointer::SaveModel(model, optimizer, step, path)` | `NCHKPT01` | per the optimizer | yes |
| `Checkpointer::SaveModel(model, path)` | `NCHKPT01` | empty count | no |
| `nc.Model.save` (`nanochat_save`) | `NCHKPT01` | empty count | no |

`nc.Model.save` writes the parameters only. The project no longer writes
`NCMDL001`; `Checkpointer::LoadModel` still reads a legacy file. The pretraining
and post-training notebooks call `MeshRuntime.save`. The post-training loop uses
a one-device `MeshRuntime`, so a save has the same code path as pretraining.

## 5. Byte layout

The container is a flat little-endian byte stream. It uses no pickle, no
language runtime, and no object graph. The writer is `Checkpoint::Save` in
`src/data.cc`.

### 5.1 Header and sections

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 8 | magic | the ASCII bytes `NCHKPT01` |
| 8 | 4 | version | `uint32`, the value 0 |
| 12 | 4 | parameter count | `uint32` |
| 16 | variable | the parameter records | see section 6 |
| after the parameters | 4 | optimizer count | `uint32` |
| after the count | variable | the optimizer records | see section 6 |

The parameter section is always first. The optimizer section is optional. A
parameter-only file writes an optimizer count of zero.

### 5.2 Integer encoding

Every integer is little-endian. The helper functions `WriteU16`, `WriteU32`,
`WriteU64`, and `WriteI64` in `src/data.cc` serialize the values by hand. A
`uint32` value is four bytes, least significant byte first. An `int64` value is
the two's-complement form of a `uint64`.

## 6. Record layout

Every parameter record and every optimizer record uses the same layout.

| Size | Field | Notes |
|---|---|---|
| `uint16` | name length | the byte count, clipped to 65535 |
| variable | name | raw bytes, not null terminated |
| `uint32` | storage dtype code | see section 7 |
| `uint32` | ndim | the tensor rank, at most 8 |
| `int64 * ndim` | dimensions | the full, unsharded shape |
| `uint32` | attribute count | zero or more |
| variable | the attribute records | see section 6.1 |
| `uint64` | payload byte count | |
| variable | payload | little-endian elements |

The name is a length-prefixed byte string. It is not a C string. The reader
rejects an ndim above 8, a negative dimension, an unknown dtype code, and a
truncated payload.

`ndim` is the tensor rank, not a data-parallel rank. The dimensions are the
full shape. FSDP and tensor parallel are out of scope, so the record carries no
shard metadata. The C++ writer stores the true shape: `[rows, cols]` for a
matrix and `[count]` for a vector.

`TensorRecord` in `include/nanochat/data.h` mirrors the record in memory. The
`numel()` method multiplies the dimensions.

### 6.1 Attribute records

An attribute record is non-recursive. It has no attribute count.

| Size | Field | Notes |
|---|---|---|
| `uint16` | attribute name length | the byte count, clipped to 65535 |
| variable | attribute name | raw bytes, not null terminated |
| `uint32` | storage dtype code | see section 7 |
| `uint32` | ndim | the tensor rank, at most 8 |
| `int64 * ndim` | dimensions | each dimension is zero or more |
| `uint64` | payload byte count | |
| variable | payload | little-endian elements |

An fp32 or bf16 record carries an attribute count of zero. An int8 record
carries one attribute, `quant.scale`. A later feature adds an attribute with
no layout change.

## 7. Dtype codes

The record dtype is a storage dtype. It is separate from the build compute
dtype. The writer stores the chosen storage dtype. The reader converts the
stored value to the build compute dtype.

| Code | Name | Element size | State |
|---|---|---|---|
| 0 | fp32 | 4 bytes | in use |
| 1 | bf16 | 2 bytes | in use |
| 2 | int8 | 1 byte | reserved |

There is no fp16 code. The fp16 compute build stays for the simulator. That
build stores fp32 or bf16.

The C values are `DType::kFp32`, `DType::kBf16`, and `DType::kInt8` in
`include/nanochat/tensor.h`. A host tool enumerates them through
`nanochat_checkpoint_dtype`, so no Python copy of the codes exists.

### 7.1 Storage dtype policy

Use fp32 for the canonical artifact. An fp32 container is exact and portable.
Use bf16 for a compact artifact. A bf16 container halves the payload.

A bf16 parameter is not exact. The fp32 `master` record in the optimizer
section keeps the exact weights. Therefore an fp32 container is the artifact
for a resume.

An int8 container stores a scale in the `quant.scale` attribute. The scale is
one value for each tensor or one value for each channel. int8 is storage only.
int8 compute is out of scope.

The optimizer step rides in one fp32 record. Its four payload bytes are the
raw little-endian `int32` step. See section 8.2.

## 8. Record names

### 8.1 Parameter records

The parameter names are the names the model registers. The source of truth is
`TrainModel` in `src/model.cc` and the canonical patterns in
`python/nanochat_cpp/checkpoint.py`.

| Name | Notes |
|---|---|
| `transformer.wte.weight` | token embedding |
| `lm_head.weight` | output projection |
| `resid_lambdas` | per-layer residual scale |
| `x0_lambdas` | per-layer embedding-mix scale |
| `smear_gate.weight`, `smear_lambda`, `backout_lambda` | scalar gates |
| `value_embeds.<layer>.weight` | value embedding, when enabled |
| `transformer.h.<layer>.attn.c_q.weight` | query projection |
| `transformer.h.<layer>.attn.c_k.weight` | key projection |
| `transformer.h.<layer>.attn.c_v.weight` | value projection |
| `transformer.h.<layer>.attn.c_proj.weight` | attention output |
| `transformer.h.<layer>.attn.ve_gate.weight` | value-embedding gate |
| `transformer.h.<layer>.mlp.c_fc.weight` | feed-forward input |
| `transformer.h.<layer>.mlp.c_proj.weight` | feed-forward output |

`<layer>` is a zero-based integer. A load matches a record by its name. The
record order does not matter to a reader. The writer uses the model parameter
order.

### 8.2 Optimizer records

The optimizer records live in the separate optimizer list. The names do not
collide with a parameter record.

| Name | Content |
|---|---|
| `master` | the concatenated fp32 master weights |
| `adamw.<kind>.m` | AdamW first moment |
| `adamw.<kind>.v` | AdamW second moment |
| `muon.<rows>x<cols>.buf1` | Muon momentum buffer |
| `muon.<rows>x<cols>.buf2` | Muon second-moment buffer |
| `anvil.<rows>x<cols>.velocity` | ANVIL twin-rail velocity |
| `anvil.<rows>x<cols>.lane_energy` | ANVIL lane energy |
| `optimizer/step` | the 1-based step, four raw little-endian `int32` bytes |

`<kind>` is one of `lm_head`, `embedding`, `value_embedding`, `resid`, `x0`,
or `smear`. The list is `ADAMW_KINDS` in `python/nanochat_cpp/checkpoint.py`
and `KindName` in `src/optim.cc`.

The step record uses dtype code 0 and shape `[1]`, but its four payload bytes
are an integer, not a float. The reader must not read the step as a float.

## 9. Reader modes

The container has two read modes.

| Mode | Reads | Used by |
|---|---|---|
| Parameter only | the header and the parameter records | `nc.Model.load`, `MeshRuntime.load(optimizer=False)` |
| Full | the header, the parameter records, and the optimizer section | A resume through `MeshRuntime.load` |

`Checkpoint::Load` gains one flag:

```cpp
// include/nanochat/data.h
bool Load(const std::string& path, std::string* error = nullptr,
          bool with_optimizer_state = true);
```

When the flag is false, the reader parses the header and the parameter
records, then returns. It does not read, parse, or allocate the optimizer
records. The optimizer section is last, so a stop is cheap.

`Checkpointer::LoadModel` chooses the flag from its arguments:

```cpp
const bool with_optimizer_state = (optimizer != nullptr) || (step != nullptr);
```

A caller that passes no optimizer gets the parameter-only read. Therefore
`nc.Model.load` skips the optimizer state by construction. A step-only caller
also reads the optimizer section, so the step record stays reachable
(`src/optimizer_state_test.cc`).

### 9.1 The runtime load

`MeshRuntime::Load` always passes the optimizer, because a resumed pretraining
run needs the moments. A fine-tuning warm start must not inherit the
pretraining moments. The supervised fine-tuning reference starts from fresh
optimizer state.

The Python runtime gains one argument:

```python
def load(self, path, optimizer=True) -> None: ...
```

A new C entry point backs it:

```c
// include/nanochat/capi.h
void nanochat_runtime_load_weights(nanochat_runtime* runtime,
                                   const char* path);
```

The function loads the parameters into every replica and leaves the optimizer
state and the step unchanged. The optimizer section is not read. The Python
`MeshRuntime.load(path, optimizer=False)` calls it. The default keeps the full
read.

The change is additive. Bump `kNanochatVersion` in `bindings/nanochat_capi.cc`;
the current value is `0.12.0`.

## 10. Versioning

The version field is 0. A reader accepts the value 0 and rejects any other
value. No earlier published container exists, so version 0 needs no
compatibility branch.

A new optimizer buffer becomes one more record, so that addition needs no
version change. A new record attribute needs no version change, because the
attribute block is already present. A new dtype code needs a reader branch. A
change to the header or to the record layout needs a version change.

Old `NCMDL001` files stay readable. `Checkpointer::LoadModel` routes on the
magic: `IsCheckpointContainer` sends a version 0 container to `Checkpoint`, and
any other file to the legacy reader.

## 11. The schema in machine-readable form

The bytes are the contract. The decoded view is one object:

```json
{
  "magic": "NCHKPT01",
  "version": 0,
  "parameters": [
    {"name": "transformer.wte.weight", "dtype": 1, "shape": [32768, 1024],
     "elements": 33554432, "bytes": 67108864, "attributes": []}
  ],
  "optimizer_state": [
    {"name": "optimizer/step", "dtype": 0, "shape": [1], "bytes": 4,
     "attributes": []}
  ]
}
```

The block is illustrative. It shows one bf16 parameter record. Under fp32 the
dtype is 0 and the byte count doubles.

The constants are the C values. The C surface reports them:
`nanochat_checkpoint_dtype` enumerates each dtype with its name, code, and
element size.

| Constant | Source |
|---|---|
| `MAGIC` | `kCheckpointMagic` in `src/data.cc` |
| `VERSION` | the `Checkpoint::Save` header, 0 |
| `DTYPE_FP32`, `DTYPE_BF16`, `DTYPE_INT8` | `DType::kFp32`, `kBf16`, `kInt8` |
| `MAX_NDIM` | the `ReadRecord` limit, 8 |

The C source of truth is `src/data.cc` plus `include/nanochat/data.h`. The C
surface exposes it, so a host tool does not repeat it (`docs/python.md`
section 7).

## 12. Size

The parameter section holds the storage copy of the weights. The storage dtype
sets the size. For a 252 million parameter model, an fp32 parameter section is
about 1.0 GiB, and a bf16 parameter section is about 0.5 GiB. The optimizer
section holds the fp32 master and the moments. That section is larger, because
AdamW keeps two moments and Muon keeps two buffers.

A parameter-only read avoids the optimizer section. That is the reason for the
flag in section 9.

---

## 13. Distributed checkpoints

The container and the runtime support replicated data parallel only. They do
not support a sharded checkpoint. They do not support Fully Sharded Data
Parallel (FSDP).

### 13.1 The current placement policy

The runtime keeps one full model replica per device. It shards only the batch.
`docs/client-runtime-design.md` section 5.4 states the policy:

| Object | Placement |
|---|---|
| Model parameters | replicated, one full copy per device |
| Optimizer state | replicated, one full state per device |
| Global batch | sharded on the `data` axis |

`validate_parameter_sharding` in `python/nanochat_cpp/mesh.py` rejects a
sharded parameter. The runtime requires a full replica per device.
Tensor parallel and pipeline parallel are out of scope by decision.

A save writes one canonical file. `MeshRuntime::Save` in
`src/mesh_runtime.cc` takes `replicas_[0]` and calls
`Checkpointer::SaveModel`. A load copies the same file into every replica.

The multi-process path is the same idea. `DistributedPlan.checkpoint_path`
returns one path, `checkpoint.ckpt`. Only rank 0 writes it. The other ranks
write nothing, because each rank holds a full replica. The file is canonical,
not a sharded checkpoint.

### 13.2 What the container lacks

An `NCHKPT01` record holds a name, a storage dtype, a shape, a payload, and an
attribute block. The block holds a quantization scale only. It has no shard
metadata. It has no mesh shape. It has no per-tensor placement. It has no
global against local shape. It has no shard index. It has no manifest that
names the shard files.

The Python converter rejects a sharded input. In
`python/nanochat_cpp/checkpoint.py`, `_muon_records` and `_anvil_records` raise
"this is a sharded rank shard, not a single-rank shard". The check runs when
the reference optimizer group stacks a different number of buffers than the
group holds.

### 13.3 What FSDP would need

1. A shard attribute for each tensor: the global shape, the local shape, the
   shard index, and the mesh axis.
2. Optimizer-state shard metadata. FSDP shards the moments. Each rank then
   holds a slice.
3. A coordinated save. Each rank writes its own shard. A manifest names the
   shards.
4. A coordinated load. Each rank reads its own shard.
5. A reshard step. A load rebuilds the full tensor or a different shard
   layout. The load device count can differ from the save device count.
6. A gather path. Under FSDP, no rank holds the full model. A single-file
   publish must gather first.

### 13.4 Why the format can accept the change

The record is self-describing. A shard attribute is an additive change,
because the attribute block is already present. The version field gates a
layout change. A load that matches by name already skips a record it does not
find, so a partial record set stays loadable.

### 13.5 Effect on the notebooks

Part B publishes one canonical checkpoint from replica 0. That is correct
today, because every device holds a full parameter copy.

The publish step changes when the project adds a sharded mode:

- Tensor parallel: rank 0 holds a partial model. The publish must gather the
  full parameters first.
- FSDP: no rank holds the full model. The publish must store every shard plus a
  manifest, or gather into a full checkpoint before the upload.

The evaluation and post-training notebooks also change. A load under a sharded
mode must reshard to the target layout.

# Part B: the three notebooks

## 14. File set

| File | Status | Purpose |
|---|---|---|
| `notebooks/setup.ipynb` | rename from `nanochat-cpp-setup.ipynb` | Manual session check and the `pi` agent |
| `notebooks/pretrain.ipynb` | new | Data, tokenizer, model, pretraining, publish the base |
| `notebooks/eval.ipynb` | new | Load a checkpoint, measure it, score samples and chat |
| `notebooks/posttrain.ipynb` | new | Load the base, fine-tune, publish the chat model |
| `notebooks/nanochat-cpp-on-2x-t4.ipynb` | remove | Replaced by the three notebooks |
| `notebooks/session.py` | new | The shared setup block |
| `notebooks/hub_checkpoint.py` | new | The shared Hugging Face Hub helper |
| `notebooks/session_test.py` | new | Tests for the setup helpers |
| `notebooks/hub_checkpoint_test.py` | new | Tests for the Hub helper, with a fake API |
| `notebooks/README.md` | new | Run order and the checkpoint flow |
| `notebooks/BUILD.bazel` | edit | Add the two libraries and the two tests |
| `notebooks/args.py`, `args_test.py` | keep | The model arguments |
| `notebooks/mesh_train.py`, `mesh_train_test.py` | keep | The pretraining launcher |
| `notebooks/mesh_worker.py` | keep | The training worker |
| `notebooks/pi_agent.py`, `pi_agent_test.py` | keep | The setup notebook agent |
| `notebooks/kaggle_setup.py`, `kaggle_setup_test.py` | keep | The Kaggle session helpers |
| `notebooks/kaggle_demo.py` | keep | The demo entry point |

## 15. Shared setup

Every notebook runs the setup first. A reader can run any one notebook alone.
The body lives in `notebooks/session.py`, so the code has one home.

```python
def load_secrets(names=("HF_TOKEN",)) -> dict[str, str]: ...
def clone_and_build(repo_url, repo_path, ref=None, arch="sm_75") -> Host: ...
def require_devices(count=2) -> list: ...
def setup(repo_url, repo_path, *, arch="sm_75", devices=2) -> Session: ...
```

Each notebook runs two cells:

```python
import os, sys
from pathlib import Path
REPO_URL = "https://github.com/ming6ao/nanochat.cpp"
REPO = Path("/kaggle/working/nanochat.cpp")
BASE = Path("/kaggle/working/nanochat-cache")
import session
sess = session.setup(REPO_URL, REPO, arch="sm_75", devices=2)
```

```python
import nanochat_cpp as nc
for device in nc.devices():
    print(device, device.name,
          f"{device.total_memory_bytes / 2**30:.1f} GiB")
```

`session.py` wraps `notebooks/kaggle_setup.py`, which keeps the repository
fetch, the bootstrap, and the build. The new `setup` adds the secrets load and
the device check.

`session.py` imports `kaggle_secrets` and `huggingface_hub` inside the
functions. The tests then run without either package.

## 16. Shared Hub helper

`notebooks/hub_checkpoint.py` holds every Hugging Face Hub call.

```python
def repo_id_for(run_name, namespace=None, prefix="nanochat"): ...
def stage_dir(root, stage): ...                    # <root>/<stage>
def build_run_record(...) -> dict: ...             # pure metadata assembly
def push_stage(repo_id, stage, folder, *, token=None, private=False,
               commit_message="", tag=None): ...
def pull_stage(repo_id, stage, dest, *, revision="main", token=None,
               patterns=STAGE_PATTERNS): ...
def load_stage(repo_id, stage, dest, *, revision="main", token=None,
               device="cuda"): ...                 # model, tokenizer, args
```

`load_stage` calls `nc.Model.load`, which uses the parameter-only read.

### 16.1 Upload

```python
from huggingface_hub import HfApi

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model",
                private=private, exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="model",
                  folder_path=str(folder), path_in_repo=stage,
                  commit_message=commit_message, token=token)
if tag:
    api.create_tag(repo_id=repo_id, tag=tag, repo_type="model", token=token)
```

### 16.2 Download

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id=repo_id, repo_type="model", revision=revision,
    local_dir=str(dest), token=token,
    allow_patterns=[f"{stage}/{p}" for p in patterns])
folder = Path(local) / stage
```

### 16.3 Load

The evaluation notebook loads into an `nc.Model`:

```python
model_args = ModelArgs.from_json((folder / "model.json").read_text())
model = nc.Model(model_args.to_config(), device=device, seed=0)
model.load(str(folder / "checkpoint.ckpt"))   # parameters only
tokenizer = nc.Tokenizer.load(str(folder / "tokenizer.nctoken"))
```

The post-training notebook loads into a one-device runtime:

```python
runtime = nc.MeshRuntime(nc.Mesh((nc.devices()[0],)),
                         model_args.to_config(), optimizer_config,
                         batch=1, grad_accum=1, seq=SFT_SEQ, seed=42)
runtime.load(str(folder / "checkpoint.ckpt"), optimizer=False)
```

`optimizer=False` skips the pretraining moments. The fine-tuning warm start
then uses fresh optimizer state.

## 17. Repository model

One repository holds the model family.

- `repo_id` is `<namespace>/nanochat-<run_name>`. The run name follows the
  shape string, for example `base-L16-H1024-n8-kv4-s4096`.
- The namespace comes from `HfApi.whoami()["name"]`. The environment variable
  `HF_USERNAME` overrides it.
- The repository is public by default. A private run passes `private=True`.
- Upload to `main`. Create one tag per run, for example `base-500`. A tag gives
  the evaluation notebook a stable revision.

### 17.1 Folder layout

```
<repo_id>/
  base/
    checkpoint.ckpt        # NCHKPT01: parameters, optimizer state, and step
    model.json             # ModelArgs
    train.json             # the mesh job or the TrainArgs
    run.json               # lineage and provenance
    tokenizer.nctoken
    metrics.jsonl
    summary.json
    README.md              # the model card
  sft/
    (the same file names)
```

### 17.2 Metadata files

`model.json` holds `args.ModelArgs.to_json()` output. The container does not
store the shape, so this file is mandatory.

`train.json` holds the training knobs. For pretraining it holds the
`MeshTrainPlan.job()` mapping. For post-training it holds
`args.TrainArgs.to_json()` output.

`run.json` holds the provenance record:

```json
{
  "stage": "base",
  "run_name": "base-L16-H1024-n8-kv4-s4096",
  "git_revision": "b59ca93",
  "build_key": "...",
  "cuda_arch": "sm_75",
  "container": "NCHKPT01",
  "parent_repo_id": "",
  "parent_revision": "",
  "step": 500,
  "created_utc": "..."
}
```

The `sft` record names the base repository and the base tag.

## 18. Notebook: `pretrain.ipynb`

| Cell | Content |
|---|---|
| 1 | Markdown: the shape, the fixed cost, the Hub output |
| 2 | Setup: secrets, clone, bootstrap, build, device check |
| 3 | Download the two ClimbMix shards |
| 4 | Pull the tokenizer from the Hub; else train it and push it |
| 5 | `args.ModelArgs` and the `model_fit` report |
| 6 | Build the `MeshTrainPlan`, then `mesh_train.launch` and stream |
| 7 | Write `model.json`, `train.json`, `run.json`, and the model card |
| 8 | `hub_checkpoint.push_stage` for the base folder, then tag the revision |
| 9 | Short sanity check: a few validation steps and two samples |
| 10 | Markdown: notes and the next notebook |

`MeshTrainPlan.checkpoint_path()` returns `checkpoint.ckpt` after the rename in
section 24.

## 19. Notebook: `eval.ipynb`

| Cell | Content |
|---|---|
| 1 | Markdown: evaluation is forward only, and it writes no checkpoint |
| 2 | Setup |
| 3 | `REPO_ID`, `STAGE`, `REVISION`; `hub_checkpoint.load_stage` |
| 4 | The train and validation bits-per-byte loop |
| 5 | The sample prompts and the greedy completions |
| 6 | `nc.chat.ChatEvaluator` on the reference tasks |
| 7 | Write one JSON report and push it beside the stage folder |
| 8 | Markdown: notes |

`STAGE` is `base` or `sft`. The load reads the parameter section only. The Hub
download still fetches the whole `checkpoint.ckpt`, but the reader does not
parse the optimizer section.

## 20. Notebook: `posttrain.ipynb`

| Cell | Content |
|---|---|
| 1 | Markdown: supervised fine-tuning of a base checkpoint |
| 2 | Setup |
| 3 | `hub_checkpoint.pull_stage` for the `base` stage |
| 4 | Build the one-device `MeshRuntime`; `runtime.load(..., optimizer=False)` |
| 5 | The base completion before fine-tuning |
| 6 | The conversation mixture and the packing |
| 7 | The fine-tuning loop on the runtime |
| 8 | `runtime.save("checkpoint.ckpt")`, then the metadata files |
| 9 | `hub_checkpoint.push_stage` for the `sft` stage, with the base revision |
| 10 | A small chat check |
| 11 | Markdown: notes |

The one-device `MeshRuntime` uses `world_size=1`, `grad_accum=1`, `batch=1`,
and `seq=SFT_SEQ`. The base load passes `optimizer=False`, so the reader skips
the pretraining optimizer section. The masked targets use `-1` for ignored
positions.

`ForwardLoss` and `BackwardAccumulate` already ignore `-1`, so the runtime
handles the supervised loss. Add one test that drives a one-device runtime with
a `-1` target and checks that the loss ignores it.

The save writes the parameters plus the SFT optimizer state, so the SFT run can
resume. The base load reads the parameter section only. The full chat
evaluation stays in `eval.ipynb` with `STAGE = "sft"`.

This path is a new decision. `docs/post-training.md` section 8 names the
`nc.sft.run` bridge and `sft_main` as the SFT job. This design replaces that
entry for the notebook with the in-process runtime loop. `sft_main` stays for a
one-shot scripted run. Update `docs/post-training.md` section 8 to match.

## 21. Failure handling

- A missing `HF_TOKEN` stops an upload with the corrective message. It does not
  stop training.
- A failed download falls back to `/kaggle/working/<run_name>` when it exists.
- A missing `model.json` or `tokenizer.nctoken` stops the load and names the
  file.
- `create_repo` can fail when the token lacks write access. Catch the error,
  report the repository name, and keep the local copy.
- A private repository needs the token for the download too.
- Never print the token.

## 22. Tests

`notebooks/BUILD.bazel` adds:

```
py_library(name = "session", srcs = ["session.py"], ...)
py_library(name = "hub_checkpoint", srcs = ["hub_checkpoint.py"], ...)
py_test(name = "session_test", srcs = ["session_test.py"], deps = [":session"])
py_test(name = "hub_checkpoint_test", srcs = ["hub_checkpoint_test.py"],
        deps = [":hub_checkpoint", ":args"])
```

Both tests join the existing `test_suite(name = "all", ...)`.

The Hub test covers the repository name and the stage path. It covers the
metadata assembly for a base run and a fine-tuned run. It checks the upload
call sequence on a fake API, the download pattern list, and the missing-token
error.

The harness adds one container test in `src/optimizer_state_test.cc`, next to
`TestParameterOnlyStaysLoadable`. It writes an `NCHKPT01` file with an
optimizer section, then reads it in parameter-only mode and in full mode. It
checks that both modes return the same parameters. It appends a truncated
optimizer section and checks that a parameter-only read tolerates it. A second
check drives `nanochat_runtime_load_weights` on a one-device runtime and
verifies that the optimizer state and the step stay unchanged.

Both notebook tests run on the CPU with no network.

The version 0 container, the storage dtype codes, the attribute block, and the
bf16 load conversion already have coverage. `src/data_test.cc` round-trips the
container and checks the bf16 conversion. `src/harness_test.cc` stores a model
as bf16 and loads it back. `bindings/nanochat_capi_test.cc` round-trips the C
checkpoint surface. The Python container test loads the shared library, so it
pins no layout constant of its own.

## 23. Definition of Done

- Three notebooks exist. Each one runs end to end on a two-T4 Kaggle session.
- Each notebook performs the setup itself.
- The notebooks write one container: `NCHKPT01` with the `.ckpt` suffix.
- `nc.Model.load` and `MeshRuntime.load(optimizer=False)` skip the optimizer
  section.
- The pretraining notebook publishes the base stage. The eval and
  post-training notebooks load it in a later session with no retraining.
- The post-training notebook publishes the fine-tuned stage.
- `tools/nanochat test` passes for the notebook tests and the container test.
- No notebook prints a token.

## 24. Sequencing

1. The version 0 container, the storage dtype codes, the attribute block, and
   the `checkpoint.ckpt` rename are in the tree. Each change to the C surface
   bumps `kNanochatVersion`; the current value is `0.12.0`.
2. The parameter-only read is in the tree, with its tests. It adds the flag on
   `Checkpoint::Load`, the `Checkpointer::LoadModel` choice,
   `nanochat_runtime_load_weights`, and the Python `optimizer` argument.
3. Write `session.py`, `hub_checkpoint.py`, and their tests.
4. Create `pretrain.ipynb`. Verify one full session, including the upload and
   the tag.
5. Create `eval.ipynb`. Verify a fresh session that downloads the tag.
6. Create `posttrain.ipynb`. Verify the base download, the fine-tune, and the
   `sft` upload.
7. Remove `nanochat-cpp-on-2x-t4.ipynb`. Rename the setup notebook. Add
   `README.md`.
8. Ask the architect to promote part A to `docs/checkpoint-format.md`.

## 25. Open items

1. The one-device runtime path for supervised fine-tuning needs the masked-loss
   test in section 20. If it fails, keep `nc.Model` plus `nc.Optimizer` for the
   loop and add a parameter-only `NCHKPT01` save entry point for the model. The
   SFT checkpoint then cannot resume from its own optimizer state.
2. The `docs/post-training.md` section 8 revision needs architect sign-off.
3. The model card template needs a final list of fields.
4. The tokenizer repository stays a separate repository. Confirm the name.
