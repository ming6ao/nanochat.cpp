# Checkpoint format and notebook lifecycle

Status: Parts A and B are implemented. Every writer writes version 1, which
adds the run metadata block between the header and the parameter count
(section 5.3), and the reader accepts version 0 and version 1. The fp32, bf16,
and reserved int8 codes and the attribute block are unchanged. The C++ writer
stores the true tensor shape. The record byte layout lives only in `src/data.cc`,
the metadata block in `src/fields.cc` and `src/checkpoint_metadata.cc`, and a
host tool reaches both through the `nanochat_checkpoint_*` C surface. The
parameter-only read is in the tree: the flag on
`Checkpoint::Load`, the `Checkpointer::LoadModel` choice, the
`nanochat_runtime_load_weights` C entry point, and the Python
`MeshRuntime.load(path, optimizer=)` argument. Part B is in the tree: the two
shared modules, the three notebooks, and the one-device runtime masked-loss
test.

The document has two parts. Part A specifies the checkpoint container. Part B
specifies the three notebooks that use it. The container format is the shared
contract, so Part A comes first.

Owners: the architect owns this file. The Notebooks workstream owns Part B.
Part A touches the training harness and adds one C entry point.

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
| Container | One format. Magic `NCHKPT01`. Writers write version 1; readers accept version 0 and version 1. |
| Run metadata | One visitor block between the header and the parameter count (section 5.3), with the drift rules of section 5.4 |
| Storage | fp32 for the canonical artifact. bf16 for a compact artifact. |
| File suffix | `.ckpt` |
| Writer | The notebooks save through the training runtime |
| Reader | A parameter-only mode reads the parameter section and stops |
| Repository | One repository for the model family. One folder per stage. |
| Namespace | Derived from `HfApi.whoami()`, with an `HF_USERNAME` override |
| Visibility | Public by default |
| Schema | `docs/checkpoint-format.md` part A. The layout lives in `src/data.cc`. |
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

`NCHKPT01` is the only container the project writes, and the `version` word
says which layout inside it a file carries: 1 for a file this tree writes, and
0 for a file an older build wrote, which the committed fixture
`tests/data/checkpoint_v0.ckpt` pins. Reads accept both versions. They also
still accept the legacy `NCMDL001` container, and no writer produces it.

## 4. Writers

The notebooks save through the training runtime, and every writer writes
version 1, so every file a notebook, a script, or a test produces carries the
metadata block. The `version` word is the only difference between the writers;
they all use the one container.

| Writer | Run metadata | Optimizer section | Step record |
|---|---|---|---|
| `MeshRuntime.save` (`nc.MeshRuntime.save`) | the step and the data-parallel world size, from the runtime | AdamW, Muon | yes |
| `Checkpointer::SaveModel(model, optimizer, options, path)` | every field of `CheckpointSaveOptions`, plus `model.config()` | per the optimizer | when `options.step` is above zero |
| `Checkpointer::SaveModel(model, optimizer, step, path)` | the step, and the defaults elsewhere | per the optimizer | yes |
| `Checkpointer::SaveModel(model, path)` | the defaults: base, step 0, world size 1 | empty count | no |
| `nc.Model.save` (`nanochat_save`) | the defaults | empty count | no |

`Checkpointer::SaveModel` fills the block from `model.config()` and from the
`CheckpointSaveOptions` the caller passes (open decision O1), and validates the
config before the write, so an impossible config fails at the save and at the
load instead of at the first GEMM. `created_utc` stays empty: a save has to be
reproducible, and the field is informational, so a tool that wants a timestamp
writes one. `src/sft_main.cc` records `Stage::kSft` and
`RlWorker::SaveCheckpoint` records `Stage::kRl`, so a reader tells the stages
apart without a side file.

`nc.Model.save` writes the parameters only. The project no longer writes
`NCMDL001`; `Checkpointer::LoadModel` still reads a legacy file. The pretraining
and post-training notebooks call `MeshRuntime.save`. The post-training loop uses
a one-device `MeshRuntime`, so a save has the same code path as pretraining.

## 5. Byte layout

The container is a flat little-endian byte stream. It uses no pickle, no
language runtime, and no object graph. The writer is `Checkpoint::Save` in
`src/data.cc`.

### 5.1 Header and sections

Version 0 is the container with no run metadata block. The tree wrote it before
the metadata work, and every artifact that already sits on the Hub is one.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 8 | magic | the ASCII bytes `NCHKPT01` |
| 8 | 4 | version | `uint32`, the value 0 |
| 12 | 4 | parameter count | `uint32` |
| 16 | variable | the parameter records | see section 6 |
| after the parameters | 4 | optimizer count | `uint32` |
| after the count | variable | the optimizer records | see section 6 |

Version 1 is what every writer in section 4 produces. It adds the metadata word
count and the metadata words directly after the header, and nothing below them
moves.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 8 | magic | the ASCII bytes `NCHKPT01`, the same family word in both versions |
| 8 | 4 | version | `uint32`, the value 1 |
| 12 | 4 | metadata word count | `uint32`, above zero and at most `kMaxBlockWords` (64 KiB) |
| 16 | `4 * count` | the metadata words | `uint32` each, see section 5.3 |
| after the metadata | 4 | parameter count | `uint32` |
| then | variable | the parameter records | byte identical to the version 0 layout above |
| then | 4 | optimizer count | `uint32` |
| then | variable | the optimizer records | byte identical to the version 0 layout above |

The record framing is byte identical in both versions, so `ReadRecord`, the
Python decoder, and the `nanochat_checkpoint_*` surface read either version
without a branch. The metadata sits before the parameter count, so a
parameter-only read keeps its early stop: the optimizer section stays last.

The parameter section is always first. The optimizer section is optional. A
parameter-only file writes an optimizer count of zero.

### 5.2 Integer encoding

Every integer is little-endian. The helper functions `WriteU16`, `WriteU32`,
`WriteU64`, and `WriteI64` in `src/data.cc` serialize the values by hand. A
`uint32` value is four bytes, least significant byte first. An `int64` value is
the two's-complement form of a `uint64`.

The metadata words are `uint32` values in the same encoding, and each one is
four bytes: the block is a word stream, not a byte stream. Section 5.3 says how
a type that does not fit one word is carried.

### 5.3 The metadata section

The block is one `IFields` word stream (`include/nanochat/fields.h`), written
and read by `CheckpointMetadata` (`include/nanochat/checkpoint_metadata.h`).
The container writes the block's word count first, so a reader bounds the block
before it visits a word and rejects an oversized one early. A version 1 file
always writes the block, so a word count of zero is a corrupted header and not
a file from an older writer.

The word stream rules (`include/nanochat/fields.h`):

- `uint32`, `int32`, and `float` take one word each. A float travels as its bit
  pattern, so a read never converts a value.
- `uint64` takes two words, the low 32 bits first and then bits 32-63 (open
  decision O6). `step` is the one 64-bit field, and the order is part of the
  contract: a stream written high word first would still decode, to a wrong
  value.
- `bool` and an enum take one word, as the `uint32` value.
- A string takes one length word and then that many words of characters, zero
  padded to the last whole word; the reader trims the trailing nulls.
- A vector takes one count word and then the words of each entry.
- A nested block takes one size word and then its payload, which the size does
  not count. The size prefix is why an older reader can skip a block it does
  not know, and why a newer reader can tell how many words a block it knows
  carried.

`CheckpointMetadata::VisitFields` is the field order, and the order is the
format:

| Order | Field | Type | Stored as |
|---|---|---|---|
| 1 | `schema` | `uint32` | one word, 1 today |
| 2 | `stage` | `Stage` | one word, the enumerator value |
| 3 | `step` | `uint64` | two words, low 32 bits first |
| 4 | `world_size` | `uint32` | one word, 1 or more |
| 5 | `precision` | `ComputePrecision` | one word, the enumerator value |
| 6 | `config` | nested block | one size word, then the config fields |
| 7 | `tokenizer_id` | string | one length word, then the characters |
| 8 | `git_revision` | string | the same |
| 9 | `created_utc` | string | the same |
| 10 | `notes` | vector of strings | one count word, then one string per entry |

The `schema` is the metadata schema version, not the file version. It moves
only when the meaning of a field moves, which the append-only rule forbids, so
it stays 1.

The config block carries the `Config` members in the order
`VisitConfig` writes them (`src/checkpoint_metadata.cc`): `num_layers`,
`num_heads`, `num_kv_heads`, `hidden_dim`, `seq_len`, `vocab_size`,
`padded_vocab_size`, `rope_base`, `window_pattern`, `value_embedding`. `Config`
stays the one declaration of the model shape: there is no mirror struct, so no
field is duplicated and no drift test is needed (open decision O4). The load
validates the recorded config, and a mismatch between the file config and the
runtime config fails with a message instead of at the first GEMM.

The stage mapping (open decision O3) is the notebook vocabulary. The stored
value is the enumerator, so a new stage is a new field and never an inserted
enumerator:

| Notebook stage | `Stage` | Stored value |
|---|---|---|
| `base` | `Stage::kBase` | 0 |
| `sft` | `Stage::kSft` | 1 |
| `rl` | `Stage::kRl` | 2 |

There is no `kPretrain`: `base` is the pretraining stage.

The `precision` is a `ComputePrecision` from `include/nanochat/mfu.h` (open
decision O2), the enum a report row already names, so the tree keeps one
precision enum. The recorded value is informational: it never changes how a
file loads, and a version 0 file reports `fp32` under every `--config` build.

The caps (rule 3 of the block):

| Item | Cap | On a larger value |
|---|---|---|
| one string | 4096 words, `kMaxStringWords` | hard error |
| one vector | 1024 entries, `kMaxVectorEntries` | hard error |
| the whole block | 64 KiB, `kMaxBlockWords` | hard error before any word is visited |

The block is a few dozen words in practice, beside gigabytes of weights, so
these caps bound a malformed file and not a real one. They are the same kind of
bound as the container's `kMaxAttributes = 64`, and they keep the block from
growing into a second config file.

### 5.4 Compatibility rules

The rules come from design decisions D5, D6, and D7. Together they are the
reason a checkpoint that already sits on the Hub keeps loading after a field is
added, in both directions.

| Case | Rule | What the reader does |
|---|---|---|
| Trailing metadata words this build does not know (D5) | Accept and count | Reads the block, reports the `extra_u32` count it did not use, and prints one warning per load |
| Metadata fields this build does not know (D6) | Accept and count | Stops at the end of the block and keeps the member initializers; `IFields::ReadResult::missing_fields` counts them for a caller that visits the block directly |
| A version 0 file | Accept | Reads the record stream and reports the metadata defaults |
| A bad value inside the block (D7) | Hard error | `Checkpoint::Load` returns false with the field in `error`, and no partial load |
| A metadata word count of zero, or above the cap (D7) | Hard error | The header is rejected before a word is visited |
| A version word above 1 (D7) | Hard error | `unsupported checkpoint version` |
| An invalid recorded config (D7) | Hard error | The config is validated at the load, so a mismatch fails there |
| A record the build cannot read (D7) | Hard error | The existing record reader already reports the reason, and nothing moved |

The defaults are the member initializers of `CheckpointMetadata`, so an unset
field is always defined: `schema` 1, `stage` base, `step` 0, `world_size` 1,
`precision` fp32, the default `Config`, and empty strings and notes.
`created_utc` and `git_revision` are informational only (rule 4), so no load
decision depends on them.

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

There is no fp16 code. The container stores fp32 or bf16 (docs/precision.md).

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
| `optimizer/step` | the 1-based step, four raw little-endian `int32` bytes |

`<kind>` is one of `lm_head`, `embedding`, `value_embedding`, `resid`, `x0`,
or `smear`. The list is `ADAMW_KINDS` in `python/nanochat_cpp/checkpoint.py`
and `KindName` in `src/optim.cc`.

The step record uses dtype code 0 and shape `[1]`, but its four payload bytes
are an integer, not a float. The reader must not read the step as a float.

## 9. Reader modes

The container has two read modes, and both read either version.

| Mode | Reads | Used by |
|---|---|---|
| Parameter only | the header, the metadata block, and the parameter records | `nc.Model.load`, `MeshRuntime.load(optimizer=False)` |
| Full | the header, the metadata block, the parameter records, and the optimizer section | A resume through `MeshRuntime.load` |

The metadata block is small and every read needs it, so neither mode skips it:
a version 1 file yields its run facts in both modes, and a version 0 file
yields the defaults. The optimizer section stays the only section a mode can
skip.

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

The change is additive. The bump in `bindings/nanochat_capi.cc` is in the
tree, so the current value is `0.12.0`. Part B adds no C entry point and does
not bump the version again.

## 10. Versioning

The `version` word names the layout inside the magic's layout family. A reader
accepts the values 0 and 1, and rejects any other value with `unsupported
checkpoint version`, so an old build fails on a new file with a clear message
instead of a misparse.

| Version | Layout | Written by | Read by |
|---|---|---|---|
| 0 | the header, then the parameter and optimizer records, with no metadata block | the tree before the metadata work; pinned by the committed fixture `tests/data/checkpoint_v0.ckpt` | every build from here on |
| 1 | the metadata block between the header and the parameter count, then the version 0 record stream unchanged | every writer in section 4 | this tree |

Version 0 needs no compatibility branch beyond "read no metadata block, then
report the member initializers", so the load of an old artifact is the same
code path with one skipped section.

The version history is short because the layout changed once:

| Date | Version | Change |
|---|---|---|
| the first container | 0 | the fp32, bf16, and reserved int8 codes, the attribute block, and the optimizer section |
| the metadata work | 1 | the run metadata block after the header; the record stream is byte identical |

A new optimizer buffer becomes one more record, so that addition needs no
version change. A new record attribute needs no version change, because the
attribute block is already present. A new metadata field needs no version
change either, because the block carries its own length and its own drift
counters (section 5.4). A new dtype code needs a reader branch. A change to the
header, to the record layout, or to the meaning of a metadata field needs a
version change.

The magic names the layout family and `version` names the layout inside it, so
a pair that disagrees is a hard error. This tree writes one magic for both
versions (open decision O5), so the check only catches a file that changed
between the two reads. A second magic word protects no in-tree reader, and an
unknown version word already fails early, so the version word is the whole
layout gate.

Old `NCMDL001` files stay readable. `Checkpointer::LoadModel` routes on
`DetectFormat`: `kContainerV0` and `kContainerV1` go to `Checkpoint`, and
`kLegacyModel` goes to the legacy reader. A version word above 1 reads as
`kContainerV1` from `DetectFormat` and is then rejected by the reader, which is
the one hard error the gate needs.

The check that old code rejects a new file is a release gate and not an
automated test in this tree (open decision O8): one build of a fixed parent
commit, one version 1 file, and the error message expected.

## 11. The schema in machine-readable form

The bytes are the contract. The decoded view is one object, and a version 1
file adds the `metadata` member:

```json
{
  "magic": "NCHKPT01",
  "version": 1,
  "metadata": {
    "schema": 1,
    "stage": "base",
    "step": 2000,
    "world_size": 2,
    "precision": "fp32",
    "config": {"num_layers": 12, "num_heads": 12, "hidden_dim": 768},
    "tokenizer_id": "vocab-32768.nctoken",
    "git_revision": "0123456",
    "created_utc": "",
    "notes": []
  },
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
dtype is 0 and the byte count doubles. The `config` member carries one key per
`Config` member; the three keys above are shown to keep the example short. A
version 0 file decodes to the same object without the `metadata` member, and
`checkpoint.read_metadata` reports the defaults for it.

The constants are the C values. The C surface reports them:
`nanochat_checkpoint_dtype` enumerates each dtype with its name, code, and
element size, and `nanochat_checkpoint_metadata_read` reports the run metadata
block.

| Constant | Source |
|---|---|
| `MAGIC` | `kCheckpointMagic` in `src/data.cc` |
| `VERSION` | the `Checkpoint::Save` header, 1 |
| `METADATA_SCHEMA` | `kMetadataSchema` in `include/nanochat/checkpoint_metadata.h`, 1 |
| `STAGE_BASE`, `STAGE_SFT`, `STAGE_RL` | `Stage::kBase`, `kSft`, `kRl`, stored 0, 1, 2 |
| `DTYPE_FP32`, `DTYPE_BF16`, `DTYPE_INT8` | `DType::kFp32`, `kBf16`, `kInt8` |
| `MAX_NDIM` | the `ReadRecord` limit, 8 |
| `MAX_METADATA_WORDS` | `kMaxBlockWords` in `include/nanochat/fields.h`, 64 KiB |

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

The metadata block is negligible beside the weights: a full one is a few dozen
words, and the cap is 64 KiB. Version 0 has no block at all.

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
`python/nanochat_cpp/checkpoint.py`, `_muon_records` raises "this is a sharded
rank shard, not a single-rank shard". The check runs when
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
| `notebooks/mesh_train.py`, `mesh_train_test.py` | keep | The thin pretraining launcher over the C++ mesh training driver |
| `notebooks/mesh_worker.py` | keep | The training worker |
| `notebooks/pi_agent.py`, `pi_agent_test.py` | keep | The setup notebook agent |
| `notebooks/kaggle_setup.py`, `kaggle_setup_test.py` | keep | The Kaggle session helpers |
| `notebooks/kaggle_demo.py` | keep, docstring edit | The SFT mixture helper and the offline fallback |

## 15. Shared setup

Every notebook runs the setup first. A reader can run any one notebook alone.
The body lives in `notebooks/session.py`, so the code has one home.

The repository must be on disk and on `sys.path` before a notebook imports
`session`. Cell 1 clones the repository and extends `sys.path`. Cell 2 calls
`session.setup`. Cell 3 reports the devices.

```python
# Cell 1: put the repository on disk and on sys.path.
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ming6ao/nanochat.cpp"
REPO = Path("/kaggle/working/nanochat.cpp")
BASE = Path("/kaggle/working/nanochat-cache")

if not (REPO / ".git").is_dir():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
sys.path[:0] = [
    str(REPO / "python"),
    str(REPO / "notebooks"),
    str(REPO / ".agents" / "skills" / "model-fit" / "scripts"),
]
```

```python
# Cell 2: secrets, bootstrap, build, and the device check.
import session

sess = session.setup(REPO_URL, REPO, arch="sm_75", devices=2, base=BASE)
```

```python
# Cell 3: report the devices.
import nanochat_cpp as nc

for device in nc.devices():
    print(device, device.name,
          f"{device.total_memory_bytes / 2**30:.1f} GiB")
```

The module exposes:

```python
@dataclasses.dataclass
class Session:
    host: Host
    secrets: dict[str, str]
    devices: list
    base: Path

def load_secrets(names=("HF_TOKEN",)) -> dict[str, str]: ...
def clone_and_build(repo_url, repo_path, ref=None, arch="sm_75",
                    devices=1, base=None) -> Session: ...
def require_devices(count=1) -> list: ...
def setup(repo_url, repo_path, *, arch="sm_75", devices=1,
          base=None) -> Session: ...
```

The `devices` argument is the number of cards the notebook needs. The
pretraining notebook passes `devices=2`. The evaluation and post-training
notebooks keep the default `devices=1`, because both use one card.

`session.py` wraps `notebooks/kaggle_setup.py`, which keeps the repository
fetch, the bootstrap, and the build. The new `setup` adds the secrets load and
the device check. The `model_fit` path from cell 1 gives the pretraining
notebook its parameter report.

`session.py` imports `kaggle_secrets` and `huggingface_hub` inside the
functions. The tests then run without either package.

## 16. Shared Hub helper

`notebooks/hub_checkpoint.py` holds every Hugging Face Hub call.

```python
#: The files one notebook needs to load a stage.
STAGE_PATTERNS = ("checkpoint.ckpt", "model.json", "tokenizer.nctoken")

#: STAGE_PATTERNS plus the provenance files the post-training notebook copies.
FULL_STAGE_PATTERNS = STAGE_PATTERNS + ("train.json", "run.json",
                                        "metrics.jsonl", "summary.json",
                                        "README.md")

def default_namespace(api=None, environ=None): ... # HF_USERNAME or whoami
def tokenizer_file(vocab_size): ...                # vocab-<size>.nctoken
def repo_id_for(run_name, namespace=None, prefix="nanochat", api=None): ...
def stage_dir(root, stage): ...                    # <root>/<stage>
def build_run_record(...) -> dict: ...             # pure metadata assembly
def push_stage(repo_id, stage, folder, *, token=None, private=False,
               commit_message="", tag=None, api=None): ...
def push_file(repo_id, path_in_repo, local_path, *, token=None,
              commit_message="", api=None): ...
def pull_stage(repo_id, stage, dest, *, revision="main", token=None,
               patterns=STAGE_PATTERNS, download=None): ...
def pull_tokenizer(repo_id, vocab_size, dest, *, revision="main", token=None,
                   download=None): ...          # path, or None on a miss
def push_tokenizer(repo_id, vocab_size, local_path, *, token=None,
                   commit_message="", api=None): ...
def load_stage(repo_id, stage, dest, *, revision="main", token=None,
               device="cuda", download=None, api=None): ...
                                                   # model, tokenizer, args
```

`load_stage` calls `nc.Model.load`, which uses the parameter-only read. The
`api` and `download` arguments default to the real Hugging Face calls. A test
passes a fake object instead.

### 16.1 Upload

Every Hugging Face import stays inside the function, so the CPU test imports
the module without the wheel. The caller can pass `api`. The default builds
`HfApi(token=token)`.

```python
def push_stage(repo_id, stage, folder, *, token=None, private=False,
               commit_message="", tag=None, api=None):
    from huggingface_hub import HfApi

    api = api or HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model",
                      folder_path=str(folder), path_in_repo=stage,
                      commit_message=commit_message, token=token)
    if tag:
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model",
                       token=token)
```

`push_file` is the same call with one file:

```python
def push_file(repo_id, path_in_repo, local_path, *, token=None,
              commit_message="", api=None):
    from huggingface_hub import HfApi

    api = api or HfApi(token=token)
    api.upload_file(repo_id=repo_id, repo_type="model",
                    path_or_fileobj=str(local_path),
                    path_in_repo=path_in_repo,
                    commit_message=commit_message, token=token)
```

### 16.2 Download

```python
def pull_stage(repo_id, stage, dest, *, revision="main", token=None,
               patterns=STAGE_PATTERNS, download=None):
    from huggingface_hub import snapshot_download

    download = download or snapshot_download
    local = download(
        repo_id=repo_id, repo_type="model", revision=revision,
        local_dir=str(dest), token=token,
        allow_patterns=[f"{stage}/{p}" for p in patterns])
    return Path(local) / stage
```

`STAGE_PATTERNS` holds the three files a load needs. The evaluation notebook
does not download `metrics.jsonl` or `summary.json`.

### 16.3 Load

The evaluation notebook loads into an `nc.Model`:

```python
model_args = ModelArgs.from_json((folder / "model.json").read_text())
model = nc.Model(model_args.to_config(), device=device, seed=0)
model.load(str(folder / "checkpoint.ckpt"))   # parameters only
tokenizer = nc.Tokenizer.load(str(folder / "tokenizer.nctoken"))
```

The chat cell passes the loaded tokenizer. `nc.chat.ChatEvaluator` otherwise
looks for `NANOCHAT_BASE_DIR/tokenizer/tokenizer.nctoken`, which a fresh
session does not hold (`python/nanochat_cpp/chat.py`):

```python
report = nc.chat.ChatEvaluator(model, tokenizer=tokenizer).run(...)
```

The post-training notebook loads into a one-device runtime:

```python
runtime = nc.MeshRuntime(nc.Mesh((nc.devices()[0],)),
                         model_args.to_config(), optimizer_config,
                         batch=1, grad_accum=1, seq=SFT_SEQ, seed=42)
runtime.load(str(folder / "checkpoint.ckpt"), optimizer=False)
```

`optimizer_config` comes from `args.TrainArgs`:

```python
SFT_TRAIN = args.TrainArgs(num_iterations=SFT_STEPS, embedding_lr=0.3, ...)
optimizer_config = nc.optimizer_config(**SFT_TRAIN.optimizer_kwargs())
```

`nc.optimizer_config` and `nc.Optimizer` share one field set, so the runtime
loop and the retired `nc.Model` loop cannot drift. The runtime schedule reads
`num_iterations`, so `SFT_TRAIN.num_iterations` equals `SFT_STEPS`.

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
    train.json             # the mesh driver job or the TrainArgs
    run.json               # lineage and provenance
    tokenizer.nctoken
    metrics.jsonl
    summary.json
    README.md              # the model card
  sft/
    (the same file names)
  reports/
    <stage>-<revision>.json   # one evaluation report per stage and revision
```

### 17.2 Metadata files

`model.json` holds `args.ModelArgs.to_json()` output. The container does not
store the shape, so this file is mandatory.

`train.json` holds the training knobs. For pretraining it holds the mesh driver
job, which the notebook writes from the job dictionary. For post-training it
holds `args.TrainArgs.to_json()` output.

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
| 2 | Setup: secrets, clone, bootstrap, build, device check (`devices=2`) |
| 3 | Download the two ClimbMix shards |
| 4 | Pull the tokenizer from the Hub; else train it and push it |
| 5 | `args.ModelArgs` and the `model_fit` report |
| 6 | Build the mesh driver job, then `mesh_train.launch` and stream the record |
| 7 | Write `model.json`, `train.json`, `run.json`, and the model card |
| 8 | `hub_checkpoint.push_stage` for the base folder, then tag the revision |
| 9 | Short sanity check: a few validation steps and two samples |
| 10 | Markdown: notes and the next notebook |

`MeshTrainConfig.checkpoint_path()` returns `checkpoint.ckpt` after the rename
in section 24.

The tokenizer lives in one repository per namespace:
`<namespace>/nanochat-tokenizer`. Each vocabulary size is one file at the
repository root, for example `vocab-32768.nctoken`. One `manifest.json` holds
the provenance: the corpus, the input shard, the training command, the
`rustbpe` commit, and the file digest. A pull misses on the first run, so the
notebook trains the tokenizer and pushes it.

The pretrain notebook builds the file name from `ModelArgs.vocab_size`, so no
manifest lookup is needed. The model stage still holds the container as
`tokenizer.nctoken`.

## 19. Notebook: `eval.ipynb`

| Cell | Content |
|---|---|
| 1 | Markdown: evaluation is forward only, and it writes no checkpoint |
| 2 | Setup (`devices=1`) |
| 3 | `REPO_ID`, `STAGE`, `REVISION`; `hub_checkpoint.load_stage` |
| 4 | Download one ClimbMix validation shard for the bits-per-byte loop |
| 5 | The train and validation bits-per-byte loop |
| 6 | The sample prompts and the greedy completions |
| 7 | `nc.chat.ChatEvaluator(model, tokenizer=tokenizer)` on the reference tasks |
| 8 | Write one JSON report; `hub_checkpoint.push_file` it to `reports/` |
| 9 | Markdown: notes |

`STAGE` is `base` or `sft`. The load reads the parameter section only. The Hub
download still fetches the whole `checkpoint.ckpt`, but the reader does not
parse the optimizer section.

## 20. Notebook: `posttrain.ipynb`

| Cell | Content |
|---|---|
| 1 | Markdown: supervised fine-tuning of a base checkpoint |
| 2 | Setup (`devices=1`) |
| 3 | `hub_checkpoint.pull_stage` for the `base` stage |
| 4 | Build the mesh driver config; the warm start is the base parameters only |
| 5 | The base completion before fine-tuning |
| 6 | The conversation mixture and the `NCSFT001` shard write |
| 7 | Run the driver on the packed shard, then check that the loss decreased |
| 8 | The metadata files; the driver wrote the checkpoint |
| 9 | `hub_checkpoint.push_stage` for the `sft` stage, with the base revision |
| 10 | A small chat check |
| 11 | Markdown: notes |

The driver uses one card, `world_size=1`, `per_rank_batch=1`, `grad_accum=1`,
and `seq=SFT_SEQ`. The warm start loads the base parameters only, so the reader
skips the pretraining optimizer section. The masked targets use `-1` for ignored
positions. The driver reads the `NCSFT001` shard that `nc.sft.pack` wrote, so
Python never steps the optimizer.

The runtime takes its optimizer and schedule from `args.TrainArgs`:

```python
SFT_TRAIN = args.TrainArgs(num_iterations=SFT_STEPS, embedding_lr=0.3,
                           unembedding_lr=0.008, matrix_lr=0.02,
                           scalar_lr=0.5, weight_decay=0.0, warmup_steps=2,
                           warmdown_ratio=0.5, final_lr_frac=0.0)
optimizer_config = nc.optimizer_config(**SFT_TRAIN.optimizer_kwargs())
```

`nc.optimizer_config` and `nc.Optimizer` share one field set
(`python/nanochat_cpp/api.py`), so the runtime loop and the retired
`nc.Model` loop cannot drift. The runtime schedule reads `num_iterations`, so
`SFT_TRAIN.num_iterations` equals `SFT_STEPS`. The mixture comes from
`kaggle_demo.build_sft_conversations`, which uses `nc.sft_data.demo_mixture`
and falls back to the offline conversation set.

`ForwardLoss` and `BackwardAccumulate` already ignore `-1`, so the runtime
handles the supervised loss. Add one test that drives a one-device runtime with
a `-1` target and checks that the loss ignores it.

The save writes the parameters plus the SFT optimizer state, so the SFT run can
resume. The base load reads the parameter section only. The full chat
evaluation stays in `eval.ipynb` with `STAGE = "sft"`.

The scripted `nc.sft.run` bridge and `sft_main` stay for a one-shot job.
`docs/post-training.md` section 8 already names both entries: the bridge and
the in-process one-device `MeshRuntime` loop in `notebooks/posttrain.ipynb`.
Keep the section as it stands. The notebook entry is now in the tree, so the
architect should add it to the status table in `docs/post-training.md`
section 11.

## 21. Failure handling

- A missing `HF_TOKEN` stops an upload with the corrective message. It does not
  stop training.
- A failed download falls back to `/kaggle/working/<run_name>` when it exists.
- A missing `model.json` or `tokenizer.nctoken` stops the load and names the
  file.
- `create_repo` can fail when the token lacks write access. Catch the error,
  report the repository name, and keep the local copy.
- A repeated run can fail at `create_tag` when the tag exists. Catch the
  error, report the tag, and keep the uploaded files.
- A missing token can stop `repo_id_for` at `HfApi.whoami()`. Report the
  corrective command before the first upload.
- A private repository needs the token for the download too.
- Never print the token.

## 22. Tests

`notebooks/BUILD.bazel` adds:

```
py_library(name = "session", srcs = ["session.py"], imports = ["."],
           deps = ["//python:nanochat_cpp", ":kaggle_setup"])
py_library(name = "hub_checkpoint", srcs = ["hub_checkpoint.py"],
           imports = ["."], deps = ["//python:nanochat_cpp", ":args"])
py_test(name = "session_test", srcs = ["session_test.py"], size = "small",
        deps = [":session"])
py_test(name = "hub_checkpoint_test", srcs = ["hub_checkpoint_test.py"],
        size = "small", deps = [":hub_checkpoint"])
```

Both tests join the existing `test_suite(name = "all", ...)`.

`hub_checkpoint.py` imports `huggingface_hub` inside each function, and each
public function takes an `api` or `download` argument. The Hub test passes a
fake object with `whoami`, `create_repo`, `upload_folder`, `create_tag`,
`upload_file`, and a fake download function. No test imports
`huggingface_hub`, so the CPU tests stay wheel-free.

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

`tests/checkpoint_compat_test.cc` is the version 0/1 guard, and it is the file
to read before changing this layout. It loads the committed fixture
`tests/data/checkpoint_v0.ckpt` and asserts the exact record names, storage
dtypes, shapes, and payload checksums, so a change that moves a version 0 byte
fails there instead of on a Hub artifact. It round-trips a version 1 file,
accepts and counts both drift directions of section 5.4, and pins the hard
errors of that table. It runs in the T0 loop and under `--config=bf16` and
`--config=fp8`, because the recorded `precision` must never change a load
decision:

```
tools/nanochat test //tests:checkpoint_compat_test
tools/nanochat test //tests:checkpoint_compat_test --config=bf16
tools/nanochat test //tests:checkpoint_compat_test --config=fp8
```

`src/fields_test.cc` covers the visitor itself, ported case for case from
`gemma.cpp` `io/fields_test.cc`: the round trip, new code over old data, old
code over new data, an invalid float, an invalid string, a second write, the
vector cap, and the enum check.

## 23. Definition of Done

- Three notebooks exist. Each one runs end to end on a two-T4 Kaggle session.
- Each notebook performs the setup itself.
- The notebooks write one container: `NCHKPT01` with the `.ckpt` suffix.
- `nc.Model.load` and `MeshRuntime.load(optimizer=False)` skip the optimizer
  section.
- The pretraining notebook publishes the base stage. The eval and
  post-training notebooks load it in a later session with no retraining.
- The post-training notebook publishes the fine-tuned stage.
- The tokenizer repository holds one `vocab-<size>.nctoken` file for each
  vocabulary size.
- `tools/nanochat test` passes for the notebook tests and the container test.
- No notebook prints a token.

## 24. Sequencing

Steps 1 and 2 are in the tree, and so is step 9. Steps 3 to 7 are implemented;
each notebook needs one Kaggle session to verify end to end. Step 8 records the
handoff to the architect.

1. The version 0 container, the storage dtype codes, the attribute block, and
   the `checkpoint.ckpt` rename are in the tree. The C surface bump landed with
   them, so `kNanochatVersion` is `0.12.0`. Part B adds no C entry point and
   does not bump the version again.
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
8. Part A is the promoted schema in this file. The architect owns the file
thereafter.
9. The run metadata block is in the tree. Writers write version 1, the reader
   accepts version 0 and version 1, `tests/checkpoint_compat_test.cc` pins
   both, and the C surface and `python/nanochat_cpp/checkpoint.py` report the
   block. The layout is sections 5.3 and 5.4 of this file, and the two examples
   in `examples/` are the runnable form of it.

## 25. Open items

1. Resolved. The one-device runtime masked-loss test is in
   `src/mesh_runtime_test.cc` (`TestMaskedTargetsIgnoreLoss`), so the runtime
   loop is the fine-tuning path. `nc.Model` plus `nc.Optimizer` is not needed.
2. `docs/post-training.md` section 8 already records the in-process notebook
   entry beside the `sft_main` bridge. No further revision to section 8 is
   needed. The architect should add the notebook entry to the status table in
   section 11.
3. The model card template needs a final list of fields.
4. Decision: the tokenizer stays in a separate repository. The name is
   `<namespace>/nanochat-tokenizer`. Each vocabulary size is one root file,
   for example `vocab-32768.nctoken`, with one `manifest.json` for provenance.
   Confirm with the architect.
