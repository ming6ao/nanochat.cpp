# API surface and modularity review

This document records the API of each component at HEAD, then proposes a smaller
API for each one. It is a review, not a specification. The frozen header set in
`AGENTS.md` section 1 governs any change here. Every proposal that edits a frozen
header needs an architect decision first.

Scope: the C++ and CUDA tree (about 51,700 lines), the C ABI, the Python
surface, and the process boundary. The counts below are exact for HEAD. The
anchors name the file and the line.

---

## 1. Component inventory

| Component | Main files | Lines | Public entry points | Owner |
|---|---|---|---|---|
| Contract headers | `include/nanochat/{tensor,kernels,config,model,model_spec,plan,capi}.h` | 2,289 | 30 seam symbols, 1 model factory, 68 C functions | Architect |
| CPU backend | `backends/cpu/kernels.cc` | 1,839 | 30 seam symbols + 2 mesh symbols | Runtime |
| CUDA backend | `backends/cuda/{device,gemm,row_reduce,span_reduce,nccl_sync}.cu`, `kernels/*.cu` (11 families), `collective/` | — | the same 30 + 2, plus 1 grad-sync factory | Kernel agents |
| Capability data | `backends/{cpu,cuda}/caps.json`, `src/device_profile.cc`, `src/capability_names.*`, `backends/*/caps_names.*` | — | `FindDeviceProfile`, `CapsFromProfile`, `DeclaredFeatureMask` | Runtime |
| Engine | `src/model.cc` 1,381, `src/ops.cc` 1,568, `src/generate.cc` 666, `src/optim.cc` 839, `src/model_impl.h`, `src/workspace.h` | 4,454 | 2 `Model::Create`, 12 virtuals, 7 free functions, 23 `ops::` functions | Workflow, Optimizer |
| Harness | `src/train.cc` 1,248, `src/data.cc`, `src/eval.cc`, `src/cli.h` 129, 10 `*_main.cc` | — | `TrainLoop`, `Checkpointer`, `EvaluatePackedRows`, 139 flag branches | Harness |
| C ABI | `include/nanochat/capi.h` 633, `bindings/nanochat_capi.cc` 1,647 | 2,280 | 68 functions, 25 struct typedefs, 3 enum typedefs | Python surface |
| Python surface | `python/nanochat_cpp/` (28 submodules), `python/tests/` | — | 45 exported names, 2 doors (in-process and process) | Python surface |
| Data and model definition | `include/nanochat/{data,dataloader,tokenizer,scheduler,mesh}.h`, `src/tokenizer/`, `src/parquet/`, `src/model_spec.cc`, `models/*/spec.json`, `spec/*.md` | — | 2 `DocumentSource` virtuals, 8 `Tokenizer` virtuals, 6 spec functions | Data pipeline, Contract |
| Process boundary | `tools/nanochat`, `python/nanochat_cpp/{cli,sandbox,gpu_queue}.py`, `tools/sandbox/profiles.json` | — | 22 subcommands, 8 profiles | Architect |

---

## 2. Component APIs

### 2.1 Contract headers

`tensor.h` (322 lines) carries three unrelated things. It carries the build
precision policy as `constexpr` data (`BuildPrecision`, `kBuildPrecision`,
`kComputeIsBf16`, line 135 to 160). It carries the on-disk dtype registry
(`DType`, `DTypeTraits`, `SizeOf`, line 194 to 232). It carries the device
capability table `Caps` (line 273 to 317) and a closed vocabulary of 15 feature
bits (line 250 to 271). There is no `Tensor` type. The only `struct Tensor` in
the tree is a test fixture.

`kernels.h` (552 lines) is the backend seam. It declares 30 functions in
`namespace nanochat::kernels`, 13 parameter structs, 5 enums, and
`kKernelSeamVersion = 2`. The 30 divide into 6 device and memory functions,
3 matrix-multiply functions, 1 fp8 quantizer, 20 compute and reduction
functions. The header closes two enums. `PointwiseOp` holds 9 codes and
`ReduceOp` holds 4 codes. A backend switch must name every code and must have
no `default` arm (line 62 to 66).

The seam has no failure channel. A bad parameter produces a wrong number, not a
status. The oracle test is the only gate.

`config.h` (112 lines) derives every hyperparameter from 11 fields and exposes
10 inline methods. `Validate` (line 70) is the one gate that every entry point
shares.

`model.h` (290 lines) is the user-facing graph API. It has 2 static factories
and 12 virtuals in `Model` plus its destructor, of which 8 are pure. `KvCache`
adds 2 pure virtuals and a destructor. It also has 7 free functions for cache
creation, prefill, decode, evaluation, scoring, and generation.

`model_spec.h` (222 lines) describes a model as data with 9 closed enums and
7 structs. The closed-field rule (line 15 to 30) forbids free-form fields, so
the spec stays out of graph-IR territory.

`plan.h` (158 lines) is the pure planner. `BuildPlan` reads the spec, the graph
kind, the device index, and `kernels::GetCaps()`. It allocates no memory and
launches no kernel. It returns `ok == false` plus a `gaps` list, and it never
throws.

`capi.h` (633 lines) is a flat C mirror of the C++ API. It has 68 functions,
7 opaque handle types, 18 data structs, and 3 enums. No C++ type crosses the
boundary. Three conventions describe errors: `nanochat_status`, a thread-local
string, and an element count or a negative error.

### 2.2 Backend seam and backends

Backend selection is a link decision, not a runtime registry. `BUILD.bazel`
defines `config_setting` values for `backend=cpu` and `backend=cuda`.
`src/BUILD.bazel` selects the backend archive. Two backends cannot link into
one binary, because both define the same strong symbols.

A device backend implements 33 symbols: the 30 seam functions,
`EnumerateDevices` and `SelectDevice` from `mesh.h`, and `CreateGradientSync`
from `distributed.h`. The CPU backend implements 32, because it takes
`CreateGradientSync` from `src/distributed.cc:141`. CUDA overrides that factory
in `backends/cuda/nccl_sync.cu:144`. There is no partial backend. A missing
symbol fails the link.

The dispatch path is direct. `src/ops.cc` calls `kernels::RmsNormForward`. That
symbol resolves to `backends/cuda/kernels/rms_norm.cu:123`. The host wrapper
launches a file-local `__global__` kernel through `cuda_backend::Launch`. No
host file above the seam names a CUDA symbol. `src/ops.cc` holds 79 `kernels::`
call sites, and it is the only runtime path into the seam.

One runtime capability switch exists: `ops::Fused(feature)` at `src/ops.cc:47`.
It caches `GetCaps()` once and returns `caps.HasFeature(feature)`. Call sites
branch on it for `rms_norm`, `qk_prep`, and `classifier`. The portable branch is
a composition over the primitive tier, so a device without fused code still
runs the whole graph.

### 2.3 Capability data

Two `caps.json` files declare 6 device entries: `cpu`, `h100`, `h200`, `a100`,
`t4`, and `gtx1080ti`. Each entry has 24 fields: 17 device facts plus
`utilities[6]`, `primitives[11]`, `pointwise_codes[9]`, `reduce_ops[4]`,
`attention`, `stateful[2]`, and `features[]`.

Two consumers read the same data by different paths. The runtime path is
`backends/cuda/device.cu:138`, which merges the declared mask with four
precision bits that come from `cudaDeviceProp`. The planner path is
`src/device_profile.cc:238`, which parses both files into a process-wide table.

The 14-row name table exists three times: `src/capability_names.h`,
`backends/cpu/caps_names.h`, and `backends/cuda/caps_names.{h,cc}`. The copy is
deliberate, because `src/` must not include a backend header, and the copy
cross-checks the header instead of trusting it. `backends/cpu/caps_test.cc:59`
adds six more hand-written closed lists: the vocabulary, the reference
features, the utilities, the pointwise codes, the reduce operations, and the
composition families.

The two lists do not match one to one. `spec/capability_bits.md` lists 15 bits
and `tensor.h` declares 15 constants, but `override_value_gate` has no constant
and `kFeatureFusedAttention` has no vocabulary row. All three tables therefore
carry 14 rows. The backend list `kBackendNames` at `src/device_profile.cc:53`
is a hand-written pair.

### 2.4 Engine

`src/model.cc` owns the graph and the buffers. `TrainModel` allocates each
parameter as `value`, `grad`, and `master` (`model.cc:337`). The model owns the
activations in one bump arena (`src/workspace.h`). The optimizer owns the
moment buffers.

The forward and the backward are literal sequences of `ops::` calls.
`RunForward` (`model.cc:1047`) and `BackwardInternal` (`model.cc:1202`) share
one topology. Saved activations are explicit struct fields in
`BlockActivations`, not a tape.

`src/ops.cc` is the cleanest boundary in the tree. It includes neither
`model.h` nor `config.h`, and it never names `Config`. It is shape-driven only.

`src/generate.cc` builds a second forward pass. It includes `src/model_impl.h`
(line 9), downcasts `Model*` to `TrainModel*` (line 429 to 443), and carries an
unused `nanochat/dataloader.h` include (line 21).

`src/optim.cc` carries AdamW, Muon, and the `Scheduler` methods. The scheduler
declaration is a separate public header, so scheduling has two homes.

### 2.5 Harness and command line

`src/cli.h` is not a parser. It supplies numeric parsers, CSV helpers, the
environment readers, and the 10 model flags through `IsModelFlag` and
`ApplyModelFlag`. Each binary parses its own arguments.

Ten binaries carry 139 flag branches in ten hand-written loops, which is about
one branch per `--flag`. Each loop repeats the same prologue: help, model-flag
redirect, a value check, then three near-identical numeric arms. `Usage()` text
exists ten times. `MakeParquetFactory` is byte-identical in three files.
Defaults and parse strictness differ between binaries for the same flag.

The training step appears three times: `Model::TrainStep`, `TrainLoop::Run`,
and the hand-written loop in `src/sft_main.cc`.

### 2.6 Python surface and C ABI

The C ABI maps one to one onto the C++ API. The name rule is `nanochat_` plus
snake case, so a reader can find the C++ body from the C name. 36
`static_assert` guards in `bindings/nanochat_capi_test.cc:34-155` turn a layout
change into a compile error. 56 `Guard` call sites convert a C++ throw into a
status, and they wrap 2 `try` blocks.

The Python side mirrors the ABI by hand. `_core.py` declares 18 `ctypes`
structures and 68 signatures, and every function is bound. `nanochat_optim_config_size`
runs on load, and the loader rejects a library that reports another size.

One Python module re-implements a C++ wire format: `data.py` reads the
`NCTOKEN1` artifact that `src/tokenizer/tokenizer.cc` owns. Three neighbours are
deliberate. `checkpoint.py` is already an ABI client (`:575` calls
`nanochat_checkpoint_read`), and it owns the name remap, the dtype policy, and
the path rules. `eval_fixture.py` is stdlib-only by design (`:6`), so a fixture
builds with no shared library. `plan.py` mirrors the reference script, not C++,
and it delegates the two model-derived numbers to `nanochat_params_get` (line 1
to 12). `mesh.py` re-implements the placement language.

The examples are the intended minimal API. `hello_world.cc` needs 11 lines from
`Config` to generated tokens. `checkpoint_roundtrip.cc` needs 6 calls.

### 2.7 Data, model definition, distribution

`DocumentSource` has 2 virtuals. `ShardDocumentSourceFactory` adds the rank
stride and the resume position. `DataLoader` packs tokens behind a producer
thread. `DataLoaderState` is the resume record.

`Tokenizer` has 8 virtuals. `LoadTokenizer` rejects any split pattern that is
not the built-in one, so a new pattern needs a version bump.

A new model is data only. It needs `models/<name>/spec.json`, a `BUILD.bazel`
with 3 targets, fixtures, tolerances, and one root label. Only `plan_main`
passes a spec name today. The other entry points call the plan-less
`Create(config)`.

`GradientSync` has 5 virtuals and one factory. Two drivers consume it. The
multi-process driver lives in `src/train.cc` and writes a data-cursor sidecar.
The in-process mesh driver lives in `src/mesh_train.cc` and has no such sidecar.

### 2.8 Process boundary

`tools/nanochat` has 22 subcommands and one authoritative row table in
`_entry.py:34-171`. Every binary gates on `RequireSandboxOrDie`, except
`src/plan_main.cc`, which computes nothing on a device.

---

## 3. Data flow through the whole system

```text
tools/nanochat  ->  sandbox.py  ->  systemd user scope  ->  *_main binary
                                                                |
                          cli.h flags                           v
   spec.json --> ModelSpec --> Config --> BuildPlan --> Model::Create
                                                     |
                                                     v
                                          TrainModel (params, arena)
                                                     |
                        +----------------------------+-------------------+
                        v                                                v
                    src/ops.cc  --->  kernels.h  --->  backends/{cpu,cuda}
                        ^                                                |
                        |                                                v
              generate.cc / optim.cc                          caps.json (data)
                        |                                                |
                        v                                                v
                 Python surface <-- capi.h <-- bindings        plan.cc (planner)
```

---

## 4. Findings

Each finding states what is not simple, and the evidence for it.

| # | Finding | Evidence |
|---|---|---|
| F1 | The primitive tier is a contract promise and a capability bit, but not a link-level option. A device that implements 16 entries still defines 14 symbols nobody calls. | `docs/kernels.md:92`, `tensor.h:255`, and the only weak symbol is `src/plan.cc:403` |
| F2 | No `Tensor` type exists, and the allocator lives inside the seam. | `kernels.h:394-403`; `src/ops.cc` 6 `Alloc` and 7 `Free` |
| F3 | `tensor.h` mixes three concerns. | precision, dtype, `Caps` in one 322-line header |
| F4 | The seam has no error channel. | `kernels.h:10-12`; `backends/cuda/device.h:23-25` |
| F5 | The C ABI mirrors 18 C++ structs by hand and keeps growing. | `capi.h`; 72 field copies in the binding |
| F6 | Three error conventions share one stack. | seam silent, headers use `bool` plus string, ABI uses a status |
| F7 | Resolved in step 1. The capability mask had two producers and a fallback, and the reference backend reported zero. | `backends/cpu/kernels.cc:182-213` now reads `DeclaredFeatureMask("cpu")`; the `src/plan.cc` fallback is gone |
| F8 | The capability vocabulary exists three times, six more closed lists are hand-written in one test, and the backend list is hardcoded. | 3 name tables; `backends/cpu/caps_test.cc:59-123`; `src/device_profile.cc:53` |
| F9 | One bit carries three names, and the header spelling describes a folding the seam does not perform. | `spec/capability_bits.md:80,210`, `tensor.h:266`, `src/capability_names.h:86` |
| F10 | Flag parsing is duplicated across 10 binaries, and defaults disagree. | 139 flag branches, 10 loops, 10 `Usage()` copies |
| F11 | The training step is written three times. | `model.cc:1312`, `train.cc:1049`, `sft_main.cc:520-540` |
| F12 | `model.cc` is the factory, the spec validator, and the plan entry point. | `model.cc:546-608`; two `Create` overloads |
| F13 | Inference depends on training internals. | `generate.cc:9, 429-443`; dead `dataloader.h` include at line 21 |
| F14 | The performance model lives in three places. | `mfu.h` declares, `train.cc:158-360` implements, `performance_model.cc:359` calls |
| F15 | Python re-implements one C++ wire format. | `data.py` (`NCTOKEN1`) |
| F16 | Two distribution drivers behave differently on resume. | `train.cc:850-944` has a cursor sidecar; `mesh_train.cc` has none |
| F17 | `spec/block_kinds.md` names kinds that `model_spec.h` does not declare. | documented in `models/nanochat-default/README.md` |
| F18 | Resolved in step 3. The multi-process resume counted the epoch twice and dropped the restored buffer, so a resume at epoch 1 or later skipped the documents that were packed but not consumed. | `src/train.cc` passed `start_on_creation = resume_state_.epoch` and then called `Reset()` once per stored epoch; the new test in `//src:train_parallel_test` failed three ways before the fix |

---

## 5. Recommendations

The order below is the recommended order of work. Cost is small, medium, or
large. "Contract" marks a change that needs an architect decision, because it
edits a frozen header or a spec version.

### R1. Split the seam into family headers  (cost large, risk medium, Contract, requires D-010)

Keep `kernels.h` as the aggregate include, so no call site changes. Add one
header per family with its own version: `device`, `gemm`, `norm`, `attention`,
`pointwise`, `classifier`, `embedding`, `optim`, `reduce`, `fp8`.

```cpp
// include/nanochat/kernels/attention.h
#define NANOCHAT_ATTENTION_VERSION 1
namespace nanochat::kernels {
void AttentionForward(const AttentionParams& p, const ComputeType* q,
                      const ComputeType* k, const ComputeType* v,
                      ComputeType* out, float* stats);
void AttentionBackward(const AttentionParams& p, /* ... */);
}  // namespace nanochat::kernels
```

Then a backend declares the families it implements, and the build links a
composition fallback for the rest. This keeps the design decision in `DESIGN.md`
section 6: there is still no runtime registry, because the choice is a link
decision.

This item revises `kernels.h:13-19`, which states that the version lives in the
preamble because one backend implements the header as one contract. Per-family
versions are a different contract model. Write decision D-010 in
`spec/decisions.md` before any header changes. The fallback mechanism has one
precedent today, the weak symbol at `src/plan.cc:403`.

Benefit: a new backend starts with one family, not with 30 symbols. The
bring-up cost in `docs/kernels.md` becomes a per-family number.

### R2. (cut) Introduce a value-type `Tensor`  (moved to section 8)

Cut from the plan. The benefit is ergonomic, not structural. A `memory.h` that a
backend includes is a third layering edge, and `DESIGN.md` section 2.1 rule 2
requires the same review for any new edge. A `int device` field in `Tensor` also
contradicts the one-device rule at `src/ops.cc:47` and the mesh runtime's
thread-per-device model. Keep it as question 2 in section 8.

### R3. Validate seam parameters in a debug wrapper  (done in step 2, cost small, risk low)

F4 is real: nothing checks a seam parameter. Do not add a status return to 30
functions, and do not add a host synchronisation. Keep the fatal
`CheckLastError` on the CUDA path, and keep the oracle as the release gate.

As built, `src/kernel_params_check.h` holds one macro,
`NANOCHAT_CHECK_KERNEL_PARAMS(family, condition)`. It aborts with the family and
the condition, and it expands to `((void)0)` under `NDEBUG`.

The calls sit in groups at the family boundaries inside `src/ops.cc`:
gemm, pointwise, the reductions, rms_norm, qk_prep, attention, classifier,
embedding, and value_gate. They check negative extents, pointer presence when a
count is positive, stride and head consistency, and finite alpha, beta, and
epsilon. They never read a device-resident `ScalarArg`.

Two families are absent. `src/optim.cc` calls `AdamWUpdate` and `MuonUpdate`,
not the composition tier, so their checks belong to the optimizer owner.

Only `-c opt` compiles the checks out, so every default `tools/nanochat` run
exercises them.

### R4a. Generate where the row is data, check where it is prose  (cost small, risk low)

`src/capability_names.h` carries a hand-written `consequence` sentence on every
row, and `spec/capability_bits.md` does not supply it. A generator therefore
cannot own that table. Split the work by content.

Emit: `backends/cpu/caps_names.h`, `backends/cuda/caps_names.h`, the six
closed lists in `backends/cpu/caps_test.cc`, and `kBackendNames`. Each of these
holds only a name, a bit, and an order. `backends/cuda/caps_names.cc` holds no
row data, because `AllNamedCapabilityBits()` and its `static_assert` read the
table from the header at compile time.

Keep every emitted spelling byte-identical to HEAD. The generated region may
reflow items and may drop an inline comment that repeats the file's prose,
because the generator owns everything between its markers.

Check: the name, bit, and order of `src/capability_names.h`, and the
`features[]` of every `caps.json`. One Python tool does both jobs in the style
of `tools/gen_matrix.py`, with a write mode and a check mode.

This must run before R14, because the inventory check consumes the same parse.

### R4b. Unify the CPU capability mask  (done in step 1, cost medium, risk medium)

The specified route was wrong, and the change records why. The reference entry
in `backends/cpu/caps.json` has `is_device: false`, so `src/device_profile.cc`
skips it, and `FindDeviceProfile("cpu")` stays null by design.
`tests/device_profile_test.cc` asserts that null. `CapsFromProfile` would also
claim `is_device` and `has_cublas`, which the reference backend must not report.

As built: one additive declaration in the allowed leaf,
`std::uint64_t DeclaredFeatureMask(const std::string&)` in
`include/nanochat/device_profile.h`. The reference backend calls it for "cpu",
and keeps its zero mask when the declaration is unreachable. The `src/plan.cc`
fallback is gone, and `src/device_caps.h` is gone with it.

This was the second frozen-header edit of step 1, so the operator reviewed it.

### R5. Align the capability spelling  (done in step 1, cost small, risk low, Contract)

F9 was a vocabulary drift, not a missing seam operand. `RmsNormForward` is
already the single-kernel replacement, and `src/ops.cc:694` calls it whenever
the bit is on. Three names described one bit: `override_rms_norm` in
`spec/capability_bits.md:80`, `kFeatureOverrideRmsNorm` in the same prose, and
`kFeatureFusedResidualRmsNorm` in `tensor.h`.

As built: the vocabulary spelling is primary for all three rows, so
`kFeatureOverrideRmsNorm`, `kFeatureOverrideQkPrep`, and
`kFeatureOverrideClassifier` replace the `kFeatureFused*` names. Each old name
survives as a one-cycle `constexpr` alias. `kFeatureFusedAttention` keeps its
name, because no vocabulary row names it.

The test-only `cuda_kernels::RmsNormForwardFusedResidual` stays, with a comment
that names its two test callers. The production path is `kernels::RmsNormForward`
plus a `PointwiseForward` for the residual blend.

### R6. Build one flag table and one parser  (cost large, risk medium)

Declare each flag once, with a type, a default, a help line, and the binaries
that accept it.

```cpp
struct FlagSpec { const char* name; FlagType type; const char* def; const char* help;
                  unsigned binaries; };
```

Generate `Usage()` from the table, and parse with one loop. The target is one
declaration per flag branch, one usage text per binary, and one strictness rule.

Do it in two stages. R6a builds the table, the parser, and the generated usage
for `train_main` and `eval_main`, the two largest flag sets. R6b migrates the
other eight binaries. Ten conversions in one diff is not reviewable.

Add the guard first. Before any migration, one test must pin each binary's
`--help` text and a sample parse of every flag it accepts. A changed default or
a changed strictness must then fail the test instead of passing review. The tree
has no such test today.

### R7. Keep one training step  (cost medium, risk medium)

Express accumulation, evaluation cadence, and save cadence as data on
`TrainLoop`, then let `sft_main` and `mesh_train` use it. `Model::TrainStep`
becomes the only loop body. The three current copies cannot then disagree on
gradient accumulation.

Two tests already guard the behaviour and must stay green through the change:
`//tests:sft_step_parity_test` and `//tests:sft_loop_parity_test`. Read them
first, because they state what the current three copies agree on.

Do it in two stages. R7a moves the accumulation, evaluation, and save cadence
onto `TrainLoop` and points `sft_main` at it. R7b points `mesh_train` at the same
schedule and deletes its own copy.

### R8. Move plan and spec work out of the model  (cost small, risk low)

Move the spec lookup, the config agreement check, the plan call, and the plan
log line out of `Model::Create` into one factory in a new `src/model_factory.cc`.
Then `model.cc` holds the graph only, and the two public overloads become thin
wrappers of the same factory.

Both public signatures stay. `model.h:56-63` records why they cannot merge: a
defaulted `const char*` makes `Create(config)` ambiguous. The duplication that
matters is not the two names, it is the 25 repeated lines of config validation
and compute-dtype rejection inside them. The factory removes those lines
without a frozen-header edit, so this item needs no request struct and no
`ModelRequest` type.

### R9. Give inference a narrow read-only view  (done in step 2, cost small, risk low)

Replace the `model_impl.h` include in `generate.cc` with a read-only view of what
inference reads.

As built, `src/inference_view.h` holds a `const Config&`, the `wte`, `lm_head`,
and `smear_gate` pointer, the block array as a pointer plus `num_layers`, and
reader function pointers for the mix scalars and the two rotary tables. The
readers stay function pointers because each read is a device copy at the point
of use, and a snapshot would add one.

The builder, `ViewForInference(Model&)`, takes a non-constant reference, because
growing the rotary table mutates the model. It lives in `src/inference_view.cc`,
the one unit that sees `TrainModel`.

The `Model*` downcasts are gone and so is the `dataloader.h` include. The
`KvCache*` downcast stays, because `KvCacheImpl` is generate.cc's own cache and
not a training type.

### R10. Put the performance model in one home  (done in step 2, cost small, risk low)

Move `CountParams`, `EstimateFlopsPerToken`, and the related functions from
`src/train.cc:158-360` into `src/performance_model.cc`. Keep `mfu.h` as the
declaration. Move the fp8 eligibility policy out of `src/ops.h`, because it is a
policy, not a graph piece.

As built: the counters moved verbatim, and the fp8 policy lives in its own leaf
header `src/fp8_policy.h` that `src/ops.h` includes. The ownership does not move:
`docs/performance-model.md:437` gave the counter struct to the Harness owner
before the change, and the destination file is Harness's, so only the path in
that row changed.

`src/performance_model.cc` still includes `src/ops.h`, because the moved
counters read `kVeGateChannels` and `kSmearChannels` from it. Removing that
include needs those two constants to move first.

Three plan documents still name `src/train.cc` as the home of the counters.
They stay as written, because they record a plan and not the current tree:
`docs/python.md:306`, `:647`, and `:719`.

### R11a. Close the two ABI gaps  (done in step 3, cost small, risk low)

Bind the 3 unbound functions: `nanochat_generate_multi`, `nanochat_rl_step`, and
`nanochat_optim_config_size`. The first needed the 18th `ctypes` structure,
`nanochat_generate_multi_params`. The load path now calls the size query and
refuses a library whose answer differs, so a stale extension fails at import
instead of misreading the optimizer fields in silence.

As built, the signature table holds one row per header declaration under the
header's name, and 68 unique rows match the 68 functions. A new test,
`//python:capi_test`, calls each formerly missing function against the real
library and checks that the guard rejects a wrong size and a library that
predates the query.

### R11b. Generate the C ABI mirrors  (done in step 3b, revised by R20)

One generator reads `capi.h` and does two jobs. It writes the 68-row signature
table and the `static_assert` block, and it checks a hand-written layout table
in `_core.py` against the header. The field copy code in the binding stays
hand-written, because it holds the semantics and not the layout.

As built, `tools/gen_capi_mirrors.py` writes only between its markers, and its
check mode names the file, the structure, and the field on a mismatch in either
direction. A second `--write` changes no byte, and a test pins that.

The `static_assert` block needs two things the header cannot supply: the C++
comparand for each mirrored struct, and the layout table that R20 moved into the
mirror. The generator therefore keeps three authored tables, `DRIFT_ROWS`,
`NO_LAYOUT_MIRROR`, and `ENUM_MIRRORS`, and the mirror keeps a fourth,
`_core._LAYOUT`. A reviewer must read those tables as data, not as derivation.

It also keeps one hand intent. `nanochat_decode` takes `char* out`, and the
mirror keeps `c_void_p`, because `c_char_p` would read the output buffer as a C
string.

Cut: a JSON door for `nanochat_optim_config` and
`nanochat_checkpoint_metadata`. A smaller field count does not justify a
`kNanochatVersion` bump of a frozen header.

### R12. Keep `data.py`, and pin it with a parity gate  (done in step 3, cost small, risk low)

The plan was to replace the `data.py` reader with an ABI call. That is wrong for
this tree. Five callers read the `NCTOKEN1` artifact where no shared library
exists: `python/nanochat_cpp_bridge/reference_data.py`, `tools/dump_fixtures.py`,
`tools/check_tokenizer_parity.py`, `tools/modal/app.py`, and the isolated import
gates. The module is standard-library only on purpose.

So the duplicate stays, and the change adds the gate it lacked. No test compared
the Python reader with the C++ reader on one artifact before, because each side
had its own recorded fixture.

As built, `//python:tokenizer_parity_test` loads one artifact twice and compares
the rank table, the byte lengths, and the token ids for six short strings,
including one non-ASCII string and the empty string. The ABI exposes no merge
list, so the test compares the rank table through the encoder: the bytes of rank
`r` must encode to the single id `[r]`. The test compares rank 0 through the
decoder, because a C string cannot carry a NUL byte.

Keep `checkpoint.py`, which is already an ABI client. Keep `eval_fixture.py`
standard-library only. Keep `plan.py`, which mirrors the reference script.

### R13. Give the two distribution drivers one resume rule  (done in step 2, cost medium, risk medium)

As built: `src/data_cursor.{h,cc}` holds the sidecar path, the writer, and the
reader. Both drivers call it. `src/train.cc` writes byte-identical files, and a
byte-layout test in `//src:data_test` pins the format.

The mesh driver also wraps its factory in `ShardDocumentSourceFactory` at one
rank, so the source reports `position()` even when nothing shards. Three tests in
`//src:mesh_train_test` cover the write, the resume, and the missing-sidecar
case. The resume test failed before the change and passes after it.

Two gaps stay open, and F18 records the first: the multi-process counter. The
second is the packed `NCSFT001` row path, which has no cursor and replays by
design.

### R14. Add one seam inventory check  (cost small, risk low)

Add one test with two halves. The first half fails when `spec/block_kinds.md`
names a kind that `src/model_spec.cc` does not accept. It depends on nothing, so
it can go first. The second half fails when a `caps.json` `primitives[]` entry
has no seam symbol. It consumes the parse from R4a, so it must follow R4a.

Then F17 and the F8 duplication fail in the default CPU test leg, not at review
time.

### R15. Remove the parameter that F18 left dead  (done in step 3b, cost small, risk low)

`ShardDocumentSourceFactory` took `start_on_creation`, and no caller passed a
nonzero value after F18. The change deleted the parameter, its header comment,
its definition, and all four call sites: two in production and two in tests.
`start` stays, because the resume still uses it.

The wrapper now carries a one-bit "first creation" flag instead of a counter. It
still applies `start` to the source the constructor makes, and a later `Reset`
still opens the next epoch at 0, so behaviour did not change.

### R16. Write the capability chain down once  (cost small, risk low, comment-only Contract)

`Caps` mixes three kinds of fact, and the tree never says which is which.

| Kind | Examples | Where it comes from |
|---|---|---|
| Detected | `compute_major`, the four precision bits | the vendor API; `PrecisionFeatureBits` overrides the declaration (`backends/cuda/device.cu:44-77`) |
| Declared | `features[]`, the profile facts | `backends/*/caps.json` |
| Required | `requires` | `models/*/spec.json` |

The rule to write down: one fact has one layer and one spelling. A detected
fact overrides a declared one. Allow a derived spelling only at an ABI
boundary, and it must read the layer below it.

Put that rule in `spec/capability_bits.md` as one short section. Add a comment
block on `Caps` that names the three kinds and states that `Caps` is the agreed
set after detection overrides declaration. The header edit is comment-only, and
it still needs the architect.

This is the cheapest item in the review, and it removes the confusion that the
current seven spellings of "does this device do fp8" produce.

### R17. Collapse the derived boolean layer  (cost medium, risk medium, Contract)

`supports_fp8` is a lossy copy of two bits. `Supports(DType)` is a copy of that
copy. So one fact has four spellings: the bits, the boolean, the dtype query,
and the string in `features[]`.

Make `Supports(DType)` read `feature_mask` directly, then delete
`supports_fp32`, `supports_bf16`, and `supports_fp8` unless a C consumer reads
them. Check `src/hw_gate.cc:444-460`, `backends/cuda/device.cu:202-205`,
`backends/cpu/kernels.cc`, and `tests/device_profile_test.cc` first.

The bits stay. Each one maps to a seam entry, which is the rule that R5 and R14
now enforce, so the vocabulary keeps them.

Do this last in stage 6, because it edits a frozen struct and it needs the rule
from R16 to justify the removal.

### R18. Produce the build offer from the build  (cost medium, risk medium)

Seven bits describe the binary, not the device: `primitive_tier_complete`, the
`stateful_*` pair, and the three `override_*` bits that have a constant.

Today `backends/*/caps.json` declares them per device, so a declaration can
claim an override that the binary never registered. Emit them from
`backends/cuda/fused_overrides.cc` and the seam instead, and make the check fail
when a declaration claims a bit the binary does not register.

This is the part of the mask that a build flag would express. Emitting it from
the build is better than a flag, because a test can check a generated bit, and
a flag escapes the check.

### R19. Delete the rows that cannot differ  (cost small, risk low)

Measured at HEAD: `utilities[6]`, `primitives[11]`, `pointwise_codes[9]`,
`reduce_ops[4]`, and `attention` are byte-identical on all six devices. That is
30 rows per device, about 180 lines of `backends/*/caps.json`.

The rows carry no information, because the linker already requires every seam
symbol. Only the 17 device facts and `features[]` vary.

Delete the rows, and state the rule in one comment: a conforming backend
implements the whole seam. Keep them only if R1 lands and a backend may be
partial. Until then, they are a placeholder, and the 30-symbol tier table in
`tools/contract_vocabulary.py` polices a tautology.

### R20. Move the ctypes layout into the mirror  (done, cost small, risk low)

The generator used to emit the 18 `ctypes` structures into `_core.py`. They are
now one `_LAYOUT` table in the same module, keyed by class and holding the real
`ctypes` types, plus one loop that sets `_fields_`. `include/nanochat/capi.h` stays the
authority. The generator's check mode compares the table to the header in both
directions.

The change does not shrink the tree. Generated lines fall from 574 to 417,
because 157 lines of field lists and 18 markers leave. Hand lines rise by 159 in
the mirror, and the checker grows by 69 to cover the new comparison. Net, the
pair grows by about 100 lines.

What it buys: the layout appears once, as real types, in the file a reader
already opens, and no writer can reformat it. What it costs: `--write` can no
longer repair the table, so a header change and a table change are two edits
with a check between them.

Two limits, recorded so a reader does not trust the check further than it goes.
On LP64, `ctypes` spells `c_int32` as `c_int` and `c_int64` as `c_long`. So the
comparison cannot see a header change from `int` to `int32_t`. The width is what
the ABI means, so this is acceptable, but the check is not a byte-level identity
check. The install loop also leaves `_class` and `_fields` as module globals.

The check compiles the module's bytes instead of importing it. An earlier
version imported `_core`. A same-size edit inside one second then left a stale
`__pycache__` entry answering for the table. Any checker that reads a Python
module needs the same care.

### Step 1 as built, changes 1 to 4

`tools/contract_vocabulary.py` does both jobs. `--write` regenerates the region
between the `BEGIN GENERATED` and `END GENERATED` markers in the six files
above. `--check` compares the tables that hold prose and reports the row on a
mismatch. `tools/contract_vocabulary_test.py` is a T0 CPU test with 10 cases.

The change departs from the plan in three places, and each departure records a
fact the plan did not know.

1. `backends/cuda/caps_names.cc` holds only lookup functions, so it gets no
generated region.
2. `src/model_spec.cc` accepts `none` for the position field, and
   `spec/block_kinds.md` section 2 has no row for it. The alias table records it
   as the absence value with a comment, so the check passes and the gap stays
   visible.
3. The seam tier table classifies all 30 symbols, because a new seam symbol
   must force a classification. A symbol with no tier row now fails the check.
   The table also records `relu2_mlp` against `relu_square`, a diverging pair
   that `src/model_spec.cc:213-214` does not list.

Changes 3 and 4 landed next, and each one adds a fact the plan did not have.
The rename made the vocabulary spelling primary for three capability constants,
and kept the old names as one-cycle aliases. The reference backend now reports
its nine declared bits through one new leaf accessor, and the planner fallback
and `src/device_caps.h` are gone.

Step 1 changed two frozen headers, and the operator reviewed both:
`include/nanochat/tensor.h` for the constant spelling, and
`include/nanochat/device_profile.h` for the additive accessor.

It also leaves two failures that are not ours. `//tests:cuda_sim_preload_test`
and `//tests:numerics_trace_test` fail with and without these changes, and a
stash-and-rerun proved the same failure sets both times.

The change touches five build files outside `tools/`, and every edit is
additive: `include/nanochat/BUILD.bazel`, `src/BUILD.bazel`,
`backends/cpu/BUILD.bazel` if needed, a new `spec/BUILD.bazel` with
`exports_files` only, and `tools/BUILD.bazel`. A reviewer needs that list,
because `AGENTS.md` section 1 gives each file a different owner.

### Step 2 as built

R8 landed as a factory in `src/model_factory.{h,cc}`, inside the existing
`//src:model` target. A separate library would be a cycle, because the factory
constructs `TrainModel` and `model.cc` calls it. The factory takes four
parameters, because the private seam has three states and the brief's three
parameters carried two:

```cpp
enum class SpecSource { kConfigOnly, kByName };
std::unique_ptr<Model> CreateTrainModel(const Config& config, SpecSource source,
                                        const char* spec_name,
                                        std::string* error);
```

`Model::Create(config)` reads no spec and logs nothing. `Model::Create(config,
nullptr)` still reads the built-in default spec and logs the plan, so both
documented behaviours survive.

R10 landed as described above. The worker also corrected four rows of
`docs/performance-model.md`, which `AGENTS.md` gives to the Harness owner.

R9 and R3 landed next as their own change set. `src/inference_view.h` plus
`src/inference_view.cc` replace the `model_impl.h` dependency, and
`src/kernel_params_check.h` holds the debug-only parameter checks that
`src/ops.cc` calls.

R13 landed too. `src/data_cursor.{h,cc}` now holds the only sidecar
implementation, and the mesh driver reads and writes it.

R13 also found a real defect that no test covered: the multi-process resume
counted the epoch twice and dropped the restored buffer. Step 3 fixed it as
F18. The mechanism was narrower than first described. The resume offset already
landed on the right creation; only the buffer was lost. The fix mirrors the mesh
driver: the offset goes to the constructor's creation, and the `Reset()` loop
goes away.

F18 also leaves a follow-up. `start_on_creation` now has no caller with a
nonzero value, so the parameter is dead weight. It is R15.

---

## 6. Recommended order

| Step | Work | Why in this position |
|---|---|---|
| 1 | done: R14, R4a, R5, R4b | Two frozen-header edits, both reviewed; removes the two principal sources of silent drift |
| 2 | done: R8, R10, R9, R3, R13 | Small local edits, each removes one blurred boundary; R13 started with a test |
| 3 | done: R11a, R12, F18 | Python and ABI gaps, then the resume defect that R13 found |
| 3b | done: R11b, R15 | The ABI generator needs the 3 bound functions as its first check; R15 removes the parameter F18 left dead |
| 4 | R6a, R6b, R7a, R7b | The two large wins, and the two largest diffs; R6 starts with the guard test, and each stage is one reviewable change |
| 5 | R1 (requires D-010) | The only item that rewrites the contract model |
| 6 | R19, R16, R18, R17 | Stage 6 is the capability cleanup: cheap data and prose first, then the build offer, then the derivation collapse. R17 goes last because it edits a frozen struct |
| 6b | done: R20 | The ctypes layout table; it edits the Python mirror only |
| deferred | R2 | Needs a decision record first |

One writer runs these in order. The items cannot run in parallel. R4a and R5
share the generated tables and `caps.json`. R4b and R14 share `src/plan.cc` and
`src/device_profile.cc`. R14 also touches `tests/` and the backend test targets.

R1 comes last on purpose. It changes a frozen header's contract model, and it
becomes easier after R14 proves which families exist.

Stage 6 sits after R1 in cost, not in dependency. R19 and R16 can run at any
time. R18 is easier after R1, because a partial backend would make the offer
rows meaningful again. R17 needs R16 to justify it.

---

## 7. What must not change

These decisions are load-bearing, and a simplification must not trade them
away.

- No autograd engine and no graph IR (`DESIGN.md` section 5 and 6).
- The backend is a link decision, not a runtime registry.
- One binary runs one backend.
- The closed-field rule for `model_spec.h` and the closed `PointwiseOp` and
  `ReduceOp` enums.
- `BuildPlan` stays pure.
- Every executable gates on `RequireSandboxOrDie`.
- The oracle fixtures stay committed, so a GPU test needs no PyTorch.
- The primitive tier stays a contract promise: `primitive_tier_complete` in
  `tensor.h:255` and the 16-entry bring-up in `docs/kernels.md:92`.
- The four contract numbers in `spec/versions.md` stay independent.
- A declared capability never overrides a detected one. The reverse direction is
  the only legal one, and it is what `PrecisionFeatureBits` does.

---

## 8. Open questions

1. R1: should the seam keep one contract and one version, or move to per-family
   versions? The oracle records this as decision D-010. A note in this document
   is not enough, because the answer changes `kernels.h:13-19`.
2. R2: is a value-type `Tensor` worth a third layering edge? It needs the same
   review as the two existing leaves. A `device` field would also contradict the
   one-device rule at `src/ops.cc:47`.
3. Two helpers now carry the name `DeclaredFeatureMask`: the leaf API for a
   declaration name, and an internal one in `backends/cuda/device.cu` for a
   vendor name. They live in different namespaces. Decide whether the pair
   needs a rename before a third caller appears.
4. R8: both public `Model::Create` overloads stay, and the factory removes the
   duplicated body. The open part is whether the two overloads should merge
   into a request struct later, which is a frozen-header edit for no behaviour
   gain.
5. Is the C ABI the right long-term door for Python, or should that surface move
   to a thin server process? The `rl.py` path already uses a pipe.
6. Does the mesh path need the multi-process path at all, once the in-process
   collective has the same resume behaviour?
