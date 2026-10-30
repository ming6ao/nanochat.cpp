# Checkpoint metadata and minimal examples: plan

How to add versioned run metadata to the checkpoint container, and how to add
the two smallest possible examples of the public API. The two parts are
independent. The roundtrip example uses the container from the metadata work,
so it lands after stage A.3.

Status: proposal. Not started. Owner: Architect.

Eight open decisions block stage A.1; see section 1.1. Stage A.3 also touches
`src/data.cc` (Harness). Stage A.5 touches `bindings/**` and `python/**`
(Python surface). Stage A.4 touches `tests/**` (Oracle).

Coordinate those three handoffs before the stage starts.

Background: [checkpoint-format.md](checkpoint-format.md) specifies the
container as it stands. [model.md](model.md) specifies the model API that the
examples use. [build.md](build.md) holds the file layout and the Bazel
conventions. [testing.md](testing.md) holds the test tiers.

The design borrows two mechanisms from `gemma.cpp`. The first is the field
visitor in `io/fields.h`, which gives forward and backward compatible metadata.
The second is the two tiny examples in `examples/`, which keep the mainline
auditable and prove that the public API stays usable.

---

## 0. Scope

| In scope | Out of scope |
|---|---|
| A field-visitor encoder in `include/nanochat/fields.h` | Changing the tensor record byte layout |
| A metadata section in the container (O5 decides the magic) | Compression, int8 payloads, shard metadata |
| A reader for both container versions, with strict tests | Runtime schema discovery in the kernels |
| Two examples, built and tested by Bazel | A new `tools/nanochat` command |
| Doc, C API, and Python updates | A rewrite of the container |

---

## 1. Metadata: design decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Does the visitor encode tensor payloads? | No. Keep the record stream. The records are already self-describing. |
| D2 | What does the visitor cover? | Run metadata only: stage, step, world size, precision, config, tokenizer id, revision, timestamp. |
| D3 | How does a reader tell versions apart? | A version bump to 1. The reader keeps the version 0 path. O5 decides whether the magic also changes. |
| D4 | Where does the metadata section sit? | Directly after the header, before the parameter count. |
| D5 | What happens on unknown data? | Trailing words mean a newer file. Accept, count the words, warn once. |
| D6 | What happens on missing fields? | Member initializers fill the gap. Accept and count them. |
| D7 | What happens on a bad value? | Hard error. No partial load. Reuse the existing `fail(...)` style. |

D1 is the load-bearing decision. `gemma.cpp` uses the same split. Its
`ModelConfig` is an `IFields`, and its tensor data lives in the blob store. The
container in this tree already behaves like the blob store, so the visitor
stays away from it.

### 1.1 Open decisions

These eight questions blocked stage A.1. Each answer below is the one the
implementers took, and the code that carries it is named, so the decisions are
closed. The prefix `O` means open decision.

| ID | Question | Answer |
|---|---|---|
| O1 | How does `Checkpointer::SaveModel` learn the run facts? | A `CheckpointSaveOptions` struct in `src/train.h`, carrying `stage`, `step`, `world_size`, `precision`, `tokenizer_id`, `git_revision`, and `notes`. The config is not in it: `SaveModel` always records `model.config()`, and a second copy could drift. The three existing overloads are thin wrappers over `SaveModel(model, optimizer, options, path, error)`, so no call site changed. |
| O2 | Is `Precision` a new enum? | No. The block records `ComputePrecision` (`include/nanochat/mfu.h`), with its `EnumValid` beside it in `src/checkpoint_metadata.cc`. One precision enum stays the source of truth, and the recorded value is informational: it never changes a load decision. |
| O3 | What are the `Stage` values? | `Stage::kBase`, `Stage::kSft`, and `Stage::kRl`, stored as 0, 1, and 2, to match the notebook vocabulary. There is no `kPretrain`. The mapping is in [checkpoint-format.md](checkpoint-format.md) section 5.3, `src/sft_main.cc` records `kSft`, and `RlWorker::SaveCheckpoint` records `kRl`. |
| O4 | Is the `ConfigFields` mirror the right shape? | No mirror. One host-only `VisitConfig(Config&, IFieldsVisitor&)` in `src/checkpoint_metadata.cc`, visited through the `ConfigBlock` wrapper that carries the block's size word. No `Config` field is duplicated, and the drift test is unnecessary. |
| O5 | New magic `NCHKPT02`, or version 1 only? | Version 1 only, with the existing magic. The table in section 2.1 shows the new-magic option; the answer is the recommended one. `DetectFormat` reads the magic and then the version word, an unknown version word is the hard error `unsupported checkpoint version`, and the two magic/version mismatch cases of stage A.4 collapse to a file that changed between the two reads. |
| O6 | How does a 64-bit `step` fit the 32-bit word stream? | Two words, the low 32 bits first and then bits 32-63. The order is part of the contract and is documented in the word-stream comment of `include/nanochat/fields.h` and in [checkpoint-format.md](checkpoint-format.md) section 5.3. The full 64-bit range is kept, because capping the step loses a long run that a resume schedule needs. |
| O7 | What does the Python surface return? | The two-element tuple of `parse_checkpoint` stays, and `read_metadata(path)` is added in `python/nanochat_cpp/checkpoint.py`. It returns one mapping keyed by `METADATA_KEYS`, and a version 0 file reports the defaults, so no caller breaks. |
| O8 | How does the release check old-code rejection? | One command against a fixed parent commit, as a release gate. It is not an automated test in this tree, and [checkpoint-format.md](checkpoint-format.md) section 10 records it. |

Section 2.1 and section 3, and stage A.6, show the recommendation from O1 to
O5. If an answer differs, change those sections in the same commit. Every
answer above is the recommendation, except the two shapes section 2.1 and
section 3 spell out: O5 keeps the existing magic, so section 2.1's `NCHKPT02`
row is the option the answer did not take, and O4 removes the mirror, so
section 3's `ConfigFields` is the shape the answer did not take.

## 2. Container layout

### 2.1 Version 1 layout

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `NCHKPT02` |
| 8 | 4 | version, `uint32`, value 1 |
| 12 | 4 | metadata word count, `uint32` |
| 16 | `4 * count` | metadata words, `uint32` each |
| after the metadata | 4 | parameter count, `uint32` |
| then | variable | parameter records, identical to version 0 section 6 |
| then | 4 | optimizer count, `uint32` |
| then | variable | optimizer records, identical to version 0 |

The record framing stays byte identical to version 0. This protects
`ReadRecord`, the Python decoder, and the `nanochat_checkpoint_*` surface at
the same time.

O5 decides the magic. The table shows the new-magic option. The recommended
option keeps `NCHKPT01` and changes only the `version` word to 1.

### 2.2 Routing

O5 decides whether `IsCheckpointContainer` becomes a two-magic test. Either
answer needs the same additions. Add a sibling enum `CheckpointFormat` with the
values `kLegacyModel`, `kContainerV0`, and `kContainerV1`, and one function
`DetectFormat(path)`. `Checkpointer::LoadModel` then routes on that value
instead of on a bool.

A new magic must not drift from the `version` field across releases. Rule for
the document: the magic names the layout family, and `version` must match it. A
mismatched pair is a hard error. This rule is moot when O5 keeps one magic.

## 3. Metadata schema

```cpp
// include/nanochat/checkpoint_metadata.h  (host-only, vendor-free)
// O4: prefer a free VisitConfig(Config&, IFieldsVisitor&) in src/ over this
// mirror struct. The mirror is shown here to make the field list explicit.
struct ConfigFields : public IFields {
  const char* Name() const override { return "Config"; }
  void VisitFields(IFieldsVisitor& v) override {
    v(num_layers); v(num_heads); v(num_kv_heads); v(hidden_dim);
    v(seq_len); v(vocab_size); v(padded_vocab_size); v(rope_base);
    v(window_pattern); v(value_embedding);   // append new fields here
  }
  std::uint32_t num_layers = 12;
  // ... defaults equal to nanochat::Config ...
};

struct CheckpointMetadata : public IFields {
  const char* Name() const override { return "CheckpointMetadata"; }
  void VisitFields(IFieldsVisitor& v) override {
    v(schema); v(stage); v(step); v(world_size); v(precision);
    v(config); v(tokenizer_id); v(git_revision); v(created_utc); v(notes);
  }
  std::uint32_t schema = 1;      // the metadata schema, not the file magic
  Stage stage = Stage::kPretrain;
  std::uint64_t step = 0;
  std::uint32_t world_size = 1;
  Precision precision = Precision::kFp32;
  ConfigFields config;
  std::string tokenizer_id;      // artifact name or hash
  std::string git_revision;
  std::string created_utc;       // ISO-8601, informational only
  std::vector<std::string> notes;
};
```

Rules to record in the header comment and in the document:

1. The field order never changes. A new field goes last.
2. An enum value never changes. A new state becomes a new field.
3. A string caps at 4096 words. A vector caps at 1024 entries. The whole
   metadata block caps at 64 KiB. A larger value is a hard error, in the same
   way as the existing `kMaxAttributes = 64`.
4. `created_utc` and `git_revision` are informational. No load decision may
   depend on them.

O2 and O3 decide the `precision` and `stage` types. The struct shows the names
as first proposed. `Stage::kPretrain` becomes `Stage::kBase` under O3, and
`Precision` becomes `ComputePrecision` under O2.

## 4. Metadata: public API additions

| Layer | Addition |
|---|---|
| `include/nanochat/data.h` | `CheckpointMetadata metadata() const`, `void set_metadata(CheckpointMetadata)` |
| `include/nanochat/data.h` | `CheckpointFormat DetectFormat(const std::string& path)` |
| `include/nanochat/capi.h` | `nanochat_checkpoint_metadata_read`, plus getters for the stage, the step, the world size, and the config |
| `src/train.h` | `Checkpointer::SaveModel` fills the metadata from the run, through `CheckpointSaveOptions` (O1) |
| Python | `checkpoint.parse_checkpoint` returns the metadata with the records |

Keep the C surface flat. One `const char*` JSON accessor for the nested config
beats twelve scalar getters. The Python module calls the C library instead of
parsing bytes, so Python gains the change once the C getter exists.

## 5. Metadata: implementation stages

Each stage ends with a green `tools/nanochat test` and one commit.

### Stage A.1: Port the field visitor

Files added:

- `include/nanochat/fields.h`
- `src/fields.cc`
- `src/fields_test.cc`
- `src/BUILD.bazel`: `cc_library(name = "fields")`,
  `cc_test(name = "fields_test")`, and the test added to
  `test_suite(name = "all")`
- [build.md](build.md): the file layout, plus one line that states `fields.h`
  is host-only and vendor-free

Port from `gemma.cpp/io/fields.h` and `io/fields.cc`, with these substitutions:

| `gemma.cpp` | `nanochat.cpp` |
|---|---|
| `hwy::Span<const uint32_t>` | `std::span<const std::uint32_t>` |
| `hwy::BitCastScalar` | `std::bit_cast` |
| `HWY_UNLIKELY`, `HWY_ASSERT` | `[[unlikely]]` and a plain return |
| `hwy::DivCeil` | a local `DivCeil` |
| `HWY_TEST_MAIN` | the local `Fail()` and `main()` convention |

Keep the names `IFields`, `IFieldsVisitor`, `SerializedSpan`, `SkipField`,
`IsReading`, `NotifyInvalid`, and `ReadResult{pos, missing_fields, extra_u32}`.
They are the contract, and a reviewer can compare the two files side by side.

Confirm the 64-bit encoding first (O6). The port keeps a `uint32` word stream,
and section 3 declares `std::uint64_t step`.

Gate: the ported test suite passes. Port every case: round trip, new code with
old data, old code with new data, invalid float, invalid string, multiple
write, vector cap, and enum validity.

### Stage A.2: Freeze a version 0 fixture

Generate a small version 0 file with the current tree before anything changes.
Add a `checkpoint` command to `tools/dump_fixtures.py`. That file is
Oracle-owned and holds the existing fixture convention. The command calls
`python/nanochat_cpp/checkpoint.write_checkpoint`, so the bytes come from the
current C writer and the script holds no copy of the layout. Commit the bytes
as `tests/data/checkpoint_v0.ckpt`. The header comment states the record names
and the expected payload checksum.

Gate: a new test `//tests:checkpoint_compat_test` reads the fixture with the
current reader and asserts the exact names, dtypes, shapes, and payloads. This
test is the permanent guard for the artifacts that already sit on the Hub.

### Stage A.3: Add the metadata and the writer

Files changed:

- `include/nanochat/checkpoint_metadata.h`, new
- `src/data.cc`: `Save` writes version 1 and the metadata section, and `Load`
  dispatches on the magic
- `include/nanochat/data.h`: the metadata accessors and `CheckpointFormat`
- [checkpoint-format.md](checkpoint-format.md): the version 1 layout, the
  compatibility rules, and a version history table

Write the metadata before the parameter count. A parameter-only read then still
stops cheaply, because the optimizer section stays last.

Gate: a round-trip test writes metadata, reads it back, and compares every
field. Add the case of empty metadata on the version 0 writer path.

### Stage A.4: Compatibility tests

New file `tests/checkpoint_compat_test.cc`, with these cases:

| Case | Expectation |
|---|---|
| Read `tests/data/checkpoint_v0.ckpt` | The records match, and the metadata holds defaults |
| Write version 1, read version 1 | All fields are equal |
| Write version 1 with 3 extra trailing words | The read succeeds, `extra_u32 == 3`, the records are unchanged |
| Write metadata that lacks the last 2 fields | The read succeeds, `missing_fields == 2`, defaults apply |
| Truncate the metadata count | A hard error, and `Load` returns false |
| Put the metadata word count above the cap | A hard error |
| Magic `NCHKPT02` with version 0 | A hard error. Only when O5 selects the new magic |
| `NCHKPT01` with version 1 | A hard error. Only when O5 selects the new magic |
| Version 1 header, then no metadata section | A hard error |
| Version 0 header, then metadata words | A hard error |

The third and fourth cases need two struct shapes. Follow the `OldFields` and
`NewFields` trick in `gemma.cpp/io/fields_test.cc`: two structs over the same
word stream, one of them truncated on purpose.

### Stage A.5: C API and Python

Files changed: `include/nanochat/capi.h`, `bindings/nanochat_capi.cc`,
`bindings/nanochat_capi_test.cc`, `python/nanochat_cpp/checkpoint.py`, and
`python/tests/checkpoint_test.py`.

Correct the capi comment that says "The file must be a version 0 container".

Gate: a Python test loads the version 0 fixture and a new version 1 file, and
compares the metadata to the C values.

### Stage A.6: Fill the metadata on save, and sweep the documents

`Checkpointer::SaveModel` fills `step`, `precision`, `world_size`, and the
config through the `CheckpointSaveOptions` struct of O1. The config comes from
`Model::config()`, and `Config::Validate` runs before the write. A mismatch
between the file config and the runtime config then fails at load with a clear
message, not at the first GEMM.

Then sweep the documents. A recursive search finds 72 matches in 27 files,
excluding this plan and caches. Rule for the sweep: a code constant keeps the
magic, and prose and comments say "the checkpoint container". Only
[checkpoint-format.md](checkpoint-format.md) names a magic in prose. Match the
whole word, so a longer identifier does not trip the gate.

Files to touch:

- Documents: [model.md](model.md), [eval.md](eval.md),
  [post-training.md](post-training.md), [parity.md](parity.md),
  [python.md](python.md), [rl-notebook.md](rl-notebook.md),
  [self-evolution.md](self-evolution.md), [README.md](README.md),
  [checkpoint-format.md](checkpoint-format.md), and `docs/README.md`.
- Notebooks: `notebooks/README.md`, `notebooks/hub_checkpoint.py`,
  `notebooks/hub_checkpoint_test.py`, and `notebooks/target_test.py`.
- Python surface and bindings: `python/nanochat_cpp/checkpoint.py`,
  `python/nanochat_cpp/chat.py`, `python/nanochat_cpp/rl.py`,
  `python/tests/checkpoint_test.py`, `tools/convert_checkpoint.py`,
  `bindings/nanochat_capi.cc`, and `include/nanochat/capi.h`. Stage A.5 owns
  these lines, so this stage only checks them.
- C++: `include/nanochat/data.h`, `src/data.cc`, `src/harness_test.cc`,
  `src/optimizer_state_test.cc`, `src/rl_worker.h`, and
  `src/rl_worker_main.cc`.

Do not build a migration tool now. The reader accepts version 0, and the next
`Save` upgrades the file. Add tooling only when a set of Hub artifacts must gain
metadata without a training run.

---

## 6. Examples: design decisions

| ID | Decision | Recommendation |
|---|---|---|
| E1 | How many examples? | Two C++ examples. A Python mirror comes later. |
| E2 | Which two? | `hello_world` for inference, and `checkpoint_roundtrip` for the container. |
| E3 | Do they need model files? | No. Both use a tiny config and a fixed seed. |
| E4 | Do they become tests? | Yes. One source file, built as `cc_binary` and `cc_test`. |
| E5 | Do they call the sandbox guard? | Yes, `RequireSandboxOrDie("example")`, like every entry point. |
| E6 | Where do they live? | A new top-level `examples/` directory, one package per example. |

`hello_world` uses only public headers: `nanochat/model.h`,
`nanochat/config.h`, and `nanochat/sandbox.h`. The whole call sequence is
`RequireSandboxOrDie("example")`, `Config`, `Model::Create`, `InitWeights`,
`GenerateParams`, `GenerateBatch`, and then print the ids. The task label is
the action, as in every other entry point. Target length: 70 lines, including
the header comment.

The tiny config must pass `Config::Validate` (`include/nanochat/config.h`):

- `hidden_dim` is 24 or more.
- `head_dim` must stay even.
- `num_kv_heads` divides `num_heads` without a remainder.
- `padded_vocab_size` is `vocab_size` or more.

`checkpoint_roundtrip` uses `nanochat/data.h` and `nanochat/tensor.h`. It builds
two records, saves, loads, and prints the metadata and the record table. Target
length: 90 lines. The example doubles as executable documentation for the
metadata work.

## 7. Examples: implementation stages

### Stage B.1: `examples/hello_world`

Files: `examples/hello_world/hello_world.cc` and
`examples/hello_world/BUILD.bazel`, with `cc_binary`, `cc_test`, and
`test_suite(name = "all")`.

Add `//examples/hello_world:all` and `//examples/checkpoint_roundtrip:all` to
`test_suite(name = "all_tests")` in the root `BUILD.bazel`. That is the
documented rule for a new package.

The header comment states three things: what the example proves, the exact
command to run it, and the one line a reader changes to try a larger config.

### Stage B.2: `examples/checkpoint_roundtrip`

The same shape. It lands after stage A.3, so it can also print the metadata
block. If the metadata work slips, ship it against version 0 and extend it
later.

### Stage B.3: Wiring and documents

- `docs/examples.md`, new: what each example shows, the run command, and the
  line budget
- `README.md`: one short section that points to `docs/examples.md`
- [build.md](build.md): add `examples/` to the file layout
- `AGENTS.md`: add the ownership entry for `examples/`

Run command:
`tools/nanochat run t0-cpu -- bazel-bin/examples/hello_world/hello_world`.
`t0-cpu` is the direct profile, so there is no queue submission. Do not add a
`tools/nanochat example` command. `python/tests/contract_test.py` asserts that
the command table and the handler table agree, so a new command adds churn for
no gain.

### Stage B.4: Guard rails

Two cheap checks keep the examples honest.

1. One rule in `CONTRIBUTING.md`: an example includes only `include/nanochat/**`
   and the `src/*.h` headers that carry a stability note. No vendor header. No
   `backends/` include.
2. One test asserts that the example prints a fixed number of rows for a fixed
   seed. The text output stays unpinned, because token ids may change with a
   legitimate kernel change. The shape and the mask stay pinned.

---

## 8. Test matrix

| Test | Tier | Runs in |
|---|---|---|
| `//src:fields_test` | T0 CPU | `tools/nanochat test` |
| `//src:data_test`, extended | T0 CPU | same |
| `//tests:checkpoint_compat_test` | T0 CPU | same |
| `//bindings:nanochat_capi_test`, extended | T0 CPU | same |
| `python/tests/checkpoint_test.py` | T0 CPU | same |
| `//examples/hello_world:hello_world_test` | T0 CPU | same |
| `//examples/checkpoint_roundtrip:roundtrip_test` | T0 CPU | same |
| The fixture read under `--config=bf16` and `--config=fp8` | T0 CPU | same, three builds |

The last row is the precision gate. The metadata must parse the same in every
precision build, and the recorded `precision` value must never change a load
decision. Add that assertion to the compatibility test.

## 9. Risks

| Risk | Mitigation |
|---|---|
| A layout change breaks Hub artifacts | A committed version 0 fixture, a reader for both versions, and the two counters |
| Silent loss of a new field | A hard error on a bad value or a short span, and counters in the load result |
| The metadata grows into a second config file | A 64 KiB cap, a 4096-word string cap, and a 1024-entry vector cap |
| Two sources of truth for the config | `ConfigFields` mirrors `nanochat::Config`, and a test compares the two field by field |
| The visitor leaks into the kernels | `fields.h` is host-only, has no vendor include, and stays out of `kernels.h` |
| The examples rot | Each example is a `cc_test` in `//:all_tests` |
| The document sweep stops halfway | A grep gate before the stage commit: no prose file outside `checkpoint-format.md` names a magic, a code constant keeps it, and the match uses a whole word |

## 10. Order and effort

| Step | Effort | Blocks |
|---|---|---|
| O.0 resolve the eight open decisions | S, a few hours | A.1 |
| A.1 field visitor | S, about one day | everything in the metadata work |
| A.2 version 0 fixture | S, half a day | nothing |
| A.3 metadata and writer | M, about one day | A.4, A.5, B.2 |
| A.4 compatibility tests | M, about one day | the release |
| A.5 C API and Python | S, half a day | the release |
| A.6 save-side fill and document sweep | M, about one day | the release |
| B.1 `hello_world` | S, half a day | nothing |
| B.2 `checkpoint_roundtrip` | S, half a day | A.3 |
| B.3 documents and wiring | S, half a day | nothing |

Do B.1 in parallel with A.1. It has no dependency on the metadata work, and it
gives the reviewer a readable artifact early.

## 11. Definition of Done

1. The new code reads version 0 and version 1 files, and the committed version
   0 fixture passes.
2. The layout gate of O5 makes an old binary reject a new file with a clear
   error, not a misparse. Check this as a release gate at the parent commit
   (O8).
3. `fields_test` covers the round trip, new code with old data, and old code
   with new data.
4. `Checkpoint::Save` writes the metadata from the run, and `Load` exposes it.
5. The C API and Python expose the metadata, and both tests pass.
6. [checkpoint-format.md](checkpoint-format.md) states the version 1 layout,
   the compatibility rules, and the version history.
7. Both examples build, run under `tools/nanochat run t0-cpu -- ...`, and pass
   as tests in `//:all_tests`.
8. `docs/examples.md` exists, and `README.md` links to it.
9. `tools/nanochat check` passes in the default fp32 build, and
   `tools/nanochat test --config=bf16` and `tools/nanochat test --config=fp8`
   pass.
10. No new vendor include sits outside `backends/`, and `kernels.h` is
    unchanged.
