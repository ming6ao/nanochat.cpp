# Native parquet input and on-the-fly tokenization

This document is the plan of record for two changes:

1. Replace the DuckDB parquet dependency with a small native C++ reader.
2. Remove the `corpus -> NANO shard` materialization step. The training and
   evaluation loops tokenize during the run, like the reference.

The tokenizer work is in [tokenizer.md](tokenizer.md). The data format is in
[model.md](model.md). The parity gaps are in [parity.md](parity.md).

## 1. Objective

Today the data path has three stages:

```
parquet -> tok_shard_main -> NANO shard -> train_main
```

Two problems follow from the stage two and stage three split:

- The training runtime reads only a pre-tokenized shard, so a pre-pass is
  mandatory.
- The parquet reader needs DuckDB, a 78 MB static library. `DESIGN.md`
  section 2.1 carries a special exception for it.

The target data path is:

```
parquet -> native reader -> on-the-fly tokenizer -> training rows
```

Two results follow:

- No shard pre-pass.
- No DuckDB. Only ZSTD remains as a host dependency.

## 2. Why a native reader is feasible

One writer produced every dataset file: `parquet-cpp-arrow version 21.0.0`.
A scan of 33 ClimbMix shards and 2732 column chunks shows one fixed shape:

| Property | Value |
|---|---|
| Columns | one OPTIONAL `BYTE_ARRAY`, `text: string` |
| Codec | `ZSTD` |
| Value encoding | `PLAIN` |
| Level encoding | RLE definition levels; no repetition levels |
| Page type | `DATA_PAGE` v1 only |
| Dictionary page | none |
| Pages per chunk | one |
| Row group | 1024 rows |
| Statistics | unused |

The reader supports this subset and rejects everything else with an error.
The supported subset is small. The container is not: it needs a Thrift
compact footer, page headers, RLE bit-packed levels, and ZSTD frames.

## 3. Dependency decision

- ZSTD comes from the Bazel Central Registry: `bazel_dep(name = "zstd",
  version = "1.5.7")`, target `@zstd//:zstd`.
- DuckDB is removed: the `http_archive` in `MODULE.bazel`, `duckdb.BUILD`, and
  `duckdb_static_loader.cc` all go away.
- `DESIGN.md` section 2.1 changes: `src/parquet/` is the one host package that
  may include the ZSTD header. The model and kernel libraries stay vendor-free.

ZSTD is about 1 MB. DuckDB is about 78 MB.

## 4. Package layout

```
src/parquet/
  thrift.{h,cc}     the Thrift compact protocol decoder
  format.{h,cc}     the parquet metadata structs and the footer parse
  levels.{h,cc}     the RLE and bit-packed hybrid decoder
  reader.{h,cc}     the file reader: pages, ZSTD, PLAIN byte arrays
  *_test.cc
```

`ParquetReader` keeps the current public shape, so `tok_train_main` changes
only its include path:

```cpp
class ParquetReader {
 public:
  struct Options {
    std::vector<std::string> files;  // paths or globs, in dataset order
    std::string text_column = "text";
    std::size_t batch_size = 256;
  };
  static std::unique_ptr<ParquetReader> Open(const Options& options,
                                             std::string* error);
  bool Next(std::vector<std::string>* documents, std::string* error);
};
```

## 5. Document sources and the loader

`DocumentSource` sits next to `DataLoader` in `include/nanochat/dataloader.h`:

```cpp
class DocumentSource {
 public:
  virtual ~DocumentSource() = default;
  virtual bool Next(std::vector<std::string>* documents,
                    std::string* error) = 0;
};
```

`ParquetSource` in `src/parquet/` implements it over `ParquetReader`. A plain
text source serves the tests.

`DataLoader` takes a `DocumentSource` and a `Tokenizer`. A producer thread
encodes documents and packs rows. Two packing modes:

- 5a. A contiguous window over a flat BOS-separated stream. This is the
  smallest change and keeps parity entry D1 open.
- 5b. The reference best-fit packer. Every row starts with BOS, the largest
  fitting document wins, and the shortest document fills the last gap. This
  closes D1.

The token byte lengths for bits-per-byte come from `Tokenizer::TokenBytes()`,
not from the `<shard>.bytes` sidecar.

## 6. Phases

### Phase 0 -- Freeze and design

- Freeze `DocumentSource` and the new `DataLoader` constructor.
- Update `DESIGN.md` section 2.1 and `docs/build.md`.
- Create `src/parquet/`.

### Phase 1 -- Native parquet reader

- Land `thrift`, `format`, `levels`, and `reader`, with tests.
- Support ZSTD, `DATA_PAGE` v1, `PLAIN`, RLE levels, one OPTIONAL column.
- Reject other codecs, dictionary pages, `DATA_PAGE_V2`, nested columns, more
  than one leaf, nulls, and other value encodings.

### Phase 2 -- Replace DuckDB

- Point `tok_train_main` at the native reader.
- Delete the DuckDB reader, the shim, the build file, and the module archive.
- Verify `tok_train_main` trains the same vocabulary on a fixture.

### Phase 3 -- Document sources

- Implement `ParquetSource` over `ParquetReader`.
- Add a plain-text source for the tests.

### Phase 4 -- Tokenizing loader

- Rewrite `src/data.cc` over `DocumentSource` and `Tokenizer`.
- Add one producer thread and a bounded document buffer.
- 4a: contiguous window. 4b: best-fit packer.

### Phase 5 -- Wire the harness

- `TrainConfig` gains a tokenizer path, a source factory, a thread count, and
  a document buffer size.
- `train_main` and `eval_main` gain `--tokenizer`, `--train-parquet`,
  `--val-parquet`, `--tokenizer-threads`, and `--buffer-docs`.
- `eval.cc` reads token bytes from the tokenizer.

### Phase 6 -- Bridge

- `python/nanochat_cpp/data.py` stops writing shards. It passes parquet paths
  and the `NCTOKEN1` path to the binary.

### Phase 7 -- Remove the shard path

- Delete `shard_writer.{h,cc}`, `tok_shard_main.cc`, and `TokenShard`.
- Remove the `--*-shard` flags and the `train_shards` and `val_shards` fields.

### Phase 8 -- Parity and performance gate

- Add a loader parity fixture that pins the C++ rows against the reference.
- Run `tools/nanochat test`, `tools/nanochat lint`, and one T2 parity job.
- Record tokens per second against the shard path in
  [performance.md](performance.md).

## 7. Tests

| Test | Tier | Purpose |
|---|---|---|
| `//src/parquet:thrift_test` | T0 | the compact decoder |
| `//src/parquet:reader_test` | T0 | order, content, batching on a fixture |
| `//src/parquet:reader_reject_test` | T0 | one error per unsupported feature |
| `//src:data_test` | T0 | the loader shift and the token bytes |
| `//src:token_loader_test` | T0 | fixed tokenizer and seed give fixed rows |
| `//tests:loader_parity_test` | T2 | the C++ rows match the reference rows |

## 8. Risks

- Reader correctness. Keep the reader strict and add the reject test.
- Dataset drift. A new writer could add dictionary encoding. The reader
  rejects it. Document the fixture regeneration.
- ZSTD dependency. If the registry is not acceptable, vendor ZSTD with
  `http_archive`, as DuckDB was.
- Host CPU load. On-the-fly tokenization competes with the GPU. Bound the
  threads and keep them inside the `train` sandbox profile.
- Determinism. The encoder is deterministic. The packer needs a fixed
  tie-break and one consumer.

## 9. Status

Phases 0 to 3 are done. The native reader replaces DuckDB, and `ParquetReader`
is a `DocumentSource`.

| Phase | State | Notes |
|---|---|---|
| 0 Design | done | `DESIGN.md` section 2.1 and `docs/build.md` updated |
| 1 Reader | done | `//src/parquet:all` passes; ZSTD comes from the registry |
| 2 DuckDB removal | done | `tok_train_main` uses the native reader |
| 3 Document source | done | `DocumentSource` in `nanochat/dataloader.h` |
| 4 Tokenizing loader | done | `DataLoader` document mode: best-fit packing |
| 5 Harness wiring | done | `--train-parquet`, `--val-parquet`, `--tokenizer` |
| 6 Bridge | done | The Python layer passes parquet when `NCTOKEN1` exists |
| 7 Shard removal | done | `TokenShard`, the shard flags, and the Python writer removed |
| 8 Parity gate | done | `//tests:loader_parity_test` |

The reader matches `pyarrow` byte for byte on three real ClimbMix shards:
count `254976`, bytes `757805763`, and an identical FNV-1a hash.
