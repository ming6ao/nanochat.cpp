# Data input and tokenization

Training and evaluation read parquet documents and tokenize them during the run.
There is no shard pre-pass, no Python at runtime, and no DuckDB.

```text
parquet -> native reader -> on-the-fly tokenizer -> packed rows -> train step
```

The native tokenizer is a host utility beside `DataLoader`. It sits below the
model API and has no kernel dependency ([build.md](build.md) § Layers). The
port replaces two reference libraries: `rustbpe` trains the vocabulary, and
`tiktoken` encodes and decodes text.

```bash
# Train a vocabulary from parquet, then train the model on raw documents.
tools/nanochat train -- ./bazel-bin/src/tokenizer/tok_train_main \
  --parquet '/data/climbmix-*.parquet' --vocab-size 32768 --out tok.bin
tools/nanochat train -- ./bazel-bin/src/nanochat_cli train \
  --train-parquet '/data/climbmix-*.parquet' --val-parquet '/data/val-*.parquet' \
  --tokenizer tok.bin --layers 12 --seq 1024
```

The checkpoint format is in [checkpoint-format.md](checkpoint-format.md). Known
divergences from the reference are in [parity.md](parity.md). The chat template
stays in the bridge ([post-training.md](post-training.md) section 9).

## 1. Scope

The tokenizer supplies these functions:

- Train the base vocabulary from a text corpus.
- Save and load a portable tokenizer artifact.
- Encode text to token ids.
- Decode token ids to text.
- Decode one token at a time during generation.
- Manage the nine special tokens.
- Report the source byte length of each token.
- Read the parquet dataset with the native reader.

These functions stay out of scope:

- The chat template and the supervised mask.
- Distributed training.
- The Python bridge policy.

## 2. Reference behavior

The port copies the reference behavior. The reference is `rustbpe` for training
and `tiktoken` for inference.

### 2.1 Training

`rustbpe` runs three steps: split each document with the split pattern, count
each unique chunk, and run the merge loop.

The merge loop keeps a max-heap of pair counts. The loop pops the pair with the
largest count. A tie breaks on the smallest pair. A pair compares the left id
first and the right id second. The loop merges every occurrence in each affected
chunk. It then refreshes the counts and the positions with a lazy update.

The output is an ordered list of merges. The merge index gives the token id.
The base vocabulary holds 256 byte tokens plus one token per merge.

### 2.2 Encoding

`tiktoken` keeps a dictionary from token bytes to rank. The rank is the token
id. Encoding splits the text, maps each piece to its bytes, and merges the pair
with the lowest rank. The loop stops when no pair matches.

### 2.3 Decoding

`tiktoken` concatenates the bytes of every token. It then converts the byte
string to UTF-8 with the `replace` error mode. This conversion is lossy. A token
that holds a partial UTF-8 sequence becomes one replacement character, U+FFFD.

A special token decodes to its own name. For example, the id for `<|bos|>`
decodes to the seven bytes `<|bos|>`.

### 2.4 Special tokens

The reference tokenizer assigns the ids in this order, directly after the base
vocabulary.

| Index | Name |
|---|---|
| 0 | `<|bos|>` |
| 1 | `<|user_start|>` |
| 2 | `<|user_end|>` |
| 3 | `<|assistant_start|>` |
| 4 | `<|assistant_end|>` |
| 5 | `<|python_start|>` |
| 6 | `<|python_end|>` |
| 7 | `<|output_start|>` |
| 8 | `<|output_end|>` |

The base vocabulary size is `vocab_size - 9`. For the shipped 32768 vocabulary,
the base holds 32759 tokens. The special ids run from 32759 to 32767.

### 2.5 The split pattern

The pattern below is the nanochat pattern. It differs from the GPT-4 pattern in
one detail: the number group holds one or two digits, not one to three.

```text
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

The pattern has seven alternatives. A match picks the leftmost alternative. The
alternatives are, in order:

1. An apostrophe plus a contraction (`'s`, `'ll`, `'ve`, and similar).
2. An optional symbol, then one or more letters.
3. One or two digits.
4. An optional space, then one or more symbols, then line ends.
5. Whitespace, then a line end.
6. Trailing whitespace.
7. A whitespace run.

`tiktoken` uses the `fancy-regex` engine. That engine supports the possessive
quantifiers (`?+`, `++`) and the lookahead (`(?!\S)`) in the pattern.

## 3. Design decisions

| ID | Decision | Reason |
|---|---|---|
| D1 | Hand-written splitter for the one fixed pattern. | No third-party dependency; small and auditable. |
| D2 | Unicode tables from Unicode 16.0.0. | Matches `regex-syntax` 0.8.11 in `tiktoken`. |
| D3 | A new `NCTOKEN1` artifact. | The reference pickle and `torch` files are not portable. |
| D4 | An exact port of the reference merge loop. | The merge order defines the vocabulary. |
| D5 | `int64` pair counts. | The reference uses `int32`; the wider type avoids overflow. |
| D6 | The tokenizer is a host utility. | [build.md](build.md) § Layers places it beside `DataLoader`. |
| D7 | A native parquet reader for the input. | Reads the `text` column in dataset order without `pyarrow` or DuckDB. |

D1 rejects an unknown pattern at load time. The C++ splitter supports the
nanochat pattern only. The generator writes the pattern into the artifact, so
the loader can check it.

D2 pins one Unicode version. The splitter needs three predicates: letter
(`\p{L}`), digit (`\p{N}`), and whitespace (`\s`). The letter table holds about
694 ranges, the digit table about 148, and the whitespace table 10.

## 4. Architecture and file layout

The tokenizer is a host utility. It uses the C++ standard library and OpenMP.
It does not include `nanochat/kernels.h`. Every test runs on the CPU (tier T0).

| File | Contents |
|---|---|
| `include/nanochat/tokenizer.h` | The `Tokenizer` interface, `TokenStreamDecoder`, and `LoadTokenizer`. |
| `include/nanochat/bpe_trainer.h` | The `BpeTrainer` interface. |
| `src/tokenizer/tokenizer.cc` | Load, save, encode, decode, special tokens, stream decoder. |
| `src/tokenizer/tokenizer_internal.h` | The internal `NCTOKEN1` writer and merge recovery. |
| `src/tokenizer/split_pattern.{h,cc}` | The fixed-pattern scanner. |
| `src/tokenizer/utf8.{h,cc}` | UTF-8 validation and the lossy replace conversion. |
| `src/tokenizer/unicode_tables.inc` | The generated range tables (Unicode 16.0.0). |
| `src/tokenizer/bpe_trainer.cc` | The chunk counts and the merge loop. |
| `src/tokenizer/tok_train_main.cc` | The training command line. |
| `src/parquet/reader.{h,cc}` | The native parquet reader. Part 1 holds its design. |
| `tools/gen_unicode_tables.py` | The table generator. |
| `tools/dump_fixtures.py tokenizer` | The reference fixture generator. |
| `tools/check_tokenizer_parity.py` | The full-vocabulary parity gate. |

## 5. Artifact format

The container is the only runtime tokenizer artifact. The native trainer
`tok_train_main` is the only producer. The runtime never reads a Python pickle.

The artifact is little-endian. The loader checks the magic and the version. It
returns an error for a bad file, and `LoadTokenizer` returns `null`.

```text
offset  size   field
0       8      magic "NCTOKEN1"
8       4      version (1)
12      4      pattern length
16      n      pattern bytes (UTF-8)
16+n    4      merge count
...     8*m    merge pairs (u32 left, u32 right); rank = 256 + index
...     4      special count
...     ...    special records (u32 name length, name bytes, u32 id)
```

The loader derives the token bytes from the merge pairs. A byte token has one
byte. A merge token has the bytes of its two parts. A special token has zero
bytes.

## 6. The splitter

`src/tokenizer/split_pattern.cc` implements the pattern as a scanner over UTF-8
codepoints. The scanner holds one start position. At each start it tries the
seven alternatives in order, and it takes the first match. It then advances the
start past the match.

The scanner must copy the possessive quantifiers. In alternative 2, the
optional symbol never gives back its match. For the string `" !abc"`, the
scanner returns `" !"` then `"abc"`. Alternative 4 wins for the first piece.

The scanner classifies each codepoint with the generated tables. The tables list
the ranges in ascending order, and a lookup uses a binary search.

## 7. The encoder

Encoding runs two steps for each piece from the splitter:

1. Map each byte of the piece to its token id. A byte id equals the byte value.
2. Merge the adjacent pair with the lowest rank until no pair matches.

Step 2 is the standard BPE loop. The reference and the port pick the same pair
because the rank order does not change. The encoder reads the ranks from the
artifact.

## 8. The decoder and on-the-fly decode

nanochat decodes a token column during generation. `nanochat/engine.py` yields
one token per row per step. The consumer prints the text for that token at once.

### 8.1 The `Decode` primitive

`Tokenizer::Decode` copies the bytes of each token into one buffer. It then
converts the buffer to a C++ `std::string`. The conversion mirrors the lossy
UTF-8 mode of `tiktoken`, so a bad byte becomes U+FFFD. A special token
contributes its name bytes.

### 8.2 Per-token bytes

The stream needs the bytes of one token. The interface has one method:

```cpp
// Append the raw bytes of one token to *out. Return false for a bad id.
virtual bool AppendTokenBytes(int id, std::string* out) const = 0;
```

`TokenBytes()` reports the byte length of each token for bits per byte. A
special token has length zero.

### 8.3 Streaming decode

A single token can end in the middle of a UTF-8 sequence. The reference decodes
that token to U+FFFD, and the next token repairs nothing. A reader then sees a
stray replacement character. The 32768 vocabulary has 74 merges with this
property, for example the two bytes `\xe2\x80`.

The port offers a small buffer to avoid the stray character:

```cpp
class TokenStreamDecoder {
 public:
  explicit TokenStreamDecoder(const Tokenizer& tokenizer);

  // Decode the next token. Append complete text to *out. Hold an incomplete
  // UTF-8 suffix for the next call.
  void Push(int id, std::string* out);

  // Append the held bytes with replacement, then clear the buffer.
  void Flush(std::string* out);

 private:
  const Tokenizer& tokenizer_;
  std::string pending_;
};
```

`Push` appends the token bytes to `pending_`, then emits the longest complete
UTF-8 prefix. It holds the rest. `Flush` emits the held bytes at the end of a
row.

The stream decoder is a local choice, not reference behavior. A consumer that
needs exact parity calls `Decode` per token instead. The parity test covers both
modes.

### 8.4 Special tokens in a stream

The stream decodes a special token to its name, like the reference. A chat
consumer often wants to hide or to highlight a special token. The interface has
a predicate for that case:

```cpp
virtual bool IsSpecial(int id) const = 0;
```

The consumer reads the id before the text. It can drop the token, or it can
print the name.

### 8.5 The generation loop

The C++ generation loop decodes as it samples. For each step:

1. Read the sampled id from the logit row.
2. If the id is a terminal id, stop the row.
3. Push the id into the stream decoder.
4. Write the returned text to the output.

The loop in `nanochat/engine.py` also decodes a Python block for the calculator
tool. That path decodes a full token list, then encodes a result string. The
same two primitives serve it: `Decode` and `Encode`.

## 9. The trainer

`BpeTrainer` ports `rustbpe`'s `train_core_incremental`:

```cpp
class BpeTrainer {
 public:
  explicit BpeTrainer(std::string pattern = NanochatSplitPattern());

  void AddDocument(std::string_view text);              // split and count
  void AddChunk(std::string_view chunk, int64_t count); // low level
  void Train(int base_vocab_size);                      // 256 + merges

  std::vector<std::pair<std::vector<uint8_t>, int>> MergeableRanks() const;
  const std::string& pattern() const;
};
```

The trainer splits each document with the same scanner as the encoder. It adds
the count of each unique chunk, and `Train` runs the merge loop. The merge loop
stays single-threaded, because the result must not depend on thread order. The
split and count step can use OpenMP, because the reduction is commutative.

The pair counter uses `int64`. The reference uses `int32`, and a real corpus can
come close to the `int32` limit. The wider type changes no result below the
limit.

```bash
# One document per line, or a parquet dataset.
./bazel-bin/src/tokenizer/tok_train_main --text corpus.txt --vocab-size 32768 \
  --doc-cap 10000 --max-chars 2000000000 --out tok.bin
./bazel-bin/src/tokenizer/tok_train_main --parquet '/data/*.parquet' \
  --text-column text --vocab-size 32768 --out tok.bin
```

`--doc-cap` is the per-document character cap. `--max-chars` is the total
character budget.

## 10. The loader and row packing

`DocumentSource` sits beside `DataLoader` in `include/nanochat/dataloader.h`:

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
encodes documents and packs rows. The packer is the reference best-fit rule:
every row starts with `BOS`, the largest fitting document wins, and the shortest
document fills the last gap. The token byte lengths for bits-per-byte come from
`Tokenizer::TokenBytes()`, not from a sidecar file.

`DataLoader::Reset` cycles epochs, so a resumed run starts at the first row of a
new epoch.

## 11. Command-line use

| Flag | Meaning |
|---|---|
| `--train-parquet <glob>` | the training dataset |
| `--val-parquet <glob>` | the validation dataset |
| `--tokenizer <path>` | the `NCTOKEN1` artifact |
| `--tokenizer-threads <n>` | the producer count |
| `--buffer-docs <n>` | the bounded document buffer |
| `--packed-rows <path>` | read pre-packed rows instead of documents |

`nanochat_cli train` and `nanochat_cli eval` take these flags. The Python layer
passes parquet paths and the `NCTOKEN1` path when that artifact exists, so no
shard is written. The former shard producer (`tok_shard_main`), its writer, and
the `NANO` shard reader are gone.

## 12. Tests and parity

Every test is tier T0. The tests need no GPU and no broker.

| Test | Purpose |
|---|---|
| `//src/tokenizer:tokenizer_test` | Round trip, special ids, token bytes, bad-file error. |
| `//src/tokenizer:split_pattern_test` | The split cases, plus a random Unicode corpus. |
| `//src/tokenizer:bpe_trainer_test` | The reference merge list, rank by rank. |
| `//src/tokenizer:tokenizer_parity_test` | The fixture: merges, encode, decode, token bytes. |
| `//src/parquet:reader_test` | The reader order and the text column on a tiny fixture. |
| `//src/parquet:reader_reject_test` | One error per unsupported feature. |
| `//src:data_test` | The loader shift and the token bytes. |
| `//tests:loader_parity_test` | The C++ rows match the reference rows (T2). |

The fixture comes from `tools/dump_fixtures.py tokenizer`. The script trains a
small reference tokenizer on a fixed corpus, and it records:

- the pattern, the ordered merges, and the special tokens;
- the split of a curated string set;
- encode cases and decode cases, including special tokens;
- the token bytes;
- the `rustbpe` commit and the `tiktoken` version.

The fixture is data, so the C++ test never imports Python. The script runs the
reference twice, and it checks that the two merge hashes agree.

`tools/check_tokenizer_parity.py` is the full-vocabulary gate. It builds a
corpus from the parquet shards, trains the reference `rustbpe`, runs the native
`tok_train_main` on the same corpus, and compares every base token. On a
40M-character ClimbMix corpus at `vocab_size` 32768, both reach 32503 merges and
all 32759 base tokens match.

The reader matches `pyarrow` byte for byte on three real ClimbMix shards: count
`254976`, bytes `757805763`, and an identical FNV-1a hash.

## 13. Limits

- **Unicode drift.** The tables must come from Unicode 16.0.0, the version inside
  `tiktoken`. The random corpus test catches a difference.
- **The pattern is closed.** Possessive quantifiers and the lookahead are the
  edge cases. A second pattern needs a second scanner.
- **Dataset drift.** A new writer could add dictionary encoding. The reader
  rejects it, so a fixture regeneration is visible.
- **Host CPU load.** On-the-fly tokenization competes with the GPU. Bound the
  threads and keep them inside the `train` sandbox profile.
- **Determinism.** The encoder is deterministic. The packer needs a fixed
  tie-break and one consumer.
- **ZSTD is the one host dependency.** It links the parquet package only, so the
  model and the kernels stay vendor-free.

## Part 1. The native parquet reader

The data path is `parquet -> native reader -> on-the-fly tokenizer -> training
rows`. Two results follow: no shard pre-pass, and no DuckDB.

### 1. Why a native reader is feasible

One writer produced every dataset file: `parquet-cpp-arrow version 21.0.0`. A
scan of 33 ClimbMix shards and 2732 column chunks shows one fixed shape:

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

The reader supports this subset and rejects everything else with an error. The
supported subset is small. The container is not: it needs a Thrift compact
footer, page headers, RLE bit-packed levels, and ZSTD frames.

### 2. The dependency

- ZSTD comes from the Bazel Central Registry: `bazel_dep(name = "zstd", version =
  "1.5.7")`, target `@zstd//:zstd`.
- `src/parquet/` is the one host package that may include the ZSTD header. The
  model and kernel libraries stay vendor-free
  ([build.md](build.md#dependency-invariants)).
- DuckDB is gone: the `http_archive` in `MODULE.bazel`, `duckdb.BUILD`, and
  `duckdb_static_loader.cc` all left the tree. ZSTD is about 1 MB and DuckDB was
  about 78 MB.

### 3. Package layout

```
src/parquet/
  thrift.{h,cc}     the Thrift compact protocol decoder
  format.{h,cc}     the parquet metadata structs and the footer parse
  levels.{h,cc}     the RLE and bit-packed hybrid decoder
  reader.{h,cc}     the file reader: pages, ZSTD, PLAIN byte arrays
  *_test.cc
```

### 4. The reader API

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

`Open` takes a file glob or a file list. `Next` streams one bounded batch of
documents. The reader selects the text column, orders the rows by file name and
row number, and preserves the dataset order. A small batch size bounds memory.

The reader rejects each unsupported feature with its own error and its own test:
another codec, a dictionary page, `DATA_PAGE_V2`, a nested column, more than one
leaf, a null value, and another value encoding.

### 5. What a new dataset needs

A new writer must keep the shape above or the reader rejects its file. When a
producer changes the shape, regenerate the packed-row parity fixture from that
producer, so `//tests:loader_parity_test` compares the same bytes on both sides.

## See also

- [model.md](model.md) — the training loop that consumes the rows.
- [build.md](build.md) — the layering that keeps the reader vendor-facing.
- [parity.md](parity.md) — the data-order entry D1.
- [post-training.md](post-training.md) — the chat data model above this layer.
