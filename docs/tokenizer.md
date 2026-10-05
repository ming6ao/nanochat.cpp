# The native tokenizer

nanochat divides tokenization between two libraries. `rustbpe` trains the byte
pair encoding (BPE) vocabulary. `tiktoken` encodes and decodes text at
inference. `nanochat.cpp` reads pre-tokenized token shards, so the tree needed
no tokenizer (see [model.md](model.md)).

This document specifies a native C++ port of both libraries. The port lets the
C++ binaries accept text and print text. The generation loop decodes each
sampled token immediately, the same way `nanochat/engine.py` does.

The tokenizer stays a host utility beside `DataLoader`. It sits below the model
API, and it has no kernel dependency ([DESIGN.md](../DESIGN.md) section 2).

The training data format is in [model.md](model.md). Known divergences are in
[parity.md](parity.md). The chat template stays in the bridge
([post-training.md](post-training.md) section 9).

## 1. Scope

The port supplies these functions:

- Train the base vocabulary from a text corpus.
- Save and load a portable tokenizer artifact.
- Encode text to token ids.
- Decode token ids to text.
- Decode one token at a time during generation.
- Manage the nine special tokens.
- Report the source byte length of each token.
- Read the parquet dataset with the DuckDB C++ API.

These functions stay out of scope:

- The chat template and the supervised mask.
- The best-fit row packer (parity entry D1).
- Distributed training.
- The Python bridge policy.

## 2. Reference behavior

The port must copy the reference behavior. The reference is `rustbpe` for
training and `tiktoken` for inference.

### 2.1 Training

`rustbpe` runs three steps:

1. Split each document with the split pattern.
2. Count each unique chunk.
3. Run the merge loop.

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
decodes to the seven bytes `<|bos|>`. The reference probes confirm both rules.

### 2.4 Special tokens

The nine names and their positions appear below. The reference tokenizer assigns
the ids in this order, directly after the base vocabulary.

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
| D6 | The tokenizer is a host utility. | [DESIGN.md](../DESIGN.md) section 2 places it beside `DataLoader`. |
| D7 | DuckDB C++ API for the parquet input. | Reads the `text` column in dataset order without `pyarrow`. |

D1 rejects an unknown pattern at load time. The C++ splitter supports the
nanochat pattern only. The generator writes the pattern into the artifact, so
the loader can check it.

D2 pins one Unicode version. The splitter needs three predicates: letter
(`\p{L}`), digit (`\p{N}`), and whitespace (`\s`). The letter table holds about
694 ranges. The digit table holds about 148 ranges. The whitespace table holds
10 ranges.

## 4. Architecture and file layout

The tokenizer is a host utility. It uses the C++ standard library and OpenMP.
It does not include `nanochat/kernels.h`. Every test runs on the CPU (tier T0).

| File | Contents |
|---|---|
| `include/nanochat/tokenizer.h` | The `Tokenizer` interface, the stream decoder, load and save. |
| `include/nanochat/bpe_trainer.h` | The `BpeTrainer` interface. |
| `src/tokenizer.cc` | Load, save, encode, decode, special tokens. |
| `src/split_pattern.{h,cc}` | The fixed-pattern scanner. |
| `src/unicode_tables.inc` | The generated range tables. |
| `src/bpe_trainer.cc` | The chunk counts and the merge loop. |
| `src/tok_train_main.cc` | The training command line. |
| `src/parquet_reader.{h,cc}` | The DuckDB C++ parquet reader. |
| `src/shard_writer.{h,cc}` | The `NANO` shard writer. |
| `src/tok_shard_main.cc` | The parquet-to-shard command line. |
| `tools/gen_unicode_tables.py` | The table generator. |
| `tools/dump_tokenizer_fixture.py` | The reference fixture generator. |
| `tools/convert_tokenizer.py` | The reference-to-`NCTOKEN1` converter. |

## 5. Artifact format

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

`src/split_pattern.cc` implements the pattern as a scanner over UTF-8
codepoints. The scanner holds one start position. At each start it tries the
seven alternatives in order, and it takes the first match. It then advances the
start past the match.

The scanner must copy the possessive quantifiers. In alternative 2, the optional
symbol never gives back its match. For the string `" !abc"`, the scanner returns
`" !"` then `"abc"`. Alternative 4 wins for the first piece.

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

This section is the reason for the port. nanochat decodes a token column during
generation. `nanochat/engine.py` yields one token per row per step. The
consumer prints the text for that token at once.

### 8.1 The `Decode` primitive

`Tokenizer::Decode` copies the bytes of each token into one buffer. It then
converts the buffer to a C++ `std::string`. The conversion mirrors the lossy
UTF-8 mode of `tiktoken`, so a bad byte becomes U+FFFD.

The method also handles a special token. A special token contributes its name
bytes.

### 8.2 Per-token bytes

The stream needs the bytes of one token. The interface gains one method:

```cpp
// Append the raw bytes of one token to *out. Return false for a bad id.
virtual bool AppendTokenBytes(int id, std::string* out) const = 0;
```

`TokenBytes()` keeps its current meaning. It reports the byte length of each
token for bits per byte. A special token has length zero.

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
consumer often wants to hide or to highlight a special token. The interface
gains a predicate for that case:

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

`BpeTrainer` ports `rustbpe`'s `train_core_incremental`. The interface is:

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
the count of each unique chunk. `Train` runs the merge loop.

The merge loop stays single-threaded. The result must not depend on thread
order. The split and count step can use OpenMP, because the reduction is
commutative.

The pair counter uses `int64`. The reference uses `int32`, and a real corpus can
come close to the `int32` limit. The wider type changes no result below the
limit.

`src/tok_train_main.cc` reads one document per line with `--text`, or a parquet
dataset with `--parquet`. It takes `--vocab-size`, `--doc-cap` (default 10000),
`--max-chars`, and `--out`. It writes `NCTOKEN1`. Section 10 gives the parquet
reader.

## 10. Parquet input and shard materialization

The reference dataset is a set of parquet files with a `text` column. The native
path reads that column with the DuckDB C++ API, so tokenization needs no
`pyarrow` and no Python.

`src/parquet_reader.cc` wraps one DuckDB connection. `Open` takes a file glob
or a file list. `Next` streams one bounded batch of documents. The reader
selects the text column, orders the rows by file name and row number, and
preserves the dataset order. A small batch size bounds memory.

`src/tok_train_main.cc` accepts `--parquet <glob>` and `--text-column <name>`
(default `text`). It reads documents from the reader and calls
`BpeTrainer::AddDocument`. The `--text <file>` mode stays for one document per
line.

`src/tok_shard_main.cc` tokenizes a parquet dataset into a `NANO` shard plus
the `<shard>.bytes` sidecar. It uses `src/shard_writer.{h,cc}` and the `Encode`
method. This tool is the native replacement for `python/nanochat_cpp/data.py`.

The DuckDB C++ library is a host dependency. `MODULE.bazel` declares it, and
`src/BUILD.bazel` links it into `tok_train_main` and `tok_shard_main` only. The
model and the kernels do not link it.

The build uses the prebuilt DuckDB release archive for the host platform. An
`http_archive` in `MODULE.bazel` unpacks `duckdb.hpp` and
`libduckdb_static.a`, and a `cc_import` target exposes them. The pinned release
is DuckDB 1.5.6. The archive sha256 is
`b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b`. The static
library keeps the tools self-contained, and the archive avoids a long
amalgamation compile.

## 11. Integration

- `generate_main`: accept a text prompt, encode it with a leading BOS, and
  stream the decoded output. This changes the current token-id-only contract.
- `eval`: decode the sample rows for base evaluation.
- `tests`: the parity test uses the fixture, so it needs no Python.
- The bridge: `python/nanochat_cpp/data.py` can build a `tiktoken` encoding from
  the `NCTOKEN1` merges, or it can call `tok_shard_main`. Its cache fingerprint
  then lists the new file.
- `data_test.cc`: the `LoadTokenizer` contract changes from "always null" to
  "null on a bad file".

## 12. Tests and parity

Every test is tier T0. The tests need no GPU and no broker.

| Test | Purpose |
|---|---|
| `src/tokenizer_test.cc` | Round trip, special ids, token bytes, bad-file error. |
| `src/split_pattern_test.cc` | The split cases, plus a random Unicode corpus. |
| `src/bpe_trainer_test.cc` | The reference merge list, rank by rank. |
| `src/tokenizer_parity_test.cc` | The fixture: merges, encode, decode, token bytes. |
| `src/parquet_reader_test.cc` | The reader order and the text column on a tiny fixture. |

The fixture comes from `tools/dump_tokenizer_fixture.py`. The script trains a
small reference tokenizer on a fixed corpus, and it records:

- the pattern, the ordered merges, and the special tokens;
- the split of a curated string set;
- encode cases and decode cases, including special tokens;
- the token bytes;
- the `rustbpe` commit and the `tiktoken` version.

The fixture is data, so the C++ test never imports Python. The script runs the
reference twice, and it checks that the two merge hashes agree.

## 13. Roadmap

| Phase | Work | Gate |
|---|---|---|
| P0 | The fixture, the Unicode tables, the frozen pattern. | Two reference runs agree. |
| P1 | Load, save, encode, decode, special tokens. | Fixture encode and decode cases pass. |
| P2 | The splitter and the tables. | Fixture split cases pass. |
| P3 | The trainer, the training command line, and the parquet reader. | Fixture merges match in order; the reader order matches the dataset. |
| P4 | Docs, parity entries, and the generation integration. | `tools/nanochat test` and `lint` pass. |

## 14. Risks

- Unicode drift. The tables must come from Unicode 16.0.0, the version inside
  `tiktoken`. The random corpus test catches a difference.
- Possessive and lookahead edge cases. The curated list covers the known cases.
  The random corpus extends the cover.
- Reference determinism. The fixture freezes the `rustbpe` commit. Two runs must
  agree before the fixture enters the repository.
- The frozen header. `include/nanochat/tokenizer.h` is architect-owned. The
  changes are additive.
- Shared files. `src/data.cc`, `src/data_test.cc`, and `src/BUILD.bazel` have
  other owners.
- DuckDB weight. The DuckDB C++ library is large and slow to build. It enters
  the host tools only, so the training runtime stays small.

## 15. Status

Planned. The reference behavior and the artifact format do not change. No code
exists yet.
