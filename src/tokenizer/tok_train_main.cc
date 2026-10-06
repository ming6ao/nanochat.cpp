// `tok_train_main` -- the native byte pair encoding trainer CLI
// (docs/tokenizer.md section 9). It reads documents from a text file (one per
// line) or from a parquet dataset, trains the base vocabulary, and writes an
// `NCTOKEN1` artifact.
//
//   tools/nanochat run t0-cpu -- <tok_train_main> --text corpus.txt
//       --vocab-size 512 --out tokenizer.nctoken
//
// Every run must go through `tools/nanochat`; the entry point refuses to start
// outside the sandbox (`RequireSandboxOrDie`).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/bpe_trainer.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/parquet/reader.h"
#include "src/tokenizer/tokenizer_internal.h"

namespace {

// The nine special tokens, in id order. The ids start at the base vocabulary
// size (docs/tokenizer.md section 2.4).
const char* const kSpecialTokens[] = {
    "<|bos|>",           "<|user_start|>",
    "<|user_end|>",      "<|assistant_start|>",
    "<|assistant_end|>", "<|python_start|>",
    "<|python_end|>",    "<|output_start|>",
    "<|output_end|>"};

void Usage() {
  std::fprintf(
      stderr,
      "usage: tok_train_main [options]\n"
      "  --text PATH            one document per line\n"
      "  --parquet PATH         a parquet file or glob (repeatable, csv)\n"
      "  --text-column NAME     parquet text column (default text)\n"
      "  --vocab-size N         full vocab size, including specials\n"
      "  --doc-cap N            stop after N documents (default 10000)\n"
      "  --max-chars N          stop after N source characters (0 = all)\n"
      "  --out PATH             write the NCTOKEN1 artifact here\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("tok_train");

  std::string text_path;
  std::vector<std::string> parquet_files;
  std::string text_column = "text";
  std::string out_path;
  int vocab_size = 32768;
  int doc_cap = 10000;
  std::uint64_t max_chars = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "tok_train_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--text") {
      text_path = value;
    } else if (flag == "--parquet") {
      nanochat::cli::AppendCsv(value, &parquet_files);
    } else if (flag == "--text-column") {
      text_column = value;
    } else if (flag == "--vocab-size") {
      if (!nanochat::cli::ParseInt(value, &parsed_int)) {
        std::fprintf(stderr, "tok_train_main: bad --vocab-size\n");
        return 2;
      }
      vocab_size = parsed_int;
    } else if (flag == "--doc-cap") {
      if (!nanochat::cli::ParseInt(value, &parsed_int)) {
        std::fprintf(stderr, "tok_train_main: bad --doc-cap\n");
        return 2;
      }
      doc_cap = parsed_int;
    } else if (flag == "--max-chars") {
      if (!nanochat::cli::ParseU64(value, &parsed_u64)) {
        std::fprintf(stderr, "tok_train_main: bad --max-chars\n");
        return 2;
      }
      max_chars = parsed_u64;
    } else if (flag == "--out") {
      out_path = value;
    } else {
      std::fprintf(stderr, "tok_train_main: unknown flag %s\n", flag.c_str());
      return 2;
    }
  }

  if (out_path.empty()) {
    std::fprintf(stderr, "tok_train_main: --out is required\n");
    return 2;
  }
  if (text_path.empty() && parquet_files.empty()) {
    std::fprintf(stderr, "tok_train_main: give --text or --parquet\n");
    return 2;
  }
  if (!text_path.empty() && !parquet_files.empty()) {
    std::fprintf(stderr,
                 "tok_train_main: --text and --parquet are exclusive\n");
    return 2;
  }

  const int special_count =
      static_cast<int>(sizeof(kSpecialTokens) / sizeof(kSpecialTokens[0]));
  const int base_vocab_size = vocab_size - special_count;
  if (base_vocab_size < 256) {
    std::fprintf(stderr, "tok_train_main: --vocab-size must be at least %d\n",
                 256 + special_count);
    return 2;
  }

  nanochat::BpeTrainer trainer;
  std::uint64_t documents = 0;
  std::uint64_t characters = 0;
  bool done = false;

  if (!text_path.empty()) {
    std::ifstream in(text_path);
    if (!in) {
      std::fprintf(stderr, "tok_train_main: cannot open %s\n",
                   text_path.c_str());
      return 1;
    }
    std::string line;
    while (!done && std::getline(in, line)) {
      if (doc_cap > 0 && documents >= static_cast<std::uint64_t>(doc_cap)) {
        break;
      }
      if (max_chars > 0 && characters >= max_chars) break;
      trainer.AddDocument(line);
      ++documents;
      characters += line.size();
    }
  } else {
    nanochat::ParquetReader::Options options;
    options.files = parquet_files;
    options.text_column = text_column;
    std::string error;
    std::unique_ptr<nanochat::ParquetReader> reader =
        nanochat::ParquetReader::Open(options, &error);
    if (reader == nullptr) {
      std::fprintf(stderr, "tok_train_main: %s\n", error.c_str());
      return 1;
    }
    std::vector<std::string> batch;
    while (!done && reader->Next(&batch, &error)) {
      for (const std::string& document : batch) {
        if (doc_cap > 0 && documents >= static_cast<std::uint64_t>(doc_cap)) {
          done = true;
          break;
        }
        if (max_chars > 0 && characters >= max_chars) {
          done = true;
          break;
        }
        trainer.AddDocument(document);
        ++documents;
        characters += document.size();
      }
    }
    if (!error.empty()) {
      std::fprintf(stderr, "tok_train_main: %s\n", error.c_str());
      return 1;
    }
  }

  trainer.Train(base_vocab_size);
  const std::vector<std::pair<std::vector<std::uint8_t>, int>> merges =
      trainer.MergeableRanks();
  const std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs =
      nanochat::RecoverMergePairs(merges);
  // The trainer stops early when the corpus has no more pairs, so the actual
  // base vocabulary can be smaller than the requested one. The special ids
  // follow the real merge count, so the artifact stays self-consistent and
  // `LoadTokenizer` accepts it.
  const int actual_base = 256 + static_cast<int>(merges.size());
  std::vector<std::pair<std::string, std::uint32_t>> specials;
  specials.reserve(static_cast<std::size_t>(special_count));
  for (int index = 0; index < special_count; ++index) {
    specials.emplace_back(kSpecialTokens[index],
                          static_cast<std::uint32_t>(actual_base + index));
  }
  if (!nanochat::SaveTokenizer(out_path, trainer.pattern(), pairs, specials)) {
    std::fprintf(stderr, "tok_train_main: cannot write %s\n", out_path.c_str());
    return 1;
  }

  std::printf(
      "tok_train_main: wrote %s (documents %llu, merges %zu, vocab %d)\n",
      out_path.c_str(), static_cast<unsigned long long>(documents),
      merges.size(), actual_base + special_count);
  return 0;
}
