// `tok_shard_main` -- materialize a parquet dataset into a `NANO` token shard
// plus the `<shard>.bytes` sidecar (docs/tokenizer.md section 10). It is the
// native replacement for `python/nanochat_cpp/data.py`. Each document is
// encoded with a leading BOS, the same way the Python path does.
//
//   tools/nanochat run t0-cpu -- <tok_shard_main> --tokenizer t.nctoken
//       --parquet 'data/*.parquet' --out data/train.bin
//
// Every run must go through `tools/nanochat`; the entry point refuses to start
// outside the sandbox (`RequireSandboxOrDie`).

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/sandbox.h"
#include "nanochat/tokenizer.h"
#include "src/cli.h"
#include "src/parquet_reader.h"
#include "src/shard_writer.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: tok_shard_main [options]\n"
               "  --parquet PATH         a parquet file or glob (repeatable, "
               "csv)\n"
               "  --text-column NAME     parquet text column (default text)\n"
               "  --tokenizer PATH       the NCTOKEN1 artifact\n"
               "  --out PATH             write the NANO shard here\n"
               "  --width N              token width, 2 or 4 (default 2)\n"
               "  --max-tokens N         stop after N tokens (0 = all)\n");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("tok_shard");

  std::vector<std::string> parquet_files;
  std::string text_column = "text";
  std::string tokenizer_path;
  std::string out_path;
  int width = 2;
  std::uint64_t max_tokens = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "tok_shard_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--parquet") {
      nanochat::cli::AppendCsv(value, &parquet_files);
    } else if (flag == "--text-column") {
      text_column = value;
    } else if (flag == "--tokenizer") {
      tokenizer_path = value;
    } else if (flag == "--out") {
      out_path = value;
    } else if (flag == "--width") {
      if (!nanochat::cli::ParseInt(value, &parsed_int)) {
        std::fprintf(stderr, "tok_shard_main: bad --width\n");
        return 2;
      }
      width = parsed_int;
    } else if (flag == "--max-tokens") {
      if (!nanochat::cli::ParseU64(value, &parsed_u64)) {
        std::fprintf(stderr, "tok_shard_main: bad --max-tokens\n");
        return 2;
      }
      max_tokens = parsed_u64;
    } else {
      std::fprintf(stderr, "tok_shard_main: unknown flag %s\n", flag.c_str());
      return 2;
    }
  }

  if (parquet_files.empty() || tokenizer_path.empty() || out_path.empty()) {
    std::fprintf(stderr,
                 "tok_shard_main: --parquet, --tokenizer, and --out are "
                 "required\n");
    return 2;
  }
  if (width != 2 && width != 4) {
    std::fprintf(stderr, "tok_shard_main: --width must be 2 or 4\n");
    return 2;
  }

  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tok_shard_main: cannot load %s\n",
                 tokenizer_path.c_str());
    return 1;
  }

  nanochat::ParquetReader::Options options;
  options.files = parquet_files;
  options.text_column = text_column;
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
    return 1;
  }

  std::unique_ptr<nanochat::ShardWriter> writer =
      nanochat::ShardWriter::Open(out_path, width, &error);
  if (writer == nullptr) {
    std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
    return 1;
  }

  const int bos = tokenizer->bos_id();
  std::vector<std::string> batch;
  std::uint64_t written = 0;
  bool done = false;
  while (!done && reader->Next(&batch, &error)) {
    for (const std::string& document : batch) {
      const std::vector<int> ids = tokenizer->Encode(document, bos);
      std::size_t take = ids.size();
      if (max_tokens > 0 && written >= max_tokens) {
        done = true;
        break;
      }
      if (max_tokens > 0) {
        const std::uint64_t room = max_tokens - written;
        if (static_cast<std::uint64_t>(take) > room) {
          take = static_cast<std::size_t>(room);
        }
      }
      if (!writer->Write(ids.data(), static_cast<std::int64_t>(take), &error)) {
        std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
        return 1;
      }
      written += take;
      if (max_tokens > 0 && written >= max_tokens) {
        done = true;
        break;
      }
    }
  }
  if (!error.empty()) {
    std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
    return 1;
  }
  if (!writer->Close(&error)) {
    std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
    return 1;
  }
  if (!nanochat::WriteTokenBytes(out_path, tokenizer->TokenBytes(),
                                 tokenizer->vocab_size(), &error)) {
    std::fprintf(stderr, "tok_shard_main: %s\n", error.c_str());
    return 1;
  }

  std::printf("tok_shard_main: wrote %s (%llu tokens, width %d)\n",
              out_path.c_str(), static_cast<unsigned long long>(written),
              width);
  return 0;
}
