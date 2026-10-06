// `eval_main` -- forward-only bits-per-byte evaluation. Reads pre-tokenized
// shards, builds the model, optionally loads a checkpoint, and prints the bpb
// for the train and/or validation split.
//
// The reference's `--split-tokens` describes how many tokens of a split to
// evaluate; here it is converted to a batch count with
// `steps = split_tokens / (batch * seq_len)`, mirroring `scripts/base_eval.py`.
//
//   tools/nanochat eval -- <eval_main> --val-shard data/val.bin --model
//   model.ckpt
//   tools/nanochat eval -- <eval_main> --train-shard data/train.bin
//   --val-shard data/val.bin --split-tokens 8192

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/model.h"
#include "nanochat/sandbox.h"
#include "nanochat/tokenizer.h"
#include "src/cli.h"
#include "src/parquet/reader.h"
#include "src/train.h"

namespace {

void Usage() {
  std::fprintf(
      stderr,
      "usage: eval_main [options]\n"
      "  --train-parquet GLOB   training parquet (repeatable, csv)\n"
      "  --val-parquet GLOB     validation parquet (repeatable, csv)\n"
      "  --tokenizer PATH       NCTOKEN1 artifact (parquet mode)\n"
      "  --text-column NAME     parquet text column (default text)\n"
      "  --tokenizer-threads N  encode workers (default 4)\n"
      "  --buffer-docs N        encoded documents in memory (default 1000)\n"
      "  --steps N              number of batches to evaluate\n"
      "  --split-tokens N       tokens per split; overrides --steps with\n"
      "                         N / (batch * seq_len) batches\n"
      "  --batch N              batch size\n"
      "  --seq N                sequence length (model flag below)\n"
      "  --model PATH           checkpoint to load\n"
      "  --seed N               weight init seed when no checkpoint\n"
      "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
      "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

// Builds the document-mode loader factory (docs/parquet-native.md).
nanochat::DocumentSourceFactory MakeParquetFactory(
    const std::vector<std::string>& files, const std::string& column) {
  return [files, column](
             std::string* error) -> std::unique_ptr<nanochat::DocumentSource> {
    return nanochat::OpenParquetSource(files, column, 256, error);
  };
}

// Evaluates one split through `loader` and prints its bpb. Returns the result
// so a caller could compare splits.
float EvalSplit(nanochat::Model* model, nanochat::DataLoader* loader,
                const char* split, int steps) {
  const float bpb = nanochat::EvalBpb(model, loader, steps);
  std::printf("eval_main: %s bpb %.6f\n", split, static_cast<double>(bpb));
  return bpb;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("eval");

  nanochat::Config model_config;
  std::vector<std::string> train_parquet;
  std::vector<std::string> val_parquet;
  std::string tokenizer_path;
  std::string text_column = "text";
  int tokenizer_threads = 4;
  int document_buffer = 1000;
  std::string model_path;
  int batch = 8;
  int steps = 8;
  int split_tokens = -1;
  std::uint64_t seed = 42;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "eval_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "eval_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--train-parquet") {
      nanochat::cli::AppendCsv(value, &train_parquet);
    } else if (flag == "--val-parquet") {
      nanochat::cli::AppendCsv(value, &val_parquet);
    } else if (flag == "--tokenizer") {
      tokenizer_path = value;
    } else if (flag == "--text-column") {
      text_column = value;
    } else if (flag == "--tokenizer-threads") {
      tokenizer_threads =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 4;
    } else if (flag == "--buffer-docs") {
      document_buffer =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 1000;
    } else if (flag == "--steps") {
      steps = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--split-tokens") {
      split_tokens =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--batch") {
      batch = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--model") {
      model_path = value;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else {
      std::fprintf(stderr, "eval_main: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (train_parquet.empty() && val_parquet.empty()) {
    std::fprintf(stderr, "eval_main: give --train-parquet or --val-parquet\n");
    return 2;
  }
  if (tokenizer_path.empty()) {
    std::fprintf(stderr, "eval_main: --tokenizer is required\n");
    return 2;
  }
  if (batch <= 0 || model_config.seq_len <= 0) {
    std::fprintf(stderr, "eval_main: --batch and --seq must be > 0\n");
    return 2;
  }
  if (split_tokens > 0) {
    const int per_step = batch * model_config.seq_len;
    steps = per_step > 0 ? split_tokens / per_step : 0;
    if (steps <= 0) steps = 1;
  }
  if (steps <= 0) {
    std::fprintf(stderr, "eval_main: --steps and --split-tokens must be > 0\n");
    return 2;
  }

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  if (!model_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "eval_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  } else {
    model->InitWeights(seed);
  }

  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "eval_main: cannot load tokenizer %s\n",
                 tokenizer_path.c_str());
    return 1;
  }
  if (!train_parquet.empty()) {
    nanochat::DataLoader loader(MakeParquetFactory(train_parquet, text_column),
                                tokenizer.get(), batch, model_config.seq_len,
                                seed, tokenizer_threads, document_buffer);
    EvalSplit(model.get(), &loader, "train", steps);
  }
  if (!val_parquet.empty()) {
    nanochat::DataLoader loader(MakeParquetFactory(val_parquet, text_column),
                                tokenizer.get(), batch, model_config.seq_len,
                                seed + 1, tokenizer_threads, document_buffer);
    EvalSplit(model.get(), &loader, "val", steps);
  }
  return 0;
}
