// `nanochat_cli eval` -- forward-only bits-per-byte evaluation. Reads
// pre-tokenized shards, builds the model, optionally loads a checkpoint, and
// prints the bpb for the train and/or validation split.
//
// The reference's `--split-tokens` describes how many tokens of a split to
// evaluate; here it is converted to a batch count with
// `steps = split_tokens / (batch * seq_len)`, mirroring `scripts/base_eval.py`.
//
//   tools/nanochat eval -- <nanochat_cli> eval --val-parquet data/val.parquet
//       --model model.ckpt
//   tools/nanochat eval -- <nanochat_cli> eval --train-parquet
//   data/train.parquet
//       --val-parquet data/val.parquet --split-tokens 8192
//
// The dispatcher applies the sandbox guard before it calls `RunEval`.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/dataloader.h"
#include "nanochat/model.h"
#include "nanochat/tokenizer.h"
#include "src/cli.h"
#include "src/command.h"
#include "src/parquet/reader.h"
#include "src/train.h"

namespace nanochat {
namespace {

// The prefix of every diagnostic. It stays `eval_main` because a reader and a
// log pattern already match that word (`tools/dump_fixtures.py`).
constexpr char kProgram[] = "eval_main";

constexpr char kUsage[] =
    "usage: nanochat_cli eval [options]   (alias: eval_main [options])\n"
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
    "   --vocab --padded-vocab --window-pattern --rope-base\n"
    "   --value-embedding <0|1>]\n";

// Builds the document-mode loader factory (docs/data.md).
DocumentSourceFactory MakeParquetFactory(const std::vector<std::string>& files,
                                         const std::string& column) {
  return
      [files, column](std::string* error) -> std::unique_ptr<DocumentSource> {
        return OpenParquetSource(files, column, 256, error);
      };
}

// Evaluates one split through `loader` and prints its bpb. Returns the result
// so a caller could compare splits.
float EvalSplit(Model* model, DataLoader* loader, const char* split,
                int steps) {
  const float bpb = EvalBpb(model, loader, steps);
  std::printf("eval_main: %s bpb %.6f\n", split, static_cast<double>(bpb));
  return bpb;
}

}  // namespace

int RunEval(int argc, char** argv) {
  Config model_config;
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

  cli::FlagParser flags(kProgram);
  flags.SetUsage(kUsage);
  // Both parquet lists accumulate, so a glob can be repeated or comma-joined.
  flags.AddValue("--train-parquet", [&train_parquet](std::string_view value) {
    cli::AppendCsv(value, &train_parquet);
    return true;
  });
  flags.AddValue("--val-parquet", [&val_parquet](std::string_view value) {
    cli::AppendCsv(value, &val_parquet);
    return true;
  });
  flags.AddString("--tokenizer", &tokenizer_path);
  flags.AddString("--text-column", &text_column);
  flags.AddInt("--tokenizer-threads", &tokenizer_threads);
  flags.AddInt("--buffer-docs", &document_buffer);
  flags.AddInt("--steps", &steps);
  flags.AddInt("--split-tokens", &split_tokens);
  flags.AddInt("--batch", &batch);
  flags.AddString("--model", &model_path);
  flags.AddU64("--seed", &seed);
  flags.AddModelFlags(&model_config);

  const int parse_status = flags.Parse(argc, argv);
  if (parse_status != -1) return parse_status;

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

  std::unique_ptr<Model> model = Model::Create(model_config);
  if (!model_path.empty()) {
    if (!Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "eval_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  }

  std::unique_ptr<Tokenizer> tokenizer = LoadTokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "eval_main: cannot load tokenizer %s\n",
                 tokenizer_path.c_str());
    return 1;
  }
  if (!train_parquet.empty()) {
    DataLoader loader(MakeParquetFactory(train_parquet, text_column),
                      tokenizer.get(), batch, model_config.seq_len, seed,
                      tokenizer_threads, document_buffer);
    EvalSplit(model.get(), &loader, "train", steps);
  }
  if (!val_parquet.empty()) {
    DataLoader loader(MakeParquetFactory(val_parquet, text_column),
                      tokenizer.get(), batch, model_config.seq_len, seed + 1,
                      tokenizer_threads, document_buffer);
    EvalSplit(model.get(), &loader, "val", steps);
  }
  return 0;
}

}  // namespace nanochat
