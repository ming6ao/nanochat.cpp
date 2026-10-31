// `sft_main` -- supervised fine-tuning on a packed SFT shard.
//
// Python writes the shard with `nanochat_cpp.sft.pack`. This binary reads the
// rows and runs the shared `Model::ForwardLoss`/`Backward`/`Optimizer::Step`
// path. The optimizer, the schedule, and the evaluation cadence follow the SFT
// recipe of `docs/sft-parity-plan.md` sections 5.1 to 5.4:
//
//   * the six AdamW groups and the Muon matrices of section 5.2, each rate
//     scaled once by `init_lr_frac = 0.8` (design decision D3), and no gradient
//     clip (D4);
//   * a warm start (`--warm-start PATH`): weights and optimizer moments from a
//     base checkpoint, fresh SFT rates, and a step counter that stays at 0
//     (D5). A `--resume PATH` continues the same run instead, moments and
//     stored step together;
//   * a learning rate driven by dataset progress
//   (`Scheduler::LrMultiplierForProgress`
//     over the running maximum of `approx_progress`), not by a step total fixed
//     before the run started (D1), while the Muon momentum keeps its 1-based
//     step (D2);
//   * gradient accumulation of
//     `total_batch_size / (device_batch_size * max_seq_len * world_size)`
//     micro-batches (section 5.1);
//   * bits-per-byte on the packed validation shard every `--eval-every` steps
//     (D6, section 5.4), and the ChatCORE cadence of the same section.
//
//   tools/nanochat train -- <sft_main> --rows path.nchsft --checkpoint out.nc
//       --warm-start base.ckpt --val-rows val.nchsft --token-bytes bytes.bin
//       --num-iterations 1000 --batch 2
//
// `--warm-start` and `--resume` both read the recipe block of the checkpoint
// (design decision D7): a file that records a batch shape and rates hands them
// to this run, and this run records its own recipe in the checkpoint it writes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/checkpoint_metadata.h"
#include "nanochat/data.h"
#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "nanochat/scheduler.h"
#include "src/cli.h"
#include "src/eval.h"
#include "src/optim_state.h"
#include "src/recipe.h"
#include "src/sft_shard.h"
#include "src/train.h"

namespace {

// The pinned SFT settings of `docs/sft-parity-plan.md` section 5, which a
// checkpoint without a recipe block falls back to. Every one of them repeats a
// `Recipe` initializer, so the two sources agree by construction.
constexpr float kSftEmbeddingLr = 0.3f;
constexpr float kSftUnembeddingLr = 0.004f;
constexpr float kSftMatrixLr = 0.02f;
constexpr float kSftScalarLr = 0.5f;
constexpr float kSftAdamEps = 1e-10f;
constexpr float kSftInitLrFrac = 0.8f;
constexpr float kSftWarmupRatio = 0.0f;
constexpr float kSftWarmdownRatio = 0.5f;
constexpr float kSftFinalLrFrac = 0.0f;
constexpr float kSftMuonMomentumSteps = 300.0f;
constexpr float kSftMuonMomentumStart = 0.85f;
constexpr float kSftMuonMomentumPeak = 0.95f;
constexpr float kSftMuonMomentumFinal = 0.95f;
// Section 5.4: the evaluation and ChatCORE intervals, and the token budget of
// one evaluation pass.
constexpr int kSftEvalEvery = 200;
constexpr int kSftEvalTokens = 40 * 524288;
constexpr int kSftChatcoreEvery = 200;

const char* kUsage =
    "usage: sft_main --rows PATH [--checkpoint PATH]\n"
    "       [--warm-start PATH | --resume PATH] [--val-rows PATH]\n"
    "       [--token-bytes PATH] [--num-iterations N] [--batch N]\n"
    "       [--grad-accum N] [--eval-every N] [--eval-tokens N]\n"
    "       [--chatcore-every N] [--seq N] [--seed N] [--device NAME]\n"
    "\n"
    "  --warm-start PATH  weights and optimizer moments from a base "
    "checkpoint.\n"
    "                     The rates are the fresh SFT ones and the step "
    "counter\n"
    "                     starts at 0 (design decision D5).\n"
    "  --resume PATH      weights, moments, and the stored step of this run.\n"
    "                     Not for a base checkpoint: its stored step breaks "
    "the\n"
    "                     SFT schedule, so use --warm-start for a base file.\n"
    "  --val-rows PATH    the packed validation shard the evaluation reads.\n"
    "  --token-bytes PATH one little-endian int32 per token id, the source "
    "bytes\n"
    "                     the id represents (bits-per-byte, design decision\n"
    "                     D6). Without a table every target counts one byte.\n"
    "  --grad-accum N     micro-batches per optimizer step. The default is "
    "the\n"
    "                     recipe's total_batch_size / (device_batch_size *\n"
    "                     max_seq_len * world_size) of section 5.1.\n"
    "  --eval-every N     bits-per-byte every N steps (default 200, 0 "
    "disables).\n"
    "  --eval-tokens N    the token budget of one evaluation pass (default\n"
    "                     40 * 524288); the pass never reads past one pass of\n"
    "                     the validation shard.\n"
    "  --chatcore-every N the ChatCORE interval (default 200, 0 disables). "
    "The\n"
    "                     C++ driver does not measure ChatCORE, so it reports\n"
    "                     the sentinel -1 at the cadence (section 5.4).\n";

// Reads one little-endian `int32` per token id from `path`. Returns false with
// a message on a missing, empty, or non-positive table.
bool LoadTokenBytes(const std::string& path, std::vector<int>* out,
                    std::string* error) {
  out->clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error != nullptr) *error = "cannot read the token-byte table: " + path;
    return false;
  }
  std::vector<unsigned char> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (raw.empty() || raw.size() % sizeof(std::int32_t) != 0) {
    if (error != nullptr) {
      *error =
          "the token-byte table is not a whole number of int32 values: " + path;
    }
    return false;
  }
  const std::size_t count = raw.size() / sizeof(std::int32_t);
  out->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::int32_t value = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(raw[i * 4]) |
        (static_cast<std::uint32_t>(raw[i * 4 + 1]) << 8) |
        (static_cast<std::uint32_t>(raw[i * 4 + 2]) << 16) |
        (static_cast<std::uint32_t>(raw[i * 4 + 3]) << 24));
    if (value < 0) {
      if (error != nullptr) {
        *error = "the token-byte table holds a negative length: " + path;
      }
      return false;
    }
    (*out)[i] = static_cast<int>(value);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("sft");

  nanochat::TrainConfig config;
  std::string rows_path;
  std::string warm_start_path;
  std::string val_rows_path;
  std::string token_bytes_path;
  // True when the caller pinned `--seq`; the shard owns the sequence length
  // otherwise.
  bool seq_pinned = false;
  // -1 means "not given": the pinned setting then applies, or the recipe.
  int grad_accum = -1;
  int eval_every = -1;
  int eval_tokens = -1;
  int chatcore_every = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      std::fputs(kUsage, stderr);
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &config.model)) {
        std::fprintf(stderr, "sft_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      if (flag == "--seq") seq_pinned = true;
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "sft_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--rows") {
      rows_path = value;
    } else if (flag == "--checkpoint") {
      config.checkpoint_path = value;
    } else if (flag == "--warm-start") {
      warm_start_path = value;
    } else if (flag == "--resume") {
      config.resume_path = value;
    } else if (flag == "--val-rows") {
      val_rows_path = value;
    } else if (flag == "--token-bytes") {
      token_bytes_path = value;
    } else if (flag == "--num-iterations") {
      config.num_iterations =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--batch") {
      config.batch =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--grad-accum") {
      grad_accum = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--eval-every") {
      eval_every = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--eval-tokens") {
      eval_tokens =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--chatcore-every") {
      chatcore_every =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--seed") {
      config.seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64
                                                                : config.seed;
    } else if (flag == "--device") {
      config.device_name = value;
    } else {
      std::fprintf(stderr, "sft_main: unknown flag %s\n", flag.c_str());
      return 2;
    }
  }

  if (rows_path.empty()) {
    std::fprintf(stderr, "sft_main: --rows is required\n");
    return 2;
  }
  // Design decision D5: a warm start and a resume are different loads. A run
  // that named both is ambiguous, and a resume of a base checkpoint would
  // continue the base step counter into the SFT schedule.
  if (!warm_start_path.empty() && !config.resume_path.empty()) {
    std::fprintf(
        stderr, "sft_main: --warm-start and --resume are mutually exclusive\n");
    return 2;
  }
  if (config.num_iterations <= 0) config.num_iterations = 1;

  // The recipe the run follows (design decision D7): the one its source
  // checkpoint recorded, when it recorded one, and the pinned section 5
  // settings otherwise. The read is metadata-only in the sense that the
  // optimizer section is not parsed; `Checkpointer::LoadModel` below re-reads
  // the file for the records themselves.
  const std::string source_path =
      !warm_start_path.empty() ? warm_start_path : config.resume_path;
  nanochat::Recipe recipe;
  recipe.device_batch_size = 0;  // "not recorded" until the shape is resolved
  recipe.total_batch_size = 0;
  recipe.embedding_lr = kSftEmbeddingLr;
  recipe.unembedding_lr = kSftUnembeddingLr;
  recipe.matrix_lr = kSftMatrixLr;
  recipe.scalar_lr = kSftScalarLr;
  recipe.adam_eps = kSftAdamEps;
  recipe.init_lr_frac = kSftInitLrFrac;
  recipe.warmup_ratio = kSftWarmupRatio;
  recipe.warmdown_ratio = kSftWarmdownRatio;
  recipe.final_lr_frac = kSftFinalLrFrac;
  recipe.muon_momentum_steps = kSftMuonMomentumSteps;
  recipe.muon_momentum_start = kSftMuonMomentumStart;
  recipe.muon_momentum_peak = kSftMuonMomentumPeak;
  recipe.muon_momentum_final = kSftMuonMomentumFinal;

  if (!source_path.empty()) {
    nanochat::Checkpoint source;
    std::string load_error;
    nanochat::CheckpointMetadata metadata;
    // A parameter-only read: this is only here for the metadata block. A legacy
    // `NCMDL001` file has no metadata at all, and `Checkpointer::LoadModel`
    // below still reads its parameters, so a failed read is a warning here and
    // the pinned recipe stays: no block is the same as an old block.
    if (source.Load(source_path, &load_error, /*with_optimizer_state=*/false)) {
      metadata = source.metadata();
    } else {
      std::fprintf(
          stderr,
          "sft_main: %s carries no metadata block (%s); the pinned SFT "
          "settings apply\n",
          source_path.c_str(), load_error.c_str());
    }
    if (!config.resume_path.empty() &&
        metadata.stage != nanochat::Stage::kSft) {
      // D5: a base checkpoint's stored step is the base run's, and adopting it
      // would start the SFT schedule somewhere in the middle of its warmdown.
      const char* recorded =
          metadata.stage == nanochat::Stage::kBase
              ? "the base stage"
              : (metadata.stage == nanochat::Stage::kRl ? "the rl stage"
                                                        : "no SFT stage");
      std::fprintf(stderr,
                   "sft_main: --resume needs an SFT checkpoint, and %s records "
                   "%s; use --warm-start for a base checkpoint\n",
                   config.resume_path.c_str(), recorded);
      return 2;
    }
    const nanochat::Recipe& inherited = metadata.recipe;
    // The header's sentinel: no recorded batch shape means no recorded recipe,
    // so this run keeps its own flags and the pinned settings.
    if (inherited.device_batch_size != 0 || inherited.total_batch_size != 0) {
      std::string recipe_error;
      if (!nanochat::RecipeUsable(inherited, &recipe_error)) {
        std::fprintf(stderr, "sft_main: %s: %s\n", source_path.c_str(),
                     recipe_error.c_str());
        return 2;
      }
      recipe = inherited;
      // An explicit `--batch` wins over the recorded device batch, so a caller
      // can shrink the shape of a recorded run without editing the file.
      if (config.batch <= 0) config.batch = recipe.device_batch_size;
    }
  }
  if (config.batch <= 0) config.batch = 1;

  // SFT optimizer defaults (section 5.2), then whatever the recipe recorded.
  nanochat::OptimizerConfig& optimizer_config = config.optimizer;
  optimizer_config.embedding_lr = recipe.embedding_lr;
  optimizer_config.unembedding_lr = recipe.unembedding_lr;
  optimizer_config.matrix_lr = recipe.matrix_lr;
  optimizer_config.scalar_lr = recipe.scalar_lr;
  optimizer_config.adam_eps = recipe.adam_eps;
  optimizer_config.init_lr_frac = recipe.init_lr_frac;
  // Design decision D4: the reference SFT loop has no gradient clip, so the C++
  // default of 1.0 is turned off here. Nothing else in the run rescales the
  // global norm.
  optimizer_config.clip = 0.0f;

  config.scheduler.warmup_steps = 0;
  config.scheduler.warmup_ratio = recipe.warmup_ratio;
  config.scheduler.warmdown_ratio = recipe.warmdown_ratio;
  config.scheduler.final_lr_frac = recipe.final_lr_frac;
  config.scheduler.muon_momentum_warmup_steps = recipe.muon_momentum_steps;
  config.scheduler.muon_momentum_start = recipe.muon_momentum_start;
  config.scheduler.muon_momentum_peak = recipe.muon_momentum_peak;
  config.scheduler.muon_momentum_final = recipe.muon_momentum_final;
  config.scheduler.num_iterations = config.num_iterations;

  // Read the shard before building the model. The shard carries the owned
  // sequence length, which the SFT mix uses verbatim.
  nanochat::SftShard shard;
  std::string shard_error;
  if (!nanochat::LoadSftShard(rows_path, &shard, &shard_error)) {
    std::fprintf(stderr, "sft_main: %s\n", shard_error.c_str());
    return 2;
  }
  // The shard owns the sequence length (section 5.1: the row capacity is
  // `max_seq_len + 1`, so the packed rows are the authority). A caller that
  // pinned `--seq` has to agree with it, because a model at another length
  // would read the packed rows at the wrong stride.
  const int shard_seq = shard.seq;
  if (seq_pinned && config.model.seq_len != shard_seq) {
    std::fprintf(stderr, "sft_main: --seq is %d, the shard seq is %d\n",
                 config.model.seq_len, shard_seq);
    return 2;
  }
  config.seq = shard_seq;
  config.model.seq_len = shard_seq;
  const nanochat::Config model_config = config.model;

  // The gradient accumulation of section 5.1:
  //   total_batch_size / (device_batch_size * max_seq_len * world_size).
  // An explicit `--grad-accum` wins, and a recipe without a recorded total
  // batch size leaves the accumulation at one micro-batch.
  const int world_size = config.world_size > 0 ? config.world_size : 1;
  if (grad_accum < 0) {
    grad_accum = 1;
    if (recipe.total_batch_size > 0 && config.batch > 0) {
      const std::int64_t per_step =
          static_cast<std::int64_t>(config.batch) * shard_seq * world_size;
      if (per_step > 0) {
        grad_accum = static_cast<int>(recipe.total_batch_size / per_step);
      }
    }
  }
  if (grad_accum <= 0) grad_accum = 1;
  config.grad_accum = grad_accum;
  // The recipe this run records is the shape it actually ran (design decision
  // D7), not the shape it inherited: a later finetune continues from what this
  // run did, and `total_batch_size / (device_batch_size * max_seq_len *
  // world_size)` then reproduces this run's accumulation.
  recipe.device_batch_size = config.batch;
  recipe.total_batch_size = config.batch * shard_seq * grad_accum * world_size;

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  nanochat::SchedulerConfig scheduler_config = config.scheduler;
  if (scheduler_config.num_iterations <= 0) {
    scheduler_config.num_iterations =
        config.num_iterations > 0 ? config.num_iterations : 1;
  }
  config.scheduler = scheduler_config;
  auto scheduler = std::make_unique<nanochat::Scheduler>(scheduler_config);
  auto optimizer =
      nanochat::CreateOptimizer(model.get(), config.optimizer, *scheduler);

  // The warm start and the resume (design decision D5). Both restore the
  // weights and the optimizer moments; only the resume restores the step,
  // because a warm start begins the fresh SFT schedule at 0. `nullptr` for the
  // step is what keeps the caller's counter (`src/train.cc` leaves it alone).
  int step = 0;
  if (!warm_start_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), optimizer.get(),
                                           nullptr, warm_start_path)) {
      std::fprintf(stderr, "sft_main: cannot warm-start from %s\n",
                   warm_start_path.c_str());
      return 2;
    }
    step = 0;
  } else if (!config.resume_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), optimizer.get(), &step,
                                           config.resume_path)) {
      std::fprintf(stderr, "sft_main: cannot resume from %s\n",
                   config.resume_path.c_str());
      return 2;
    }
  } else {
    model->InitWeights(config.seed);
  }

  // The evaluation source (design decision D6, section 5.4). The pass reads one
  // pass of the packed validation rows, so the step count is the rows the shard
  // holds, cut down to the token budget of section 5.4.
  nanochat::SftShard val_shard;
  bool have_val = false;
  if (!val_rows_path.empty()) {
    std::string val_error;
    if (!nanochat::LoadSftShard(val_rows_path, &val_shard, &val_error)) {
      std::fprintf(stderr, "sft_main: %s\n", val_error.c_str());
      return 2;
    }
    if (val_shard.seq != shard_seq) {
      std::fprintf(stderr,
                   "sft_main: the validation shard has seq %d, the run %d\n",
                   val_shard.seq, shard_seq);
      return 2;
    }
    have_val = true;
  }
  if (eval_every < 0) eval_every = kSftEvalEvery;
  if (eval_tokens < 0) eval_tokens = kSftEvalTokens;
  if (chatcore_every < 0) chatcore_every = kSftChatcoreEvery;

  std::vector<int> token_bytes;
  if (have_val) {
    if (!token_bytes_path.empty()) {
      std::string table_error;
      if (!LoadTokenBytes(token_bytes_path, &token_bytes, &table_error)) {
        std::fprintf(stderr, "sft_main: %s\n", table_error.c_str());
        return 2;
      }
    } else {
      // The `EvalBpb` fallback: with no byte table every target counts one
      // byte.
      token_bytes.assign(
          static_cast<std::size_t>(std::max(model_config.vocab_size, 1)), 1);
      std::fprintf(stderr,
                   "sft_main: no --token-bytes table, so every target counts "
                   "one byte\n");
    }
  }

  // The driver's own step counter. `Optimizer::Step` is 1-based, and the
  // schedules that stay step-indexed evaluate at `step - 1`, which is the
  // reference's 0-based `it` (design decision D2): the Muon momentum of step
  // `k` is `MuonMomentum(k)`, computed at `k - 1`, so a warm start that resets
  // the counter to 0 restarts the momentum ramp exactly as the reference does.
  // No code changes for D2.
  const int first_step = step > 0 ? step + 1 : 1;
  const int target_steps =
      std::max(first_step - 1 + config.num_iterations, config.num_iterations);

  const std::size_t micro_values =
      static_cast<std::size_t>(config.batch) * shard_seq;
  std::vector<int> tokens(micro_values);
  std::vector<int> targets(micro_values);
  int row_cursor = 0;
  int epoch = 1;
  std::int64_t rows_consumed = 0;
  // The running maximum of the reference `approx_progress` (section 5.2, and
  // `PackedRowStream.progress` in python/nanochat_cpp/sft_data.py).
  float progress = 0.0f;
  float last_multiplier = 1.0f;
  int last_step = target_steps;

  for (int current = first_step; current <= target_steps; ++current) {
    float last_micro_loss = 0.0f;
    model->ZeroGrad();
    for (int micro = 0; micro < config.grad_accum; ++micro) {
      for (int r = 0; r < config.batch; ++r) {
        if (row_cursor >= shard.rows) {
          row_cursor = 0;
          ++epoch;
        }
        const std::size_t source =
            static_cast<std::size_t>(row_cursor) * shard_seq;
        const std::size_t destination = static_cast<std::size_t>(r) * shard_seq;
        std::copy_n(shard.tokens.begin() + source, shard_seq,
                    tokens.begin() + destination);
        std::copy_n(shard.targets.begin() + source, shard_seq,
                    targets.begin() + destination);
        ++row_cursor;
        ++rows_consumed;
      }
      // The loss is the mean over the batch's valid targets, so the
      // accumulation scale of `src/train.h` is 1 / grad_accum: the micro-batch
      // sums then give the mean over the global batch.
      last_micro_loss = model->ForwardLoss(tokens.data(), targets.data(),
                                           config.batch, shard_seq);
      model->BackwardAccumulate(1.0f / static_cast<float>(config.grad_accum));
    }

    // The progress stream (design decision D1): the reference advances progress
    // once per batch boundary and keeps its running maximum. The iteration
    // count when the run knows its total, and the fraction of the shard it
    // consumed otherwise. The count is relative to this invocation's first
    // step, so a resumed run walks its own warmdown instead of starting past
    // the end of the previous one, and the value is capped at 1 so the
    // multiplier can never go negative.
    const int iterations_done = current - first_step + 1;
    const float approx_progress =
        config.num_iterations > 0
            ? static_cast<float>(iterations_done) /
                  static_cast<float>(config.num_iterations)
            : static_cast<float>(rows_consumed) /
                  static_cast<float>(std::max(shard.rows, 1));
    progress = std::min(1.0f, std::max(progress, approx_progress));
    last_multiplier = scheduler->LrMultiplierForProgress(progress);
    nanochat::SetOptimizerProgressMultiplier(optimizer.get(), last_multiplier);
    optimizer->Step(current);
    last_step = current;

    std::fprintf(stderr,
                 "sft: step %d epoch %d progress %.6f lr_multiplier %.6f "
                 "momentum %.6f loss %.6f\n",
                 current, epoch, static_cast<double>(progress),
                 static_cast<double>(last_multiplier),
                 static_cast<double>(scheduler->MuonMomentum(current)),
                 static_cast<double>(last_micro_loss));

    // Section 5.4: the bits-per-byte evaluation every `eval_every` steps, plus
    // the final step, which is the evaluation the reference always runs.
    const bool final_step = current == target_steps;
    if (have_val && eval_every > 0 &&
        (current % eval_every == 0 || final_step)) {
      int eval_steps = val_shard.rows / std::max(config.batch, 1);
      if (eval_tokens > 0) {
        const std::int64_t tokens_per_step =
            static_cast<std::int64_t>(config.batch) * shard_seq;
        if (tokens_per_step > 0) {
          const std::int64_t budget = eval_tokens / tokens_per_step;
          if (budget < eval_steps) eval_steps = static_cast<int>(budget);
        }
      }
      if (eval_steps < 1) eval_steps = 1;
      nanochat::PackedBpb bpb;
      std::string eval_error;
      if (nanochat::EvaluatePackedRows(model.get(), val_shard, token_bytes,
                                       config.batch, eval_steps, &bpb,
                                       &eval_error)) {
        std::fprintf(stderr,
                     "sft: eval step %d bpb %.6f valid_targets %lld "
                     "target_bytes %lld\n",
                     current, static_cast<double>(bpb.bpb),
                     static_cast<long long>(bpb.valid_targets),
                     static_cast<long long>(bpb.target_bytes));
      } else {
        std::fprintf(stderr, "sft: eval step %d failed: %s\n", current,
                     eval_error.c_str());
        return 2;
      }
    }
    // The ChatCORE cadence of section 5.4 is recorded, not measured: ChatCORE
    // needs the chat bundle and the task datasets, so the driver reports the
    // same sentinel the loop fixture pins (-1, `chatcore_measured = 0`) and
    // phase G measures it on the real run.
    if (chatcore_every > 0 && (current % chatcore_every == 0 || final_step)) {
      std::fprintf(stderr, "sft: chatcore step %d -1\n", current);
    }
  }

  if (!config.checkpoint_path.empty()) {
    // The finetune records its stage, so a reader can tell an SFT checkpoint
    // from a base or RL one without a side file (stage A.6, open decision O3),
    // and its recipe, so a later finetune inherits the batch shape and the
    // rates (design decision D7).
    nanochat::CheckpointSaveOptions options;
    options.stage = nanochat::Stage::kSft;
    options.step = last_step > 0 ? static_cast<std::uint64_t>(last_step) : 0;
    options.world_size = world_size;
    options.recipe = recipe;
    std::string save_error;
    if (!nanochat::Checkpointer::SaveModel(*model, optimizer.get(), options,
                                           config.checkpoint_path,
                                           &save_error)) {
      std::fprintf(stderr, "sft_main: %s\n", save_error.c_str());
      return 2;
    }
    std::fprintf(stderr, "sft: wrote %s at step %d\n",
                 config.checkpoint_path.c_str(), last_step);
  }
  return 0;
}
