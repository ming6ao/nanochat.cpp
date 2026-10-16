// `sft_main` -- supervised fine-tuning on a packed SFT shard.
//
// Python writes the shard with `nanochat_cpp.sft.pack`. This binary reads the
// rows and runs the shared `Model::ForwardLoss`/`Backward`/`Optimizer::Step`
// path. The optimizer and learning-rate settings follow the SFT recipe in
// `docs/post-training.md` section 5.3.
//
//   tools/nanochat train -- <sft_main> --rows path.nchsft --checkpoint out.nc
//       --num-iterations 1000 --batch 2
//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/sft_shard.h"
#include "src/train.h"

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("sft");

  nanochat::TrainConfig config;
  std::string rows_path;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      std::fprintf(stderr, "usage: sft_main --rows PATH [--checkpoint PATH]\n");
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &config.model)) {
        std::fprintf(stderr, "sft_main: bad value for %s\n", flag.c_str());
        return 2;
      }
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
    } else if (flag == "--resume") {
      config.resume_path = value;
    } else if (flag == "--num-iterations") {
      config.num_iterations =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--batch") {
      config.batch =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--seq") {
      config.seq = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
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
  if (config.num_iterations <= 0) config.num_iterations = 1;
  if (config.batch <= 0) config.batch = 1;

  // SFT optimizer defaults. Flags above only cover core knobs; the per-group
  // betas and decays live in `src/optim.cc`.
  nanochat::OptimizerConfig& optimizer_config = config.optimizer;
  optimizer_config.embedding_lr = 0.3f;
  optimizer_config.unembedding_lr = 0.004f;
  optimizer_config.matrix_lr = 0.02f;
  optimizer_config.scalar_lr = 0.5f;
  optimizer_config.adam_eps = 1e-10f;
  config.scheduler.warmup_steps = 0;
  config.scheduler.warmdown_ratio = 0.5f;
  config.scheduler.final_lr_frac = 0.0f;
  config.scheduler.muon_momentum_warmup_steps = 300.0f;
  config.scheduler.muon_momentum_start = 0.85f;
  config.scheduler.muon_momentum_peak = 0.95f;
  config.scheduler.muon_momentum_final = 0.95f;

  // Read the shard before building the model. The shard carries the owned
  // sequence length, which the SFT mix uses verbatim.
  nanochat::SftShard shard;
  std::string shard_error;
  if (!nanochat::LoadSftShard(rows_path, &shard, &shard_error)) {
    std::fprintf(stderr, "sft_main: %s\n", shard_error.c_str());
    return 2;
  }
  const int shard_seq = shard.seq;
  if (config.seq <= 0) config.seq = shard_seq;
  if (config.seq != shard_seq) {
    std::fprintf(stderr, "sft_main: seq mismatch (%d vs %d)\n", config.seq,
                 shard_seq);
    return 2;
  }

  config.seq = shard_seq;
  if (config.model.seq_len <= 0) config.model.seq_len = shard_seq;

  nanochat::Config model_config = config.model;
  if (model_config.seq_len <= 0) model_config.seq_len = shard_seq;

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

  int step = 0;
  if (!config.resume_path.empty()) {
    nanochat::Checkpointer::LoadModel(model.get(), optimizer.get(), &step,
                                      config.resume_path);
  } else {
    model->InitWeights(config.seed);
  }
  const int first_step = step > 0 ? step + 1 : 1;
  const int target_steps =
      std::max(first_step - 1 + config.num_iterations, config.num_iterations);

  std::vector<int> tokens(static_cast<std::size_t>(config.batch) * shard_seq);
  std::vector<int> targets(tokens.size());
  int row_cursor = 0;
  for (int current = first_step; current <= target_steps; ++current) {
    for (int r = 0; r < config.batch; ++r) {
      if (row_cursor >= shard.rows) row_cursor = 0;
      const std::size_t source =
          static_cast<std::size_t>(row_cursor) * shard_seq;
      const std::size_t destination = static_cast<std::size_t>(r) * shard_seq;
      std::copy_n(shard.tokens.begin() + source, shard_seq,
                  tokens.begin() + destination);
      std::copy_n(shard.targets.begin() + source, shard_seq,
                  targets.begin() + destination);
      ++row_cursor;
    }

    model->ZeroGrad();
    const float loss = model->ForwardLoss(tokens.data(), targets.data(),
                                          config.batch, shard_seq);
    model->Backward();
    optimizer->Step(current);

    std::fprintf(stderr, "sft: step %d loss %.6f\n", current,
                 static_cast<double>(loss));
  }

  if (!config.checkpoint_path.empty()) {
    nanochat::Checkpointer::SaveModel(*model, *optimizer, target_steps,
                                      config.checkpoint_path);
  }
  return 0;
}
