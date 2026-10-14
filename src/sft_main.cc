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

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/optim.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/train.h"

namespace {

struct PackedShard {
  std::uint64_t rows = 0;
  std::uint64_t seq = 0;
};

bool ReadHeader(std::ifstream& in, PackedShard* shard) {
  char magic[8];
  in.read(magic, 8);
  if (!in) return false;
  if (std::string(magic, 8) != "NCSFT001") return false;
  in.read(reinterpret_cast<char*>(&shard->seq), sizeof(shard->seq));
  if (!in) return false;
  in.read(reinterpret_cast<char*>(&shard->rows), sizeof(shard->rows));
  if (!in) return false;
  return true;
}

}  // namespace

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
      config.seed = nanochat::cli::ParseU64(value, &parsed_u64)
                        ? parsed_u64
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

  // Read the shard header before building the model. The shard carries the
  // owned sequence length, which the SFT mix uses verbatim.
  std::ifstream in(rows_path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "sft_main: cannot open %s\n", rows_path.c_str());
    return 2;
  }
  PackedShard shard;
  if (!ReadHeader(in, &shard)) {
    std::fprintf(stderr, "sft_main: bad shard header\n");
    return 2;
  }
  const int shard_seq = static_cast<int>(shard.seq);
  if (config.seq <= 0) config.seq = shard_seq;
  if (config.seq != shard_seq) {
    std::fprintf(stderr, "sft_main: seq mismatch (%d vs %d)\n", config.seq,
                 shard_seq);
    return 2;
  }
  if (shard.rows == 0) {
    std::fprintf(stderr, "sft_main: empty shard\n");
    return 2;
  }

  config.seq = shard_seq;
  if (config.model.seq_len <= 0) config.model.seq_len = shard_seq;

  nanochat::Config model_config = config.model;
  if (model_config.seq_len <= 0) model_config.seq_len = shard_seq;

  std::unique_ptr<nanochat::Model> model = nanochat::Model::Create(model_config);
  nanochat::SchedulerConfig scheduler_config = config.scheduler;
  if (scheduler_config.num_iterations <= 0) {
    scheduler_config.num_iterations =
        config.num_iterations > 0 ? config.num_iterations : 1;
  }
  config.scheduler = scheduler_config;
  auto scheduler = std::make_unique<nanochat::Scheduler>(scheduler_config);
  auto optimizer = nanochat::CreateOptimizer(model.get(), config.optimizer,
                                            *scheduler);

  int step = 0;
  if (!config.resume_path.empty()) {
    nanochat::Checkpointer::LoadModel(model.get(), optimizer.get(), &step,
                                      config.resume_path);
  } else {
    model->InitWeights(config.seed);
  }
  const int first_step = step > 0 ? step + 1 : 1;
  const int target_steps = std::max(first_step - 1 + config.num_iterations,
                                   config.num_iterations);

  const std::uint64_t rows_total = shard.rows;
  std::uint64_t row_cursor = 0;
  for (int current = first_step; current <= target_steps; ++current) {
    if (row_cursor >= rows_total) row_cursor = 0;  // cycle epochs
    std::vector<int> tokens;
    std::vector<int> targets;
    tokens.reserve(static_cast<std::size_t>(config.batch) * shard_seq);
    targets.reserve(static_cast<std::size_t>(config.batch) * shard_seq);

    int rows_in_batch = 0;
    for (int r = 0; r < config.batch; ++r) {
      if (row_cursor >= rows_total) {
        row_cursor = 0;
        if (rows_in_batch == 0) break;
      }
      if (row_cursor >= rows_total) break;
      const std::uint64_t row_offset = row_cursor;
      in.clear();
      in.seekg(static_cast<std::streamoff>(
                   8 * 3 + row_offset * shard_seq * 2 * sizeof(std::int32_t)),
               std::ios::beg);
      // Header size: magic 8 + seq 8 + rows 8 = 24.
      std::vector<std::int32_t> row_tokens(shard_seq);
      std::vector<std::int32_t> row_targets(shard_seq);
      in.read(reinterpret_cast<char*>(row_tokens.data()),
              static_cast<std::streamsize>(shard_seq * sizeof(std::int32_t)));
      in.read(reinterpret_cast<char*>(row_targets.data()),
              static_cast<std::streamsize>(shard_seq * sizeof(std::int32_t)));
      if (!in) {
        std::fprintf(stderr, "sft_main: read failure at row %llu\n",
                     static_cast<unsigned long long>(row_offset));
        return 2;
      }
      for (int t = 0; t < shard_seq; ++t) {
        tokens.push_back(static_cast<int>(row_tokens[t]));
        targets.push_back(static_cast<int>(row_targets[t]));
      }
      ++row_cursor;
      ++rows_in_batch;
    }
    if (rows_in_batch == 0) break;
    if (rows_in_batch < config.batch) {
      const int missing = config.batch - rows_in_batch;
      (void)missing;
    }

    optimizer->ZeroGrad();
    const float loss =
        model->ForwardLoss(tokens.data(), targets.data(), rows_in_batch, shard_seq);
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
