// `nanochat_cli train` -- the training command. The actual loop lives in
// `src/train.cc`.
//
//   tools/nanochat train -- <nanochat_cli> train --train-parquet
//   'data/*.parquet'
//       --tokenizer tokenizer.nctoken ...
//
// Every run must go through `tools/nanochat`; see docs/sandbox.md. The
// dispatcher applies the sandbox guard before it calls `RunTrain`.

#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli.h"
#include "src/command.h"
#include "src/train.h"

namespace nanochat {
namespace {

// The prefix of every diagnostic. It stays `train_main` because a reader and a
// log pattern already match that word (`tools/dump_fixtures.py`).
constexpr char kProgram[] = "train_main";

constexpr char kUsage[] =
    "usage: nanochat_cli train [options]   (alias: train_main [options])\n"
    "  --preset NAME          config preset (track3)\n"
    "  --train-parquet GLOB   training parquet (repeatable, csv)\n"
    "  --val-parquet GLOB     validation parquet (repeatable, csv)\n"
    "  --tokenizer PATH       NCTOKEN1 artifact (document mode)\n"
    "  --text-column NAME     parquet text column (default text)\n"
    "  --tokenizer-threads N  encode workers (default 4)\n"
    "  --buffer-docs N        encoded documents in memory (default 1000)\n"
    "  --batch N              batch size\n"
    "  --grad-accum N         micro-batches per optimizer step\n"
    "  --num-iterations N     optimizer steps\n"
    "  --log-every N          log cadence\n"
    "  --eval-every N         evaluation cadence (0 disables)\n"
    "  --save-every N         checkpoint cadence (0 disables)\n"
    "  --eval-steps N         validation batches per evaluation\n"
    "  --seed N               weight init and data seed\n"
    "  --checkpoint PATH      write the final checkpoint here\n"
    "  --resume PATH          resume weights from here\n"
    "  --log PATH             also write the run log here\n"
    "  --device NAME          device name for MFU\n"
    "  --hw-check MODE        startup gate: auto (default), on, or off\n"
    "  --rank N               data parallel rank (default 0)\n"
    "  --world-size N         data parallel group size (default 1)\n"
    "  --master HOST          rank 0 host for the gradient sync\n"
    "  --port N               rank 0 port for the gradient sync\n"
    "  [optimizer flags: --embedding-lr --unembedding-lr --matrix-lr\n"
    "   --scalar-lr --weight-decay --weight-decay-base --clip\n"
    "   --adam-eps --muon-ns-steps --muon-beta2]\n"
    "  [schedule flags: --warmup-steps --warmdown-ratio --final-lr-frac\n"
    "   --muon-momentum-warmup-steps --muon-momentum-start\n"
    "   --muon-momentum-peak --muon-momentum-final]\n"
    "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
    "   --vocab --padded-vocab --window-pattern --rope-base\n"
    "   --value-embedding <0|1>]\n";

}  // namespace

int RunTrain(int argc, char** argv) {
  TrainConfig config;
  // The launcher's environment contract (docs/distributed.md section 10,
  // section 4). A flag overrides the variable, because the parser runs after
  // this. The names follow `torchrun`, so the command also runs under
  // `torchrun`, `mpirun`, or a shell loop.
  config.rank = cli::EnvInt("NANOCHAT_RANK", config.rank);
  config.world_size = cli::EnvInt("NANOCHAT_WORLD_SIZE", config.world_size);
  config.master = cli::EnvString("NANOCHAT_MASTER_ADDR", config.master);
  config.port = cli::EnvInt("NANOCHAT_MASTER_PORT", config.port);

  cli::FlagParser flags(kProgram);
  flags.SetUsage(kUsage);
  // A preset rewrites the model, the optimizer, and the schedule, so it runs at
  // its position on the command line: a later flag still overrides it.
  flags.AddValue("--preset", [&config](std::string_view value) {
    if (value != "track3") {
      std::fprintf(stderr, "train_main: unknown preset %.*s\n",
                   static_cast<int>(value.size()), value.data());
      return false;
    }
    TrainConfig preset = TrackThreeBaseline();
    config.model = preset.model;
    config.optimizer = preset.optimizer;
    config.scheduler = preset.scheduler;
    return true;
  });
  // Both parquet lists accumulate, so a glob can be repeated or comma-joined.
  flags.AddValue("--train-parquet", [&config](std::string_view value) {
    cli::AppendCsv(value, &config.train_parquet);
    return true;
  });
  flags.AddValue("--val-parquet", [&config](std::string_view value) {
    cli::AppendCsv(value, &config.val_parquet);
    return true;
  });
  flags.AddString("--tokenizer", &config.tokenizer_path);
  flags.AddString("--text-column", &config.text_column);
  flags.AddInt("--tokenizer-threads", &config.tokenizer_threads);
  flags.AddInt("--buffer-docs", &config.document_buffer);
  flags.AddInt("--batch", &config.batch);
  flags.AddInt("--grad-accum", &config.grad_accum);
  flags.AddInt("--num-iterations", &config.num_iterations);
  flags.AddInt("--log-every", &config.log_every);
  flags.AddInt("--eval-every", &config.eval_every);
  flags.AddInt("--save-every", &config.save_every);
  flags.AddInt("--eval-steps", &config.eval_steps);
  flags.AddU64("--seed", &config.seed);
  flags.AddString("--checkpoint", &config.checkpoint_path);
  flags.AddString("--resume", &config.resume_path);
  flags.AddString("--log", &config.log_path);
  flags.AddString("--device", &config.device_name);
  flags.AddValue("--hw-check", [&config](std::string_view value) {
    return ParseHwCheck(value, &config.hw_check);
  });
  flags.AddInt("--rank", &config.rank);
  flags.AddInt("--world-size", &config.world_size);
  flags.AddString("--master", &config.master);
  flags.AddInt("--port", &config.port);
  flags.AddFloat("--embedding-lr", &config.optimizer.embedding_lr);
  flags.AddFloat("--unembedding-lr", &config.optimizer.unembedding_lr);
  flags.AddFloat("--matrix-lr", &config.optimizer.matrix_lr);
  flags.AddFloat("--scalar-lr", &config.optimizer.scalar_lr);
  flags.AddFloat("--weight-decay", &config.optimizer.weight_decay);
  flags.AddInt("--muon-ns-steps", &config.optimizer.muon_ns_steps);
  flags.AddFloat("--muon-beta2", &config.optimizer.muon_beta2);
  flags.AddFloat("--adam-eps", &config.optimizer.adam_eps);
  flags.AddFloat("--clip", &config.optimizer.clip);
  flags.AddInt("--warmup-steps", &config.scheduler.warmup_steps);
  flags.AddFloat("--warmdown-ratio", &config.scheduler.warmdown_ratio);
  flags.AddFloat("--final-lr-frac", &config.scheduler.final_lr_frac);
  flags.AddFloat("--weight-decay-base", &config.scheduler.weight_decay_base);
  flags.AddFloat("--muon-momentum-warmup-steps",
                 &config.scheduler.muon_momentum_warmup_steps);
  flags.AddFloat("--muon-momentum-start",
                 &config.scheduler.muon_momentum_start);
  flags.AddFloat("--muon-momentum-peak", &config.scheduler.muon_momentum_peak);
  flags.AddFloat("--muon-momentum-final",
                 &config.scheduler.muon_momentum_final);
  flags.AddModelFlags(&config.model);

  const int parse_status = flags.Parse(argc, argv);
  if (parse_status != -1) return parse_status;

  if (config.train_parquet.empty()) {
    std::fprintf(stderr, "train_main: --train-parquet is required\n");
    return 2;
  }
  if (config.tokenizer_path.empty()) {
    std::fprintf(stderr, "train_main: --tokenizer is required\n");
    return 2;
  }
  if (config.num_iterations <= 0 || config.batch <= 0 ||
      config.grad_accum <= 0) {
    std::fprintf(stderr,
                 "train_main: --num-iterations, --batch, and --grad-accum "
                 "must be > 0\n");
    return 2;
  }
  if (config.world_size < 1 || config.rank < 0 ||
      config.rank >= config.world_size) {
    std::fprintf(stderr, "train_main: --rank must be in [0, --world-size)\n");
    return 2;
  }
  if (config.port <= 0 || config.port > 65535) {
    std::fprintf(stderr, "train_main: --port must be in [1, 65535]\n");
    return 2;
  }

  TrainLoop loop(std::move(config));
  const float loss = loop.Run();
  std::printf("train_main: final loss %.6f at step %d\n",
              static_cast<double>(loss), loop.last_step());
  return 0;
}

}  // namespace nanochat
