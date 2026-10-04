// `generate_main` -- a small prefill/decode CLI. Because the BPE tokenizer is
// out of scope in this tree, the prompt is given as token ids (inline or in a
// file). Weights come from a `Checkpoint` container written by `train_main`.
//
//   tools/nanochat gpu -- <generate_main> --model model.ckpt --tokens 1,2,3

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/sampler.h"
#include "nanochat/sandbox.h"
#include "src/cli.h"
#include "src/train.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: generate_main [options]\n"
               "  --model PATH           checkpoint to load\n"
               "  --tokens a,b,c         prompt token ids\n"
               "  --prompt-file PATH     read prompt token ids from a file\n"
               "  --max-tokens N         number of tokens to generate\n"
               "  --num-samples N        number of rows to sample (default 1)\n"
               "  --temperature F        sampling temperature (0 = greedy)\n"
               "  --top-k N              top-k filter (0 disables)\n"
               "  --stop-id N            terminal token id (-1 disables)\n"
               "  --bos-id N             prepended prompt token (-1 disables)\n"
               "  --seed N               weight init and sampling seed\n"
               "  --out PATH             write tokens and mask per sample\n"
               "  [model flags: --layers --heads --kv-heads --hidden --seq\n"
               "   --vocab --padded-vocab --window-pattern --rope-base]\n");
}

std::vector<int> ParseTokenIds(const std::string& text) {
  std::vector<int> ids;
  std::string current;
  auto flush = [&]() {
    if (current.empty()) return;
    char* end = nullptr;
    const long value = std::strtol(current.c_str(), &end, 10);
    if (end != current.c_str() && *end == '\0') {
      ids.push_back(static_cast<int>(value));
    }
    current.clear();
  };
  for (char c : text) {
    if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
      flush();
    } else {
      current.push_back(c);
    }
  }
  flush();
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("generate");

  nanochat::Config model_config;
  std::string model_path;
  std::string prompt_text;
  std::string prompt_file;
  int max_tokens = 16;
  int num_samples = 1;
  int stop_id = -1;
  int bos_id = -1;
  int top_k = 0;
  float temperature = 1.0f;
  std::uint64_t seed = 42;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      Usage();
      return 0;
    }
    if (nanochat::cli::IsModelFlag(flag)) {
      if (i + 1 >= argc ||
          !nanochat::cli::ApplyModelFlag(flag, argv[i + 1], &model_config)) {
        std::fprintf(stderr, "generate_main: bad value for %s\n", flag.c_str());
        return 2;
      }
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "generate_main: %s needs a value\n", flag.c_str());
      return 2;
    }
    const char* value = argv[++i];
    int parsed_int = 0;
    float parsed_float = 0.0f;
    std::uint64_t parsed_u64 = 0;
    if (flag == "--model") {
      model_path = value;
    } else if (flag == "--tokens") {
      prompt_text = value;
    } else if (flag == "--prompt-file") {
      prompt_file = value;
    } else if (flag == "--max-tokens") {
      max_tokens =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : -1;
    } else if (flag == "--num-samples") {
      num_samples =
          nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--stop-id") {
      stop_id = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : -1;
    } else if (flag == "--bos-id") {
      bos_id = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : -1;
    } else if (flag == "--top-k") {
      top_k = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--temperature") {
      temperature =
          nanochat::cli::ParseFloat(value, &parsed_float) ? parsed_float : 0.0f;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
    } else if (flag == "--out") {
      out_path = value;
    } else {
      std::fprintf(stderr, "generate_main: unknown flag %s\n", flag.c_str());
      Usage();
      return 2;
    }
  }

  if (!prompt_file.empty()) {
    std::ifstream in(prompt_file);
    if (!in) {
      std::fprintf(stderr, "generate_main: cannot read %s\n",
                   prompt_file.c_str());
      return 1;
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    prompt_text = contents.str();
  }

  const std::vector<int> prompt = ParseTokenIds(prompt_text);
  if (prompt.empty()) {
    std::fprintf(stderr,
                 "generate_main: provide a prompt with --tokens or "
                 "--prompt-file\n");
    return 2;
  }
  if (max_tokens <= 0) {
    std::fprintf(stderr, "generate_main: --max-tokens must be > 0\n");
    return 2;
  }
  if (num_samples <= 0) {
    std::fprintf(stderr, "generate_main: --num-samples must be > 0\n");
    return 2;
  }

  std::unique_ptr<nanochat::Model> model =
      nanochat::Model::Create(model_config);
  if (!model_path.empty()) {
    if (!nanochat::Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "generate_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  } else {
    model->InitWeights(seed);
  }

  nanochat::GenerateParams gen;
  gen.num_samples = num_samples;
  gen.max_tokens = max_tokens;
  gen.temperature = temperature;
  gen.top_k = top_k;
  gen.seed = seed;
  gen.stop_id = stop_id;
  gen.bos_id = bos_id;

  std::vector<nanochat::GeneratedSequence> rows;
  nanochat::GenerateBatch(model.get(), prompt.data(),
                          static_cast<int>(prompt.size()), gen, &rows);

  std::printf("generate_main: prompt");
  for (int id : prompt) std::printf(" %d", id);
  std::printf("\n");
  for (std::size_t r = 0; r < rows.size(); ++r) {
    std::printf("generate_main: sample %zu tokens", r);
    for (int id : rows[r].tokens) std::printf(" %d", id);
    std::printf("\n");
  }

  if (!out_path.empty()) {
    std::ofstream out(out_path);
    if (!out) {
      std::fprintf(stderr, "generate_main: cannot write %s\n",
                   out_path.c_str());
      return 1;
    }
    for (const nanochat::GeneratedSequence& row : rows) {
      for (std::size_t i = 0; i < row.tokens.size(); ++i) {
        if (i != 0) out << ' ';
        out << row.tokens[i];
      }
      out << " |";
      for (std::uint8_t bit : row.mask) {
        out << ' ' << static_cast<int>(bit);
      }
      out << '\n';
    }
  }

  return 0;
}
