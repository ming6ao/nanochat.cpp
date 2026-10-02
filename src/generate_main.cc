// `generate_main` — a small prefill/decode CLI. Because the BPE tokenizer is
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

#include "cli.h"
#include "nanochat/model.h"
#include "nanochat/sampler.h"
#include "nanochat/sandbox.h"
#include "train.h"

namespace {

void Usage() {
  std::fprintf(stderr,
               "usage: generate_main [options]\n"
               "  --model PATH           checkpoint to load\n"
               "  --tokens a,b,c         prompt token ids\n"
               "  --prompt-file PATH     read prompt token ids from a file\n"
               "  --max-tokens N         number of tokens to generate\n"
               "  --temperature F        sampling temperature (0 = greedy)\n"
               "  --top-k N              top-k filter (0 disables)\n"
               "  --seed N               weight init and sampling seed\n"
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
  int top_k = 0;
  float temperature = 1.0f;
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
    } else if (flag == "--top-k") {
      top_k = nanochat::cli::ParseInt(value, &parsed_int) ? parsed_int : 0;
    } else if (flag == "--temperature") {
      temperature =
          nanochat::cli::ParseFloat(value, &parsed_float) ? parsed_float : 0.0f;
    } else if (flag == "--seed") {
      seed = nanochat::cli::ParseU64(value, &parsed_u64) ? parsed_u64 : 0;
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

  const int capacity = static_cast<int>(prompt.size()) + max_tokens + 2;
  std::unique_ptr<nanochat::KvCache> kv =
      nanochat::CreateKvCache(model_config, capacity);

  const int prefill_len = static_cast<int>(prompt.size()) - 1;
  if (prefill_len > 0) {
    nanochat::Prefill(model.get(), prompt.data(), prefill_len, kv.get());
  }

  nanochat::SampleParams params;
  params.temperature = temperature;
  params.top_k = top_k;
  params.seed = seed;

  std::vector<int> generated;
  int next = prompt.back();
  for (int i = 0; i < max_tokens; ++i) {
    next = nanochat::Decode(model.get(), next, kv.get(), params);
    if (next < 0) break;
    generated.push_back(next);
  }

  std::printf("generate_main: prompt");
  for (int id : prompt) std::printf(" %d", id);
  std::printf("\ngenerate_main: output");
  for (int id : generated) std::printf(" %d", id);
  std::printf("\n");
  return 0;
}
