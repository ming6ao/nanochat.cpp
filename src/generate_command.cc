// `nanochat_cli generate` -- sample tokens from a checkpoint or from freshly
// seeded weights.
//
//   tools/nanochat eval -- <nanochat_cli> generate --model model.ckpt
//       --tokenizer tokenizer.nctoken --prompt "hello" --max-tokens 32
//
// The dispatcher applies the sandbox guard before it calls `RunGenerate`.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/model.h"
#include "nanochat/sampler.h"
#include "nanochat/tokenizer.h"
#include "src/cli.h"
#include "src/command.h"
#include "src/eval.h"
#include "src/train.h"

namespace nanochat {
namespace {

constexpr char kProgram[] = "generate_main";

constexpr char kUsage[] =
    "usage: nanochat_cli generate [options]   (alias: generate_main"
    " [options])\n"
    "  --model PATH           checkpoint to load\n"
    "  --tokenizer PATH       NCTOKEN1 artifact (for text prompts)\n"
    "  --prompt TEXT          text prompt (needs --tokenizer)\n"
    "  --text-file PATH       read the text prompt from a file\n"
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
    "   --vocab --padded-vocab --window-pattern --rope-base\n"
    "   --value-embedding <0|1>]\n";

std::vector<int> ParseTokenIds(std::string_view text) {
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

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream contents;
  contents << in.rdbuf();
  *out = contents.str();
  return true;
}

}  // namespace

int RunGenerate(int argc, char** argv) {
  Config model_config;
  std::string model_path;
  std::string tokenizer_path;
  std::string prompt_text;
  std::string prompt_file;
  std::string text_file;
  bool has_text_prompt = false;
  int max_tokens = 16;
  int num_samples = 1;
  int stop_id = -1;
  bool has_stop_id = false;
  int bos_id = -1;
  int top_k = 0;
  float temperature = 1.0f;
  std::uint64_t seed = 42;
  std::string out_path;

  cli::FlagParser flags(kProgram);
  flags.SetUsage(kUsage);
  flags.AddString("--model", &model_path);
  flags.AddString("--tokenizer", &tokenizer_path);
  // `--prompt` marks the text as text, so the tokenizer encodes it. `--tokens`
  // shares the field and stays a raw id list.
  flags.AddValue("--prompt",
                 [&prompt_text, &has_text_prompt](std::string_view value) {
                   prompt_text = value;
                   has_text_prompt = true;
                   return true;
                 });
  flags.AddString("--text-file", &text_file);
  flags.AddString("--tokens", &prompt_text);
  flags.AddString("--prompt-file", &prompt_file);
  flags.AddInt("--max-tokens", &max_tokens);
  flags.AddInt("--num-samples", &num_samples);
  flags.AddValue("--stop-id", [&stop_id, &has_stop_id](std::string_view value) {
    if (!cli::ParseInt(value, &stop_id)) return false;
    has_stop_id = true;
    return true;
  });
  flags.AddInt("--bos-id", &bos_id);
  flags.AddInt("--top-k", &top_k);
  flags.AddFloat("--temperature", &temperature);
  flags.AddU64("--seed", &seed);
  flags.AddString("--out", &out_path);
  flags.AddModelFlags(&model_config);

  const int parse_status = flags.Parse(argc, argv);
  if (parse_status != -1) return parse_status;

  std::unique_ptr<Tokenizer> tokenizer;
  if (!tokenizer_path.empty()) {
    tokenizer = LoadTokenizer(tokenizer_path);
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "generate_main: cannot load tokenizer %s\n",
                   tokenizer_path.c_str());
      return 1;
    }
  }
  // Stop a text row on the tokenizer's assistant-end token unless the caller
  // named a terminal id. A missing special leaves the id at -1.
  if (!has_stop_id && tokenizer != nullptr) {
    stop_id = tokenizer->SpecialId("<|assistant_end|>");
  }

  // A text prompt encodes to ids with a leading beginning-of-sequence id. The
  // BOS sits in the prompt, so `GenerateBatch` must not add a second one.
  std::vector<int> prompt;
  if (!text_file.empty()) {
    if (!ReadFile(text_file, &prompt_text)) {
      std::fprintf(stderr, "generate_main: cannot read %s\n",
                   text_file.c_str());
      return 1;
    }
    has_text_prompt = true;
  }
  if (has_text_prompt) {
    if (tokenizer == nullptr) {
      std::fprintf(stderr,
                   "generate_main: --prompt and --text-file need "
                   "--tokenizer\n");
      return 2;
    }
    const int prepend = bos_id >= 0 ? bos_id : tokenizer->bos_id();
    prompt = tokenizer->Encode(prompt_text, prepend);
    bos_id = -1;
  } else {
    if (!prompt_file.empty()) {
      if (!ReadFile(prompt_file, &prompt_text)) {
        std::fprintf(stderr, "generate_main: cannot read %s\n",
                     prompt_file.c_str());
        return 1;
      }
    }
    prompt = ParseTokenIds(prompt_text);
  }

  if (prompt.empty()) {
    std::fprintf(stderr,
                 "generate_main: provide a prompt with --prompt, --text-file, "
                 "--tokens, or --prompt-file\n");
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

  std::unique_ptr<Model> model = Model::Create(model_config);
  if (!model_path.empty()) {
    if (!Checkpointer::LoadModel(model.get(), model_path)) {
      std::fprintf(stderr, "generate_main: cannot load checkpoint %s\n",
                   model_path.c_str());
      return 1;
    }
  } else {
    model->InitWeights(seed);
  }

  GenerateParams gen;
  gen.num_samples = num_samples;
  gen.max_tokens = max_tokens;
  gen.temperature = temperature;
  gen.top_k = top_k;
  gen.seed = seed;
  gen.stop_id = stop_id;
  gen.bos_id = bos_id;

  std::vector<GeneratedSequence> rows;
  GenerateBatch(model.get(), prompt.data(), static_cast<int>(prompt.size()),
                gen, &rows);

  // Decode the generated rows through `TokenStreamDecoder`; a token can end
  // in the middle of a UTF-8 sequence, and the stream decoder holds the
  // incomplete suffix for the next token.
  std::vector<std::string> texts;
  if (tokenizer != nullptr) {
    DecodeGeneratedRows(*tokenizer, rows, &texts);
  }

  std::printf("generate_main: prompt");
  for (int id : prompt) std::printf(" %d", id);
  std::printf("\n");
  for (std::size_t r = 0; r < rows.size(); ++r) {
    std::printf("generate_main: sample %zu tokens", r);
    for (int id : rows[r].tokens) std::printf(" %d", id);
    std::printf("\n");
    if (r < texts.size()) {
      std::printf("generate_main: sample %zu text: ", r);
      std::fwrite(texts[r].data(), 1, texts[r].size(), stdout);
      std::printf("\n");
    }
  }

  if (!out_path.empty()) {
    std::ofstream out(out_path);
    if (!out) {
      std::fprintf(stderr, "generate_main: cannot write %s\n",
                   out_path.c_str());
      return 1;
    }
    for (const GeneratedSequence& row : rows) {
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

}  // namespace nanochat
