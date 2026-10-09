// The persistent reinforcement-learning worker (docs/rl-notebook.md section 4).
// It holds the model and the optimizer across steps and speaks the pipe
// protocol documented in `src/rl_worker.h`. The RL step is the shared
// `nanochat::RlStep`, so the worker and the file-driven `rl_step` binary cannot
// drift apart (docs/training-seam.md section 6.4).

#include "src/rl_worker.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "nanochat/scheduler.h"
#include "src/cli.h"
#include "src/rl.h"
#include "src/train.h"

namespace nanochat {

namespace {

// --- Message parsing -------------------------------------------------------

struct Message {
  std::string type;
  std::map<std::string, std::string> fields;
};

bool ParseMessage(const std::string& line, Message* message) {
  std::istringstream stream(line);
  if (!(stream >> message->type)) return false;
  std::string token;
  while (stream >> token) {
    const std::size_t equals = token.find('=');
    if (equals == std::string::npos) {
      message->fields.emplace(token, std::string());
    } else {
      message->fields.emplace(token.substr(0, equals),
                              token.substr(equals + 1));
    }
  }
  return true;
}

const std::string* Field(const Message& message, const char* key) {
  const auto it = message.fields.find(key);
  return it == message.fields.end() ? nullptr : &it->second;
}

bool GetInt(const Message& message, const char* key, int* out) {
  const std::string* value = Field(message, key);
  return value != nullptr && cli::ParseInt(value->c_str(), out);
}

bool GetIntDefault(const Message& message, const char* key, int fallback,
                   int* out) {
  if (Field(message, key) == nullptr) {
    *out = fallback;
    return true;
  }
  return GetInt(message, key, out);
}

bool GetFloatDefault(const Message& message, const char* key, float fallback,
                     float* out) {
  const std::string* value = Field(message, key);
  if (value == nullptr) {
    *out = fallback;
    return true;
  }
  return cli::ParseFloat(value->c_str(), out);
}

bool GetU64Default(const Message& message, const char* key,
                   std::uint64_t fallback, std::uint64_t* out) {
  const std::string* value = Field(message, key);
  if (value == nullptr) {
    *out = fallback;
    return true;
  }
  return cli::ParseU64(value->c_str(), out);
}

bool GetIntList(const Message& message, const char* key,
                std::vector<int>* out) {
  const std::string* value = Field(message, key);
  if (value == nullptr) return false;
  out->clear();
  for (const std::string& part : cli::SplitCsv(*value)) {
    int parsed = 0;
    if (!cli::ParseInt(part.c_str(), &parsed)) return false;
    out->push_back(parsed);
  }
  return true;
}

bool GetFloatList(const Message& message, const char* key,
                  std::vector<float>* out) {
  const std::string* value = Field(message, key);
  if (value == nullptr) return false;
  out->clear();
  for (const std::string& part : cli::SplitCsv(*value)) {
    float parsed = 0.0f;
    if (!cli::ParseFloat(part.c_str(), &parsed)) return false;
    out->push_back(parsed);
  }
  return true;
}

bool GetString(const Message& message, const char* key, std::string* out) {
  const std::string* value = Field(message, key);
  if (value == nullptr || value->empty()) return false;
  *out = *value;
  return true;
}

// --- Message serialization -------------------------------------------------

std::string FormatFloat(float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
  return std::string(buffer);
}

std::string JoinInts(const std::vector<int>& values) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out.push_back(',');
    out += std::to_string(values[i]);
  }
  return out;
}

std::string JoinBytes(const std::vector<std::uint8_t>& values) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out.push_back(',');
    out += std::to_string(static_cast<unsigned>(values[i]));
  }
  return out;
}

// Keeps an error message to a single frame: the protocol is one line per
// message, so any whitespace becomes an underscore.
std::string SanitizeError(const std::string& text) {
  std::string out;
  for (char c : text) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      out.push_back('_');
    } else {
      out.push_back(c);
    }
  }
  return out.empty() ? std::string("unknown") : out;
}

std::string Error(const std::string& type, const std::string& text) {
  return type + " ok=0 error=" + SanitizeError(text);
}

void Emit(std::ostream& out, const std::string& line) {
  out << line << '\n';
  out.flush();
}

// --- Request handlers ------------------------------------------------------

std::string HandleRollout(RlWorker* worker, const Message& message) {
  const std::string type = "rollout";
  int prompt_len = 0;
  int num_prompts = 0;
  int num_samples = 0;
  int max_tokens = 0;
  int top_k = 0;
  int stop_id = 0;
  int bos_id = 0;
  if (!GetInt(message, "prompt_len", &prompt_len) || prompt_len <= 0) {
    return Error(type, "prompt_len_must_be_positive");
  }
  if (!GetInt(message, "num_prompts", &num_prompts) || num_prompts <= 0) {
    return Error(type, "num_prompts_must_be_positive");
  }
  if (!GetIntDefault(message, "num_samples", 1, &num_samples) ||
      num_samples <= 0) {
    return Error(type, "num_samples_must_be_positive");
  }
  if (!GetInt(message, "max_tokens", &max_tokens) || max_tokens <= 0) {
    return Error(type, "max_tokens_must_be_positive");
  }
  if (!GetIntDefault(message, "top_k", 0, &top_k) || top_k < 0) {
    return Error(type, "top_k_must_be_nonnegative");
  }
  if (!GetIntDefault(message, "stop_id", -1, &stop_id)) {
    return Error(type, "bad_stop_id");
  }
  if (!GetIntDefault(message, "bos_id", -1, &bos_id)) {
    return Error(type, "bad_bos_id");
  }
  float temperature = 1.0f;
  std::uint64_t seed = 42;
  if (!GetFloatDefault(message, "temperature", 1.0f, &temperature)) {
    return Error(type, "bad_temperature");
  }
  if (!GetU64Default(message, "seed", 42, &seed)) {
    return Error(type, "bad_seed");
  }

  std::vector<int> prompts;
  if (!GetIntList(message, "prompts", &prompts)) {
    return Error(type, "missing_prompts");
  }
  const std::int64_t expected =
      static_cast<std::int64_t>(num_prompts) * prompt_len;
  if (static_cast<std::int64_t>(prompts.size()) != expected) {
    return Error(type, "prompts_size_mismatch");
  }

  GenerateParams params;
  params.num_samples = num_samples;
  params.max_tokens = max_tokens;
  params.temperature = temperature;
  params.top_k = top_k;
  params.seed = seed;
  params.stop_id = stop_id;
  params.bos_id = bos_id;

  std::vector<int> stop_ids;
  if (Field(message, "stop_ids") != nullptr) {
    if (!GetIntList(message, "stop_ids", &stop_ids) ||
        static_cast<int>(stop_ids.size()) != num_samples) {
      return Error(type, "stop_ids_size_mismatch");
    }
    params.stop_ids = stop_ids.data();
  }

  std::vector<GeneratedSequence> rows;
  worker->Generate(prompts, num_prompts, prompt_len, params, &rows);

  std::vector<int> lengths;
  std::vector<std::uint8_t> masks;
  std::vector<int> tokens;
  lengths.reserve(rows.size());
  for (const GeneratedSequence& row : rows) {
    lengths.push_back(static_cast<int>(row.tokens.size()));
    tokens.insert(tokens.end(), row.tokens.begin(), row.tokens.end());
    masks.insert(masks.end(), row.mask.begin(), row.mask.end());
  }

  std::string response = type;
  response += " ok=1 num_prompts=" + std::to_string(num_prompts);
  response += " num_samples=" + std::to_string(num_samples);
  response += " rows=" + std::to_string(rows.size());
  response += " lengths=" + JoinInts(lengths);
  response += " masks=" + JoinBytes(masks);
  response += " tokens=" + JoinInts(tokens);
  return response;
}

std::string HandleAdvantage(RlWorker* worker, const Message& message) {
  const std::string type = "advantage";
  int batch = 0;
  int seq = 0;
  int num_passes = 0;
  int examples_per_rank = 0;
  if (!GetInt(message, "batch", &batch) || batch <= 0) {
    return Error(type, "batch_must_be_positive");
  }
  if (!GetInt(message, "seq", &seq) || seq <= 0) {
    return Error(type, "seq_must_be_positive");
  }
  if (!GetInt(message, "num_passes", &num_passes) || num_passes <= 0) {
    return Error(type, "num_passes_must_be_positive");
  }
  if (!GetInt(message, "examples_per_rank", &examples_per_rank) ||
      examples_per_rank <= 0) {
    return Error(type, "examples_per_rank_must_be_positive");
  }

  std::vector<int> tokens;
  std::vector<int> targets;
  std::vector<float> advantages;
  if (!GetIntList(message, "tokens", &tokens)) {
    return Error(type, "missing_tokens");
  }
  if (!GetIntList(message, "targets", &targets)) {
    return Error(type, "missing_targets");
  }
  if (!GetFloatList(message, "advantages", &advantages)) {
    return Error(type, "missing_advantages");
  }
  const std::int64_t expected =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  if (static_cast<std::int64_t>(tokens.size()) != expected ||
      static_cast<std::int64_t>(targets.size()) != expected ||
      static_cast<std::int64_t>(advantages.size()) != expected) {
    return Error(type, "batch_seq_size_mismatch");
  }

  const RlWorkerResult result =
      worker->Step(tokens.data(), targets.data(), advantages.data(), batch, seq,
                   num_passes, examples_per_rank);

  std::string response = type;
  response += " ok=1 step=" + std::to_string(result.step);
  response += " loss=" + FormatFloat(result.loss);
  response += " grad_norm=" + FormatFloat(result.grad_norm);
  response += " valid_targets=" + std::to_string(result.valid_targets);
  return response;
}

std::string HandleSave(RlWorker* worker, const Message& message) {
  const std::string type = "save";
  std::string path;
  if (!GetString(message, "path", &path)) {
    return Error(type, "missing_path");
  }
  std::string error;
  if (!worker->SaveCheckpoint(path, &error)) {
    return Error(type, error);
  }
  return type + " ok=1";
}

}  // namespace

// --- RlWorker --------------------------------------------------------------

RlWorker::RlWorker(const Config& model_config,
                   const OptimizerConfig& optimizer_config,
                   const SchedulerConfig& scheduler_config)
    : config_(model_config) {
  model_ = Model::Create(config_);
  if (model_ == nullptr) return;
  const Scheduler scheduler(scheduler_config);
  optimizer_ = CreateOptimizer(model_.get(), optimizer_config, scheduler);
}

RlWorker::~RlWorker() = default;

RlWorkerResult RlWorker::Step(const int* tokens, const int* targets,
                              const float* advantages, int batch, int seq,
                              int num_passes, int examples_per_rank) {
  const std::int64_t rows =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(seq);
  int valid = 0;
  for (std::int64_t i = 0; i < rows; ++i) {
    if (targets[i] != -1) ++valid;
  }

  const int next_step = step_ + 1;
  const float loss =
      RlStep(model_.get(), optimizer_.get(), tokens, targets, advantages, batch,
             seq, num_passes, examples_per_rank, next_step);
  step_ = next_step;

  RlWorkerResult result;
  result.loss = loss;
  result.grad_norm = optimizer_->GradNorm();
  result.valid_targets = valid;
  result.step = step_;
  return result;
}

void RlWorker::Generate(const std::vector<int>& prompts, int num_prompts,
                        int prompt_len, const GenerateParams& params,
                        std::vector<GeneratedSequence>* out) {
  // The multi-prompt rollout is one call: the same `num_samples` rows per
  // prompt, with `params` (including any per-sample `stop_ids`) reused for
  // every prompt. It shares the generation compute with the file-driven
  // binary and the C ABI (docs/rl-notebook.md section 5 phase 5).
  GenerateMultiPrompt(model_.get(), prompts.data(), num_prompts, prompt_len,
                      params, nullptr, out);
}

bool RlWorker::SaveCheckpoint(const std::string& path, std::string* error) {
  if (!Checkpointer::SaveModel(*model_, *optimizer_, step_, path)) {
    *error = "cannot write checkpoint " + path;
    return false;
  }
  return true;
}

int RlWorker::Serve(std::istream& in, std::ostream& out) {
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Message message;
    if (!ParseMessage(line, &message)) {
      Emit(out, Error("error", "bad_message"));
      continue;
    }
    if (message.type == "quit") {
      Emit(out, "quit ok=1");
      break;
    }
    std::string response;
    try {
      if (message.type == "rollout") {
        response = HandleRollout(this, message);
      } else if (message.type == "advantage") {
        response = HandleAdvantage(this, message);
      } else if (message.type == "save") {
        response = HandleSave(this, message);
      } else if (message.type == "ping") {
        response = "pong ok=1";
      } else {
        response = Error(message.type, "unknown_message");
      }
    } catch (const std::exception& error) {
      response = Error(message.type, error.what());
    }
    Emit(out, response);
  }
  return in.bad() ? 1 : 0;
}

}  // namespace nanochat
