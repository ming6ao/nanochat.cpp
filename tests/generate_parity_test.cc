// tests/generate_parity_test.cc — the generation-parity gate.
//
// Reads a fixture produced by `tools/dump_generate_fixture.py` and drives
// `nanochat.cpp`'s `GenerateBatch` in greedy mode from the recorded parameters
// and prompt. The fixture is data, so the same source links the reference CPU
// backend by default and the CUDA backend under `--config=cuda`, and every
// device read goes through `kernels::Memcpy`.
//
// The comparison is deliberate and discrete: `GenerateBatch` must reproduce the
// reference token ids exactly and the per-position mask (1 for a sampled id, 0
// for a prompt id). The fixture's parameters are trained for a few hundred
// steps before the trace is recorded, so the classifier is peaked and the
// argmax is stable against the fp32 roundoff that separates the two
// implementations. Beyond the exact match, the test also checks the two
// behaviors the batched API adds on top of a single greedy loop: several rows
// decode identically, and a terminal id truncates its row before it is
// appended.
//
// The source builds as `//tests:generate_parity_test` (CPU) and
// `//tests:generate_parity_cuda_test` (GPU, `gpu` tag).

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/model.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::GeneratedSequence;
using nanochat::GenerateParams;
using nanochat::Model;
using nanochat::ParamView;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

template <typename T = ComputeType>
void StoreFloat(T* dst, float value) {
  if constexpr (std::is_same_v<T, float>) {
    *dst = value;
  } else {
    *dst = nanochat::Fp16FromFloat(value);
  }
}

Config ConfigFromFixture(const Fixture& fixture) {
  Config config;
  config.num_layers =
      static_cast<int>(fixture.Get("config/layers").scalar_int());
  config.num_heads = static_cast<int>(fixture.Get("config/heads").scalar_int());
  config.num_kv_heads =
      static_cast<int>(fixture.Get("config/kv_heads").scalar_int());
  config.hidden_dim = static_cast<int>(fixture.Get("config/embd").scalar_int());
  config.seq_len = static_cast<int>(fixture.Get("config/seq").scalar_int());
  config.vocab_size =
      static_cast<int>(fixture.Get("config/vocab").scalar_int());
  config.padded_vocab_size =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  config.rope_base = 100000.0f;
  const Tensor& pattern = fixture.Get("config/window_pattern");
  config.window_pattern =
      std::string(pattern.data, pattern.data + pattern.numel());
  return config;
}

// Loads `param/<name>` into every model parameter. The names match PyTorch's
// `named_parameters()` (proven by the oracle test), so the mapping is direct.
void LoadParameters(const Fixture& fixture, Model* model) {
  for (const ParamView& view : model->params()) {
    const std::string record = std::string("param/") + view.name;
    if (!fixture.Has(record)) {
      Fail("fixture is missing parameter record '" + record + "'");
      continue;
    }
    const Tensor& source = fixture.Get(record);
    if (source.numel() != view.count) {
      Fail(Format("parameter '%s' count mismatch: model %lld fixture %lld",
                  view.name, static_cast<long long>(view.count),
                  static_cast<long long>(source.numel())));
      continue;
    }
    std::vector<ComputeType> host(static_cast<std::size_t>(view.count));
    for (std::int64_t i = 0; i < view.count; ++i) {
      StoreFloat(host.data() + i, source.f32()[i]);
    }
    nanochat::kernels::Memcpy(
        view.value, host.data(),
        static_cast<std::size_t>(view.count) * sizeof(ComputeType),
        CopyDir::kHostToDevice);
  }
}

std::vector<int> ReadInts(const Tensor& tensor) {
  const std::int32_t* data = tensor.i32();
  return std::vector<int>(data, data + tensor.numel());
}

// Builds the expected `GenerateBatch` row for a full reference trace truncated
// at the first occurrence of `stop_id` in the generated span. A negative stop
// id (or one that never appears) keeps the whole generated span. The prompt and
// generated spans are the fixture's `prompt/tokens` and `generate/tokens`
//; the mask is 0 for the prompt and 1 for the kept generated ids.
struct ExpectedRow {
  std::vector<int> tokens;
  std::vector<std::uint8_t> mask;
};

ExpectedRow ExpectedFromStop(const std::vector<int>& prompt,
                             const std::vector<int>& generated, int stop_id) {
  ExpectedRow row;
  row.tokens = prompt;
  row.mask.assign(prompt.size(), 0);
  for (int token : generated) {
    if (stop_id >= 0 && token == stop_id) break;
    row.tokens.push_back(token);
    row.mask.push_back(1);
  }
  return row;
}

void CheckRow(const std::string& label, const GeneratedSequence& got,
              const ExpectedRow& want) {
  if (got.tokens != want.tokens) {
    std::string detail;
    for (std::size_t i = 0; i < std::max(got.tokens.size(), want.tokens.size());
         ++i) {
      const int a = i < got.tokens.size() ? got.tokens[i] : -1;
      const int b = i < want.tokens.size() ? want.tokens[i] : -1;
      if (a != b) {
        detail =
            Format(" first difference at index %zu (got %d want %d)", i, a, b);
        break;
      }
    }
    Fail(label + ": token ids differ from the reference" + detail);
  }
  if (got.mask != want.mask) {
    Fail(label + ": per-position mask differs from the reference");
  }
  if (got.tokens.size() != got.mask.size()) {
    Fail(label + ": mask and tokens are not aligned");
  }
}

void Run(const std::string& path) {
  const Fixture fixture = Fixture::Load(path);
  if (fixture.Get("config/version").scalar_int() != 1) {
    throw std::runtime_error("unsupported generation-parity fixture version");
  }

  const Config config = ConfigFromFixture(fixture);
  const int prompt_len =
      static_cast<int>(fixture.Get("config/prompt_len").scalar_int());
  const int max_tokens =
      static_cast<int>(fixture.Get("config/max_tokens").scalar_int());
  const int bos_id =
      static_cast<int>(fixture.Get("config/bos_id").scalar_int());

  std::unique_ptr<Model> model = Model::Create(config);
  LoadParameters(fixture, model.get());

  const std::vector<int> prompt = ReadInts(fixture.Get("prompt/tokens"));
  const std::vector<int> effective = ReadInts(fixture.Get("generate/tokens"));
  const Tensor& mask_tensor = fixture.Get("generate/mask");
  const std::vector<std::uint8_t> mask(mask_tensor.data,
                                       mask_tensor.data + mask_tensor.numel());
  if (static_cast<int>(prompt.size()) != prompt_len) {
    Fail(Format("fixture prompt length %zu disagrees with config/prompt_len %d",
                prompt.size(), prompt_len));
  }
  if (effective.size() < prompt.size()) {
    throw std::runtime_error(
        "fixture generated span is shorter than the prompt");
  }
  const std::vector<int> generated(effective.begin() + prompt.size(),
                                   effective.end());

  // --- exact greedy parity: one row, no stop, no prepended bos. -------------
  GenerateParams greedy;
  greedy.num_samples = 1;
  greedy.max_tokens = max_tokens;
  greedy.temperature = 0.0f;  // argmax
  greedy.top_k = 0;
  greedy.seed =
      static_cast<std::uint64_t>(fixture.Get("config/seed").scalar_int());
  greedy.stop_id = -1;
  greedy.bos_id = bos_id;
  greedy.stop_ids = nullptr;

  {
    std::vector<GeneratedSequence> rows;
    nanochat::GenerateBatch(model.get(), prompt.data(), prompt_len, greedy,
                            &rows);
    if (rows.size() != 1) {
      Fail(Format("greedy batch produced %zu rows, want 1", rows.size()));
    } else {
      ExpectedRow want;
      want.tokens = effective;
      want.mask = mask;
      CheckRow("greedy batch", rows[0], want);
    }
  }

  // --- several greedy rows must decode identically. -------------------------
  {
    GenerateParams batched = greedy;
    batched.num_samples = 3;
    std::vector<GeneratedSequence> rows;
    nanochat::GenerateBatch(model.get(), prompt.data(), prompt_len, batched,
                            &rows);
    if (rows.size() != 3) {
      Fail(Format("determinism batch produced %zu rows, want 3", rows.size()));
    } else {
      for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].tokens != rows[0].tokens || rows[r].mask != rows[0].mask) {
          Fail("greedy rows are not identical at row " + std::to_string(r));
        }
      }
    }
  }

  // --- a global terminal id truncates the row before it is appended. --------
  const int last_generated = generated.empty() ? -1 : generated.back();
  if (last_generated >= 0) {
    GenerateParams stopped = greedy;
    stopped.stop_id = last_generated;
    std::vector<GeneratedSequence> rows;
    nanochat::GenerateBatch(model.get(), prompt.data(), prompt_len, stopped,
                            &rows);
    if (rows.size() != 1) {
      Fail(Format("global-stop batch produced %zu rows, want 1", rows.size()));
    } else {
      CheckRow("global-stop batch", rows[0],
               ExpectedFromStop(prompt, generated, last_generated));
    }
  }

  // --- per-row terminal ids let the rows diverge independently. -------------
  if (generated.size() >= 3) {
    const std::vector<int> stops = {generated[1], generated[0], generated[2]};
    GenerateParams per_row = greedy;
    per_row.num_samples = 3;
    per_row.stop_ids = stops.data();
    std::vector<GeneratedSequence> rows;
    nanochat::GenerateBatch(model.get(), prompt.data(), prompt_len, per_row,
                            &rows);
    if (rows.size() != 3) {
      Fail(Format("per-row-stop batch produced %zu rows, want 3", rows.size()));
    } else {
      for (std::size_t r = 0; r < rows.size(); ++r) {
        CheckRow("per-row-stop batch row " + std::to_string(r), rows[r],
                 ExpectedFromStop(prompt, generated, stops[r]));
      }
    }
  }

  std::printf(
      "generate_parity: matched %zu reference tokens over %d greedy steps\n",
      effective.size(), max_tokens);
}

std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/generate_parity.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/generate_parity.bin");
  }
  candidates.emplace_back("tests/data/generate_parity.bin");
  candidates.emplace_back("../tests/data/generate_parity.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "generation-parity fixture not found; pass it as argv[1]");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(LocateFixture(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "generate_parity: %s\n", error.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "generate_parity: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("generate_parity: all checks passed\n");
  return 0;
}
