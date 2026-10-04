// tests/score_parity_test.cc -- the CORE scoring-parity gate.
//
// Reads the synthetic CORE fixture produced by
// `tools/dump_eval_fixture.py` and runs the real `score_main` binary on it,
// then compares `score_main`'s result fixture against the PyTorch reference
// recorded in the same file:
//
//   * per-position NLL (`result/nll`) within an fp32 tolerance;
//   * per-position argmax (`result/argmax`) exactly;
//   * the focused logits (`result/focus_logits`) after applying the model's
//     logit softcap, since `ScoreBatch` returns the raw pre-softcap values.
//
// The fixture carries the model configuration and every parameter, so the test
// rebuilds the reference model, writes a checkpoint, and feeds both to
// `score_main`. The committed fixture is hermetic: it needs no network, no
// `eval_bundle.zip`, and no tokenizer. Its input covers the three CORE
// candidate-span shapes (multiple_choice, schema, language_modeling) plus a
// categorical focus set, which the test asserts before comparing the scores.
//
// `score_main` is launched as a subprocess from the runfiles tree, so this test
// exercises the binary's fixture input/output, not just `ScoreBatch`.

#include <sys/wait.h>
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
#include "src/train.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::Checkpointer;
using nanochat::ComputeType;
using nanochat::Config;
using nanochat::CopyDir;
using nanochat::Model;
using nanochat::ParamView;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

// fp32 roundoff between the reference and the CPU backend is ~1e-8 for this
// tiny graph; 1e-4 leaves ample headroom without hiding a real divergence.
constexpr double kTolerance = 1e-4;

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

void CheckClosed(double got, double want, const std::string& what) {
  if (std::fabs(got - want) > kTolerance * (1.0 + std::fabs(want))) {
    Fail(Format("%s: got %.9g want %.9g", what.c_str(), got, want));
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

template <typename T = ComputeType>
void StoreFloat(T* dst, float value) {
  if constexpr (std::is_same_v<T, float>) {
    *dst = value;
  } else {
    *dst = nanochat::Fp16FromFloat(value);
  }
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

std::string Getenv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : std::string();
}

// Locates `tests/data/core_eval.bin` in the runfiles tree, then relative to the
// source tree for a manual run.
std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  const std::string src_dir = Getenv("TEST_SRCDIR");
  const std::string workspace = Getenv("TEST_WORKSPACE");
  if (!src_dir.empty()) {
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace +
                           "/tests/data/core_eval.bin");
    }
    candidates.push_back(src_dir + "/_main/tests/data/core_eval.bin");
  }
  candidates.emplace_back("tests/data/core_eval.bin");
  candidates.emplace_back("../tests/data/core_eval.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "score fixture not found; pass tests/data/core_eval.bin as argv[1]");
}

// Locates the `score_main` binary next to the test in the runfiles tree. The
// `//src:score_main` data dependency is what puts it there.
std::string LocateScoreMain() {
  std::vector<std::string> candidates;
  const std::string override = Getenv("NANOCHAT_SCORE_MAIN");
  if (!override.empty()) candidates.push_back(override);
  const std::string src_dir = Getenv("TEST_SRCDIR");
  const std::string workspace = Getenv("TEST_WORKSPACE");
  if (!src_dir.empty()) {
    if (!workspace.empty()) {
      candidates.push_back(src_dir + "/" + workspace + "/src/score_main");
    }
    candidates.push_back(src_dir + "/_main/src/score_main");
  }
  candidates.emplace_back("bazel-bin/src/score_main");
  candidates.emplace_back("../bazel-bin/src/score_main");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error(
      "score_main not found in the runfiles tree; add //src:score_main to the "
      "test data or set NANOCHAT_SCORE_MAIN");
}

std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::string TempDir() {
  const std::string dir = Getenv("TEST_TMPDIR");
  return dir.empty() ? std::string("/tmp") : dir;
}

// The candidate spans recorded by the dumper. The CORE reduction scores
// prediction positions `[start - 1, end - 1)`; the test asserts the three
// shapes are present and that every span is well formed.
struct Span {
  int start = 0;
  int end = 0;
  int length = 0;
};

std::vector<Span> ReadSpans(const Fixture& fixture, int batch) {
  const std::vector<int> spans = ReadInts(fixture.Get("input/spans"));
  const std::vector<int> lengths = ReadInts(fixture.Get("input/lengths"));
  std::vector<Span> out(static_cast<std::size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    out[static_cast<std::size_t>(b)].start = spans[2 * b];
    out[static_cast<std::size_t>(b)].end = spans[2 * b + 1];
    out[static_cast<std::size_t>(b)].length = lengths[b];
  }
  return out;
}

void CheckSpanShapes(const Fixture& fixture, int batch) {
  if (batch < 5) {
    Fail(Format("the fixture has %d rows, expected the three span shapes",
                batch));
    return;
  }
  const std::vector<Span> spans = ReadSpans(fixture, batch);
  for (int b = 0; b < batch; ++b) {
    const Span& span = spans[static_cast<std::size_t>(b)];
    if (span.start < 0 || span.start > span.end || span.end > span.length) {
      Fail(Format("row %d has an invalid span [%d, %d) over length %d", b,
                  span.start, span.end, span.length));
    }
    if (span.start != span.end && span.start < 1) {
      Fail(Format("row %d has a non-empty span starting before token 1", b));
    }
  }
  // multiple_choice: a shared prefix (equal starts), different continuations.
  if (!(spans[0].start == spans[1].start && spans[0].end != spans[1].end)) {
    Fail("rows 0-1 do not have the multiple_choice shape");
  }
  // schema: a shared suffix, so the candidate spans have equal length while
  // the contexts (and therefore the ends) differ.
  if (!((spans[2].end - spans[2].start) == (spans[3].end - spans[3].start) &&
        spans[2].end != spans[3].end)) {
    Fail("rows 2-3 do not have the schema shape");
  }
  // language_modeling: the prompt alone is a strict prefix of the prompt with
  // the continuation.
  if (!(spans[4].start < spans[4].end)) {
    Fail("row 4 does not have the language_modeling shape");
  }
  const std::vector<int> focus_positions =
      ReadInts(fixture.Get("input/focus_positions"));
  int focused = 0;
  for (int position : focus_positions) {
    if (position >= 0) ++focused;
  }
  if (focused == 0) Fail("the fixture carries no focus request");
}

// `score_main` returns raw pre-softcap logits; the reference stores the
// post-softcap values the model produces, so reapply the monotonic softcap.
float ApplySoftcap(float raw, float cap) { return cap * std::tanh(raw / cap); }

void CompareResults(const Fixture& reference, const Fixture& actual,
                    float softcap, int batch, int seq) {
  const Tensor& want_nll = reference.Get("result/nll");
  const Tensor& got_nll = actual.Get("result/nll");
  const Tensor& want_argmax = reference.Get("result/argmax");
  const Tensor& got_argmax = actual.Get("result/argmax");
  if (got_nll.numel() != want_nll.numel() ||
      got_argmax.numel() != want_argmax.numel()) {
    Fail("score_main wrote a result of the wrong shape");
    return;
  }
  for (std::int64_t i = 0; i < want_nll.numel(); ++i) {
    const int row = static_cast<int>(i / seq);
    const int position = static_cast<int>(i % seq);
    CheckClosed(got_nll.f32()[i], want_nll.f32()[i],
                Format("nll row %d position %d", row, position));
    if (got_argmax.i32()[i] != want_argmax.i32()[i]) {
      Fail(Format("argmax row %d position %d: got %d want %d", row, position,
                  got_argmax.i32()[i], want_argmax.i32()[i]));
    }
  }

  const Tensor& want_offsets = reference.Get("result/focus_offsets");
  const Tensor& got_offsets = actual.Get("result/focus_offsets");
  if (got_offsets.numel() != want_offsets.numel()) {
    Fail("score_main wrote the wrong number of focus offsets");
    return;
  }
  for (std::int64_t i = 0; i < want_offsets.numel(); ++i) {
    if (got_offsets.i32()[i] != want_offsets.i32()[i]) {
      Fail(Format("focus offset %lld: got %d want %d",
                  static_cast<long long>(i), got_offsets.i32()[i],
                  want_offsets.i32()[i]));
    }
  }

  const Tensor& want_logits = reference.Get("result/focus_logits");
  const Tensor& got_logits = actual.Get("result/focus_logits");
  if (got_logits.numel() != want_logits.numel()) {
    Fail("score_main wrote the wrong number of focused logits");
    return;
  }
  for (std::int64_t i = 0; i < want_logits.numel(); ++i) {
    const double got = ApplySoftcap(got_logits.f32()[i], softcap);
    CheckClosed(got, want_logits.f32()[i],
                Format("focus logit %lld", static_cast<long long>(i)));
  }
  (void)batch;
}

void Run(const std::string& fixture_path) {
  const Fixture fixture = Fixture::Load(fixture_path);
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const float softcap = fixture.Get("config/softcap").scalar_f32();
  if (fixture.Get("config/version").scalar_int() != 1) {
    throw std::runtime_error("unsupported score-parity fixture version");
  }
  if (batch <= 0 || seq <= 0) {
    throw std::runtime_error("the fixture has no usable batch or seq");
  }

  CheckSpanShapes(fixture, batch);

  const Config config = ConfigFromFixture(fixture);
  std::unique_ptr<Model> model = Model::Create(config);
  if (model == nullptr) {
    throw std::runtime_error("cannot create the model");
  }
  LoadParameters(fixture, model.get());

  const std::string checkpoint = TempDir() + "/core_eval.ckpt";
  if (!Checkpointer::SaveModel(*model, checkpoint)) {
    throw std::runtime_error("cannot write the temporary checkpoint");
  }

  const std::string out_path = TempDir() + "/core_eval_result.bin";
  const std::string binary = LocateScoreMain();
  std::string command =
      ShellQuote(binary) + " --fixture " + ShellQuote(fixture_path) +
      " --out " + ShellQuote(out_path) + " --model " + ShellQuote(checkpoint) +
      " --layers " + std::to_string(config.num_layers) + " --heads " +
      std::to_string(config.num_heads) + " --kv-heads " +
      std::to_string(config.num_kv_heads) + " --hidden " +
      std::to_string(config.hidden_dim) + " --seq " +
      std::to_string(config.seq_len) + " --vocab " +
      std::to_string(config.vocab_size) + " --padded-vocab " +
      std::to_string(config.padded_vocab_size) + " --window-pattern " +
      ShellQuote(config.window_pattern);
  const int status = std::system(command.c_str());
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    Fail("score_main did not run successfully (exit " + std::to_string(status) +
         ")");
    return;
  }

  const Fixture actual = Fixture::Load(out_path);
  CompareResults(fixture, actual, softcap, batch, seq);
  std::printf("score_parity: matched %d x %d reference positions\n", batch,
              seq);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(LocateFixture(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "score_parity: %s\n", error.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "score_parity: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("score_parity: all checks passed\n");
  return 0;
}
