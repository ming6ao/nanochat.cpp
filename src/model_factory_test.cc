// The model factory's three-state selector (docs/api-surface.md R8).
//
// The two public `Model::Create` overloads differ in one way: `Create(config)`
// reads no spec at all, and `Create(config, spec_name)` reads one, where a null
// name still selects the built-in default spec. Both are one-line wrappers of
// `CreateTrainModel`, so the three states are spelled by `SpecSource` instead
// of by a nullable name. This test holds that distinction down: the plan-less
// path builds a model with no plan, and the named path with a null name reads
// the default spec and therefore rejects a `Config` that spec does not derive.
//
// The success case of the named path is not checked here. The default spec
// derives `Config`'s own defaults (12 blocks, 768 hidden, 32768 padded vocab),
// which allocates about 1.1 GB of parameters - over the `t0-cpu` memory budget
// and pointless for a selector check. The cheap half is the interesting one:
// the same `Config` is accepted by one source and rejected by the other, which
// is what a collapsed selector would get wrong. CPU only.

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "nanochat/model.h"
#include "nanochat/plan.h"
#include "src/model_impl.h"

namespace {

using nanochat::Config;
using nanochat::Model;
using nanochat::TrainModel;

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

void Expect(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

void ExpectContains(const std::string& text, const std::string& needle,
                    const std::string& what) {
  if (text.find(needle) == std::string::npos) {
    Fail(what + ": \"" + text + "\" does not contain \"" + needle + "\"");
  }
}

// The tiny architecture the other `src/` tests build: cheap to allocate.
Config TinyConfig() {
  Config config;
  config.num_layers = 2;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  config.hidden_dim = 32;
  config.seq_len = 8;
  config.vocab_size = 64;
  config.padded_vocab_size = 64;
  config.window_pattern = "SL";
  return config;
}

// `SpecSource::kConfigOnly`: no spec is read, so the model is built and its
// plan is plan-less - no choices, no spec name, and no log line.
void CheckTheConfigOnlySourceIsPlanLess() {
  const Config config = TinyConfig();
  std::string error;
  const std::unique_ptr<Model> model = Model::Create(config, &error);
  if (model == nullptr) {
    Fail("the config-only source rejected a valid Config: " + error);
    return;
  }
  const auto* train = dynamic_cast<const TrainModel*>(model.get());
  if (train == nullptr) {
    Fail("the config-only source did not build a TrainModel");
    return;
  }
  Expect(!train->plan().ok, "the config-only plan is not ok (it is plan-less)");
  Expect(train->plan().spec_name.empty(), "the config-only plan names no spec");
  Expect(train->plan().choices.empty(),
         "the config-only plan has no composition choices");
  Expect(train->plan().gaps.empty(), "the config-only plan lists no gaps");
  // The block list comes from the `Config` alone, and it is the one the
  // window pattern and the value-embedding flag imply.
  Expect(train->spec().blocks.size() ==
             static_cast<std::size_t>(config.num_layers),
         "the config-only spec has one block per layer");
  Expect(train->spec().name == "config",
         "the config-only spec is named after its source");
}

// `SpecSource::kByName` with a null name: the default spec is read, so a tiny
// `Config` that the default spec does not derive is rejected with the
// disagreement, not accepted as the plan-less path would accept it.
void CheckANullNameStillReadsTheDefaultSpec() {
  const Config config = TinyConfig();
  std::string error;
  const std::unique_ptr<Model> model = Model::Create(config, nullptr, &error);
  Expect(model == nullptr,
         "a null name did not select a spec to disagree with");
  ExpectContains(error, "spec \"default\" disagrees with the given Config",
                 "the null-name rejection names the default spec");
  ExpectContains(error, "the spec derives a Config whose num_layers is 12",
                 "the null-name rejection names the first differing field");
}

// `SpecSource::kByName` with an unknown name: the selector reads a file, and a
// name with no `models/<name>/spec.json` says so.
void CheckAnUnknownNameReportsTheMissingSpec() {
  const Config config = TinyConfig();
  std::string error;
  const std::unique_ptr<Model> model =
      Model::Create(config, "no-such-model-package", &error);
  Expect(model == nullptr, "an unknown spec name built a model");
  ExpectContains(error, "models/no-such-model-package/spec.json",
                 "the unknown-name rejection names the path it read");
}

// The shared validation runs first and is the same for both sources: an
// invalid `Config` is rejected by `Config::Validate` before any spec is read,
// so the named source does not report the missing file.
void CheckBothSourcesShareTheValidation() {
  Config config = TinyConfig();
  config.num_layers = 0;
  const std::string expected =
      "model dimensions and vocabulary sizes must be positive";

  std::string config_only_error;
  const std::unique_ptr<Model> config_only =
      Model::Create(config, &config_only_error);
  Expect(config_only == nullptr, "an invalid Config built a model");
  Expect(config_only_error == expected,
         Format("the config-only rejection is \"%s\", not \"%s\"",
                config_only_error.c_str(), expected.c_str()));

  std::string by_name_error;
  const std::unique_ptr<Model> by_name =
      Model::Create(config, "no-such-model-package", &by_name_error);
  Expect(by_name == nullptr, "an invalid Config built a by-name model");
  Expect(by_name_error == expected,
         Format("the by-name rejection is \"%s\", not \"%s\"",
                by_name_error.c_str(), expected.c_str()));
}

}  // namespace

int main() {
  CheckTheConfigOnlySourceIsPlanLess();
  CheckANullNameStillReadsTheDefaultSpec();
  CheckAnUnknownNameReportsTheMissingSpec();
  CheckBothSourcesShareTheValidation();
  if (g_failures != 0) {
    std::fprintf(stderr, "model_factory: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("model_factory: ok\n");
  return 0;
}
