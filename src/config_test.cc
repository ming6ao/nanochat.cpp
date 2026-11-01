#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "nanochat/config.h"
#include "nanochat/model.h"
#include "nanochat/model_spec.h"
#include "nanochat/plan.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

nanochat::Config TinyConfig() {
  nanochat::Config config;
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

void ExpectInvalid(const nanochat::Config& config, const char* expected) {
  std::string error = "stale error";
  if (config.Validate(&error)) {
    Fail("invalid config passed validation");
    return;
  }
  if (error.find(expected) == std::string::npos) {
    Fail("unexpected config error: " + error);
  }
}

void CheckValidConfig() {
  nanochat::Config config = TinyConfig();
  std::string error = "stale error";
  if (!config.Validate(&error)) Fail("valid config failed: " + error);
  if (!error.empty()) Fail("successful validation did not clear the error");

  config.window_pattern.clear();
  if (!config.Validate(&error)) {
    Fail("empty window pattern failed: " + error);
  }
}

void CheckInvalidDimensions() {
  nanochat::Config config = TinyConfig();
  config.num_layers = 0;
  ExpectInvalid(config, "must be positive");

  config = TinyConfig();
  config.hidden_dim = 16;
  ExpectInvalid(config, "at least 24");

  config = TinyConfig();
  config.hidden_dim = 33;
  ExpectInvalid(config, "divisible by num_heads");

  config = TinyConfig();
  config.num_kv_heads = 3;
  ExpectInvalid(config, "divisible by num_kv_heads");

  config = TinyConfig();
  config.hidden_dim = 40;
  config.num_heads = 8;
  config.num_kv_heads = 2;
  ExpectInvalid(config, "head_dim must be even");
}

void CheckFactoryRejectsInvalidConfig() {
  nanochat::Config config = TinyConfig();
  config.num_heads = 0;
  std::string error;
  if (nanochat::Model::Create(config, &error) != nullptr) {
    Fail("model factory accepted invalid config");
  }
  if (error.empty()) Fail("model factory returned no config error");
}

void CheckInvalidMetadata() {
  nanochat::Config config = TinyConfig();
  config.padded_vocab_size = 32;
  ExpectInvalid(config, "padded_vocab_size must be >= vocab_size");

  config = TinyConfig();
  config.rope_base = 0.0f;
  ExpectInvalid(config, "rope_base must be finite and positive");

  config = TinyConfig();
  config.window_pattern = "SX";
  ExpectInvalid(config, "window_pattern can contain only");
}

// ---------------------------------------------------------------------------
// The spec reader and the derivation (model-and-device-bringup section 8, and
// the P2 gate of section 16): the first model's spec must derive exactly
// today's `Config`, field by field, with no numeric change. This one equality
// is the whole migration cost of the first model - a spec that derived one
// field differently would be a second definition of the model, and the oracle
// fixture would be comparing against two.
// ---------------------------------------------------------------------------

// The spec's path, from the Bazel runfiles first and the source tree second, so
// the check runs under `tools/nanochat test` and by hand.
std::string LocateSpec(const std::string& relative) {
  std::vector<std::string> candidates;
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.push_back(std::string(src_dir) + "/" + workspace + "/" +
                           relative);
    }
    candidates.push_back(std::string(src_dir) + "/_main/" + relative);
  }
  candidates.push_back(relative);
  candidates.push_back("../" + relative);
  candidates.push_back("../../" + relative);
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate);
    if (in.good()) return candidate;
  }
  return std::string();
}

// A spec file with one line replaced, so a reader rejection can be exercised
// without a second committed file.
std::string SpecWithEdit(const std::string& path, const std::string& from,
                         const std::string& to) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  const std::size_t at = text.find(from);
  if (at == std::string::npos) return std::string();
  return text.replace(at, from.size(), to);
}

void WriteTemp(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

void CheckSpecDerivesTodaysConfig() {
  const std::string path = LocateSpec("models/nanochat-default/spec.json");
  if (path.empty()) {
    Fail("models/nanochat-default/spec.json is not reachable");
    return;
  }
  nanochat::ModelSpec spec;
  std::string error;
  if (!nanochat::ParseModelSpec(path, &spec, &error)) {
    Fail("the flagship spec does not parse: " + error);
    return;
  }

  nanochat::Config derived;
  if (!nanochat::ModelSpecToConfig(spec, &derived, &error)) {
    Fail("the flagship spec does not derive a Config: " + error);
    return;
  }
  // Field by field, against the struct `Config`'s own defaults: this is the
  // equality the migration cost is measured in, so it is spelled as a field
  // comparison rather than as a struct comparison - a new field must be added
  // here explicitly, and that is the point.
  const nanochat::Config today;
  const auto expect = [](const char* field, bool same) {
    if (!same) Fail(std::string("the spec derives a different ") + field);
  };
  expect("num_layers", derived.num_layers == today.num_layers);
  expect("num_heads", derived.num_heads == today.num_heads);
  expect("num_kv_heads", derived.num_kv_heads == today.num_kv_heads);
  expect("hidden_dim", derived.hidden_dim == today.hidden_dim);
  expect("seq_len", derived.seq_len == today.seq_len);
  expect("vocab_size", derived.vocab_size == today.vocab_size);
  expect("padded_vocab_size",
         derived.padded_vocab_size == today.padded_vocab_size);
  expect("rope_base", derived.rope_base == today.rope_base);
  expect("window_pattern", derived.window_pattern == today.window_pattern);
  expect("value_embedding", derived.value_embedding == today.value_embedding);
  // The derived accessors stay on `Config`, and the block list cannot disagree
  // with them: the pattern is one round of the block order.
  expect("head_dim()", derived.head_dim() == today.head_dim());
  expect("mlp_dim()", derived.mlp_dim() == today.mlp_dim());
  expect("query_dim()", derived.query_dim() == today.query_dim());
  expect("kv_dim()", derived.kv_dim() == today.kv_dim());
  expect("short_window()", derived.short_window() == today.short_window());
  expect("window_right()", derived.window_right() == today.window_right());
  expect("window_left() at layer 0",
         derived.window_left(0) == today.window_left(0));
  expect("window_left() at the last layer",
         derived.window_left(derived.num_layers - 1) ==
             today.window_left(today.num_layers - 1));
  expect("has_value_embedding() at layer 1",
         derived.has_value_embedding(1) == today.has_value_embedding(1));
  expect("the block count",
         spec.blocks.size() == static_cast<std::size_t>(today.num_layers));
}

void CheckReaderRefusesWhatItDoesNotImplement() {
  const std::string path = LocateSpec("models/nanochat-default/spec.json");
  if (path.empty()) {
    Fail("models/nanochat-default/spec.json is not reachable");
    return;
  }
  nanochat::ModelSpec spec;
  std::string error;

  // A spec format this reader does not implement is refused by number.
  const std::string versioned =
      SpecWithEdit(path, "\"spec_version\": 1", "\"spec_version\": 2");
  WriteTemp("/tmp/nanochat_spec_version.json", versioned);
  if (nanochat::ParseModelSpec("/tmp/nanochat_spec_version.json", &spec,
                               &error)) {
    Fail("the reader accepted an unknown spec_version");
  } else if (error.find("spec_version") == std::string::npos ||
             error.find("2") == std::string::npos) {
    Fail("the spec_version rejection does not name the version: " + error);
  }

  // An unknown block kind is refused by name, not rounded to a default.
  const std::string kinded =
      SpecWithEdit(path, "\"mlp\": \"relu_square\"", "\"mlp\": \"geglu\"");
  WriteTemp("/tmp/nanochat_spec_kind.json", kinded);
  if (nanochat::ParseModelSpec("/tmp/nanochat_spec_kind.json", &spec, &error)) {
    Fail("the reader accepted an unknown block kind");
  } else if (error.find("geglu") == std::string::npos) {
    Fail("the block-kind rejection does not name the kind: " + error);
  }

  // An unknown key is refused, because the field set is closed.
  const std::string keyed =
      SpecWithEdit(path, "\"name\": \"nanochat-default\"",
                   "\"name\": \"nanochat-default\", \"hidden_states\": 4");
  WriteTemp("/tmp/nanochat_spec_key.json", keyed);
  if (nanochat::ParseModelSpec("/tmp/nanochat_spec_key.json", &spec, &error)) {
    Fail("the reader accepted an unknown field");
  } else if (error.find("hidden_states") == std::string::npos) {
    Fail("the unknown-field rejection does not name the field: " + error);
  }

  // A null spec name selects the built-in default, which is the spec that
  // reproduces `Config`'s defaults, so the historical `Create(config)` meaning
  // is intact.
  if (!nanochat::LookupModelSpec(nullptr, &spec, &error)) {
    Fail("a null spec name did not resolve: " + error);
  }
}

// The composition path is what runs on a device whose declaration offers no
// override bit. The reference backend's `caps.json` declares none on purpose
// (`backends/cpu/caps.json`: "with no fused override declared, the planner
// resolves every composition to the composition path"), so a plan built from
// the live capability answer is the check: every composition must resolve to
// the composition tier, which is the definition of the semantics, and no
// decision may claim a fused family.
void CheckPlanRunsTheCompositionPath() {
  const std::string path = LocateSpec("models/nanochat-default/spec.json");
  if (path.empty()) {
    Fail("models/nanochat-default/spec.json is not reachable");
    return;
  }
  nanochat::ModelSpec spec;
  std::string error;
  if (!nanochat::ParseModelSpec(path, &spec, &error)) {
    Fail("the flagship spec does not parse: " + error);
    return;
  }
  nanochat::PlanRequest request;
  request.model = &spec;
  request.graph = nanochat::GraphKind::kTrain;
  request.device_index = 0;
  std::string why_not;
  const nanochat::Plan plan = nanochat::BuildPlan(request, &why_not);
  if (!plan.ok) {
    Fail("the flagship spec has no plan on the reference backend: " + why_not);
    return;
  }
  for (const nanochat::CompositionChoice& choice : plan.choices) {
    if (choice.fused) {
      Fail(std::string("composition ") +
           nanochat::CompositionName(choice.composition) +
           " resolved to a fused family, but the reference declaration offers "
           "no override bit");
    }
    if (choice.implementation != "composition") {
      Fail(std::string("composition ") +
           nanochat::CompositionName(choice.composition) + " resolved to \"" +
           choice.implementation + "\" instead of the composition tier");
    }
  }
}

}  // namespace

int main() {
  CheckValidConfig();
  CheckInvalidDimensions();
  CheckInvalidMetadata();
  CheckFactoryRejectsInvalidConfig();
  CheckSpecDerivesTodaysConfig();
  CheckReaderRefusesWhatItDoesNotImplement();
  CheckPlanRunsTheCompositionPath();

  if (g_failures != 0) {
    std::fprintf(stderr, "config_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("config_test: all checks passed\n");
  return 0;
}
