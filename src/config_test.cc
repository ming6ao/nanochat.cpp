#include <cstdio>
#include <string>

#include "nanochat/config.h"
#include "nanochat/model.h"

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

}  // namespace

int main() {
  CheckValidConfig();
  CheckInvalidDimensions();
  CheckInvalidMetadata();
  CheckFactoryRejectsInvalidConfig();

  if (g_failures != 0) {
    std::fprintf(stderr, "config_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("config_test: all checks passed\n");
  return 0;
}
