// CPU oracle test.
//
// Reads the committed fixture `tests/data/debug_state.bin` (produced offline by
// `tools/dump_oracle.py` from nanochat's `gpt.py`) and checks the reference
// math that the C++ model must reproduce:
//
//   1. the post-softcap logits are `15 * tanh(raw / 15)`;
//   2. the loss is the batch-mean cross-entropy of those logits;
//   3. `grad/logits` is `(softmax(logits) - onehot(targets)) / count`;
//   4. `grad/raw_logits` chains (3) through the softcap derivative;
//   5. every parameter has a matching, finite gradient;
//   6. the first optimizer step matches the canonical AdamW update for each
//      AdamW group, and the recorded loss curve descends.
//
// The comparison against the *model's* forward/backward is added when the
// model skeleton lands (`src/model.cc`) and extended to the CUDA backend by the
// parity node; this test is deliberately self-contained so it builds and runs
// before the model exists.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/oracle_fixture.h"

namespace {

using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

void ExpectTrue(bool condition, const std::string& what) {
  if (!condition) Fail(what);
}

std::string Format(const char* fmt, ...) {
  char buffer[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

void ExpectNear(double got, double want, double tolerance,
                const std::string& what) {
  const double diff = std::fabs(got - want);
  if (!(diff <= tolerance)) {
    Fail(Format("%s: got %.9g want %.9g (|diff|=%.3g > tol=%.3g)", what.c_str(),
                got, want, diff, tolerance));
  }
}

std::string Join(const std::vector<std::string>& parts) {
  std::string joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) joined += ", ";
    joined += parts[i];
  }
  return joined;
}

// Resolve the fixture path. Bazel passes `$(rootpath ...)` as argv[1]; when the
// binary is run directly (for example by `tools/nanochat verify`) fall back to
// the runfiles tree and then to the source-relative path.
std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/debug_state.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/debug_state.bin");
    candidates.emplace_back(std::string(src_dir) +
                            "/nanochat_cpp/tests/data/debug_state.bin");
  }
  candidates.emplace_back("tests/data/debug_state.bin");
  candidates.emplace_back("nanochat.cpp/tests/data/debug_state.bin");
  candidates.emplace_back("../tests/data/debug_state.bin");

  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error("oracle fixture not found; tried: " +
                           Join(candidates));
}

double CrossEntropy(const float* logits, const std::int32_t* targets, int batch,
                    int seq, int vocab) {
  const int count = batch * seq;
  double total = 0.0;
  for (int i = 0; i < count; ++i) {
    const float* row = logits + static_cast<std::size_t>(i) * vocab;
    double max_logit = row[0];
    for (int v = 1; v < vocab; ++v) {
      max_logit = std::max(max_logit, static_cast<double>(row[v]));
    }
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
      sum += std::exp(static_cast<double>(row[v]) - max_logit);
    }
    const int target = targets[i];
    total += -(static_cast<double>(row[target]) - max_logit - std::log(sum));
  }
  return total / count;
}

// Check that one AdamW group's first step matches the canonical update used by
// nanochat's `setup_optimizer` / `adamw_step_fused`.
void CheckAdamWStep1(const Fixture& fixture, const std::string& name, double lr,
                     double beta1, double beta2, double eps,
                     double weight_decay) {
  const Tensor& p0 = fixture.Get("param/" + name);
  const Tensor& grad = fixture.Get("grad/" + name);
  const Tensor& p1 = fixture.Get("step/1/param/" + name);
  const std::int64_t count = p0.numel();
  if (grad.numel() != count || p1.numel() != count) {
    Fail(name + ": parameter/gradient/update shape mismatch");
    return;
  }
  const float* p0_data = p0.f32();
  const float* grad_data = grad.f32();
  const float* p1_data = p1.f32();
  double max_diff = 0.0;
  for (std::int64_t i = 0; i < count; ++i) {
    const double p = p0_data[i];
    const double g = grad_data[i];
    const double decayed = p * (1.0 - lr * weight_decay);
    const double exp_avg = (1.0 - beta1) * g;
    const double exp_avg_sq = (1.0 - beta2) * g * g;
    const double denom = std::sqrt(exp_avg_sq / (1.0 - beta2)) + eps;
    const double step_size = lr / (1.0 - beta1);
    const double updated = decayed - step_size * (exp_avg / denom);
    max_diff = std::max(max_diff,
                        std::fabs(updated - static_cast<double>(p1_data[i])));
  }
  ExpectTrue(max_diff < 1e-6,
             Format("AdamW step-1 mismatch for %s: max |diff| = %.3g",
                    name.c_str(), max_diff));
}

void RunChecks(const Fixture& fixture) {
  // --- configuration -------------------------------------------------------
  const std::int64_t layers = fixture.Get("config/layers").scalar_int();
  const std::int64_t heads = fixture.Get("config/heads").scalar_int();
  const std::int64_t kv_heads = fixture.Get("config/kv_heads").scalar_int();
  const std::int64_t embd = fixture.Get("config/embd").scalar_int();
  const std::int64_t vocab = fixture.Get("config/vocab").scalar_int();
  const std::int64_t padded_vocab =
      fixture.Get("config/padded_vocab").scalar_int();
  const std::int64_t seq = fixture.Get("config/seq").scalar_int();
  const std::int64_t batch = fixture.Get("config/batch").scalar_int();
  const std::int64_t steps = fixture.Get("config/steps").scalar_int();

  ExpectTrue(layers == 2 && heads == 8 && kv_heads == 2 && embd == 32,
             "unexpected model config in fixture");
  ExpectTrue(vocab == 64 && padded_vocab == 64 && seq == 8 && batch == 2,
             "unexpected token config in fixture");
  ExpectTrue(steps >= 1, "fixture must record at least one optimizer step");

  const Tensor& raw = fixture.Get("forward/raw_logits");
  const Tensor& logits = fixture.Get("forward/logits");
  const Tensor& loss = fixture.Get("forward/loss");
  const Tensor& grad_logits = fixture.Get("grad/logits");
  const Tensor& grad_raw = fixture.Get("grad/raw_logits");
  const Tensor& targets = fixture.Get("input/targets");

  ExpectTrue(raw.shape.size() == 3 && raw.shape[0] == batch &&
                 raw.shape[1] == seq && raw.shape[2] == vocab,
             "forward/raw_logits has an unexpected shape");
  ExpectTrue(logits.shape == raw.shape, "forward/logits shape mismatch");
  ExpectTrue(grad_logits.shape == raw.shape, "grad/logits shape mismatch");
  ExpectTrue(grad_raw.shape == raw.shape, "grad/raw_logits shape mismatch");

  const std::int64_t element_count = raw.numel();
  const float* raw_data = raw.f32();
  const float* logits_data = logits.f32();
  const float* grad_logits_data = grad_logits.f32();
  const float* grad_raw_data = grad_raw.f32();
  const std::int32_t* target_data = targets.i32();

  // --- 1. softcap ----------------------------------------------------------
  double softcap_error = 0.0;
  for (std::int64_t i = 0; i < element_count; ++i) {
    const double expected =
        15.0 * std::tanh(static_cast<double>(raw_data[i]) / 15.0);
    softcap_error =
        std::max(softcap_error,
                 std::fabs(expected - static_cast<double>(logits_data[i])));
  }
  ExpectTrue(softcap_error < 1e-6,
             Format("softcap mismatch: max |diff| = %.3g", softcap_error));

  // --- 2. cross-entropy loss ----------------------------------------------
  const double recomputed_loss =
      CrossEntropy(logits_data, target_data, static_cast<int>(batch),
                   static_cast<int>(seq), static_cast<int>(vocab));
  ExpectNear(recomputed_loss, loss.scalar_f32(), 1e-5, "cross-entropy loss");

  // --- 3. d(loss)/d(logits) ------------------------------------------------
  double grad_logits_error = 0.0;
  double grad_raw_error = 0.0;
  const double count = static_cast<double>(batch * seq);
  for (std::int64_t i = 0; i < batch * seq; ++i) {
    const float* row = logits_data + i * vocab;
    const std::int32_t target = target_data[i];
    double max_logit = row[0];
    for (std::int64_t v = 1; v < vocab; ++v) {
      max_logit = std::max(max_logit, static_cast<double>(row[v]));
    }
    double sum = 0.0;
    for (std::int64_t v = 0; v < vocab; ++v) {
      sum += std::exp(static_cast<double>(row[v]) - max_logit);
    }
    for (std::int64_t v = 0; v < vocab; ++v) {
      const double probability =
          std::exp(static_cast<double>(row[v]) - max_logit) / sum;
      const double expected_logits =
          (probability - (v == target ? 1.0 : 0.0)) / count;
      grad_logits_error = std::max(
          grad_logits_error,
          std::fabs(expected_logits -
                    static_cast<double>(grad_logits_data[i * vocab + v])));
      const double softcap_derivative =
          1.0 -
          std::pow(static_cast<double>(logits_data[i * vocab + v]) / 15.0, 2.0);
      const double expected_raw = expected_logits * softcap_derivative;
      grad_raw_error = std::max(
          grad_raw_error,
          std::fabs(expected_raw -
                    static_cast<double>(grad_raw_data[i * vocab + v])));
    }
  }
  ExpectTrue(
      grad_logits_error < 1e-6,
      Format("grad/logits mismatch: max |diff| = %.3g", grad_logits_error));
  ExpectTrue(
      grad_raw_error < 1e-6,
      Format("grad/raw_logits mismatch: max |diff| = %.3g", grad_raw_error));

  // --- 4. parameters and gradients ----------------------------------------
  std::vector<std::string> parameter_names;
  for (const std::string& name : fixture.names()) {
    if (name.rfind("param/", 0) == 0) {
      parameter_names.push_back(name.substr(std::string("param/").size()));
    }
  }
  ExpectTrue(!parameter_names.empty(), "fixture has no parameter records");

  double grad_magnitude = 0.0;
  for (const std::string& name : parameter_names) {
    const Tensor& parameter = fixture.Get("param/" + name);
    const std::string grad_name = "grad/" + name;
    if (!fixture.Has(grad_name)) {
      Fail("missing gradient for parameter " + name);
      continue;
    }
    const Tensor& gradient = fixture.Get(grad_name);
    if (gradient.shape != parameter.shape) {
      Fail("gradient shape mismatch for parameter " + name);
      continue;
    }
    for (std::int64_t i = 0; i < parameter.numel(); ++i) {
      if (!std::isfinite(parameter.f32()[i])) {
        Fail("non-finite parameter value in " + name);
        break;
      }
    }
    for (std::int64_t i = 0; i < gradient.numel(); ++i) {
      const float value = gradient.f32()[i];
      if (!std::isfinite(value)) {
        Fail("non-finite gradient value in " + name);
        break;
      }
      grad_magnitude += std::fabs(static_cast<double>(value));
    }
  }
  ExpectTrue(grad_magnitude > 0.0, "all recorded gradients are zero");

  // --- 5. optimizer trajectory --------------------------------------------
  std::vector<double> losses(static_cast<std::size_t>(steps) + 1);
  for (std::int64_t step = 0; step <= steps; ++step) {
    const std::string name = "step/" + std::to_string(step) + "/loss";
    if (!fixture.Has(name)) {
      Fail("missing loss record " + name);
      continue;
    }
    losses[static_cast<std::size_t>(step)] = fixture.Get(name).scalar_f32();
    ExpectTrue(std::isfinite(losses[static_cast<std::size_t>(step)]),
               "non-finite loss in " + name);
  }
  ExpectTrue(losses.back() < losses.front(),
             "optimizer steps did not reduce the loss");

  // Every parameter must have a recorded trajectory, and at least one must
  // actually change (a silent no-op optimizer would still pass the loss check
  // only by accident).
  double parameter_change = 0.0;
  for (const std::string& name : parameter_names) {
    const std::string initial_name = "step/0/param/" + name;
    const std::string final_name =
        "step/" + std::to_string(steps) + "/param/" + name;
    if (!fixture.Has(initial_name) || !fixture.Has(final_name)) {
      Fail("missing optimizer trajectory for parameter " + name);
      continue;
    }
    const Tensor& initial = fixture.Get(initial_name);
    const Tensor& updated = fixture.Get(final_name);
    if (initial.shape != updated.shape) {
      Fail("optimizer trajectory shape mismatch for parameter " + name);
      continue;
    }
    for (std::int64_t i = 0; i < initial.numel(); ++i) {
      parameter_change += std::fabs(static_cast<double>(updated.f32()[i]) -
                                    static_cast<double>(initial.f32()[i]));
    }
  }
  ExpectTrue(parameter_change > 0.0,
             "optimizer steps did not change any parameter");

  // --- 6. first step matches canonical AdamW ------------------------------
  const double scale = std::sqrt(768.0 / static_cast<double>(embd));
  const double unembedding_lr =
      fixture.Get("config/opt/unembedding_lr").scalar_f32();
  const double embedding_lr =
      fixture.Get("config/opt/embedding_lr").scalar_f32();
  const double scalar_lr = fixture.Get("config/opt/scalar_lr").scalar_f32();
  const double eps = 1e-10;
  // lm_head: AdamW, betas (0.8, 0.96), weight_decay 0.01.
  CheckAdamWStep1(fixture, "lm_head.weight", unembedding_lr * scale, 0.8, 0.96,
                  eps, 0.01);
  // token embedding: betas (0.8, 0.995), weight_decay 0.001.
  CheckAdamWStep1(fixture, "transformer.wte.weight", embedding_lr * scale, 0.8,
                  0.995, eps, 0.001);
  // value embeddings: half the embedding learning rate, weight_decay 0.01.
  CheckAdamWStep1(fixture, "value_embeds.1.weight", embedding_lr * scale * 0.5,
                  0.8, 0.995, eps, 0.01);
  // resid lambdas: 0.01x scalar learning rate, betas (0.8, 0.95), weight_decay
  // 0.05.
  CheckAdamWStep1(fixture, "resid_lambdas", scalar_lr * 0.01, 0.8, 0.95, eps,
                  0.05);
  // x0 lambdas: scalar learning rate, betas (0.96, 0.95), no weight decay.
  CheckAdamWStep1(fixture, "x0_lambdas", scalar_lr, 0.96, 0.95, eps, 0.0);
  // smear/backout scalars: fixed learning rate 0.2, betas (0.8, 0.95), no
  // decay.
  CheckAdamWStep1(fixture, "smear_gate.weight", 0.2, 0.8, 0.95, eps, 0.0);
  CheckAdamWStep1(fixture, "backout_lambda", 0.2, 0.8, 0.95, eps, 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  try {
    path = LocateFixture(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  Fixture fixture;
  try {
    fixture = Fixture::Load(path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  std::printf("oracle_test: loaded %s (%zu records)\n", path.c_str(),
              fixture.size());
  RunChecks(fixture);

  if (g_failures != 0) {
    std::fprintf(stderr, "oracle_test: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("oracle_test: all checks passed\n");
  return 0;
}
