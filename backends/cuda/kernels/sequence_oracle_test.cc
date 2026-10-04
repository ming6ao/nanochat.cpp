// Oracle fixture test for the sequence families. It reads the committed
// `tests/data/debug_state.bin` (produced offline from nanochat's gpt.py) and:
//
//   1. runs ClassifierForward on the fixture's raw logits and compares the
//      per-row losses to the fixture's post-softcap logits and the batch-mean
//      loss, then runs ClassifierBackward and compares to `grad/raw_logits`;
//   2. runs EmbeddingForward on the fixture's token ids and word-embedding
//      table and checks the gathered rows, then checks the scatter-add
//      backward against the host reference on the same data;
//   3. drives an Attention check from the fixture's config (grouped-query,
//      B=2, T=8, heads=8, kv_heads=2, head_dim=4) against the host reference.
//
// The fixture carries no attention intermediates, so (3) uses the fixture
// shapes rather than a saved tensor. See docs/testing.md.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ClassifierParams;
using nanochat::ComputeType;
using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using namespace nanochat::dev::seqref;

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
  }
  candidates.emplace_back("tests/data/debug_state.bin");
  candidates.emplace_back("../tests/data/debug_state.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error("oracle fixture not found");
}

void CheckClassifier(const Fixture& fixture) {
  const Tensor& raw = fixture.Get("forward/raw_logits");
  const Tensor& logits = fixture.Get("forward/logits");
  const Tensor& loss = fixture.Get("forward/loss");
  const Tensor& grad_raw = fixture.Get("grad/raw_logits");
  const Tensor& targets_t = fixture.Get("input/targets");

  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int vocab = static_cast<int>(fixture.Get("config/vocab").scalar_int());
  const int padded =
      static_cast<int>(fixture.Get("config/padded_vocab").scalar_int());
  const int rows = batch * seq;

  ClassifierParams p;
  p.rows = rows;
  p.vocab_size = vocab;
  p.padded_vocab_size = padded;
  p.softcap = 15.0f;
  p.ignore_index = -1;

  const std::int32_t* targets = targets_t.i32();

  DevBuf<ComputeType> logits_dev(
      ToStorage(std::vector<float>(raw.f32(), raw.f32() + raw.numel())));
  DevBuf<ComputeType> losses_dev(rows);
  nanochat::kernels::ClassifierForward(p, logits_dev.ptr, targets,
                                       losses_dev.ptr);
  nanochat::kernels::Synchronize();

  // Expected per-row cross-entropy from the fixture's post-softcap logits.
  std::vector<float> expected_losses(rows, 0.0f);
  const float* capped = logits.f32();
  for (int r = 0; r < rows; ++r) {
    const float* row = capped + static_cast<std::size_t>(r) * vocab;
    float row_max = row[0];
    for (int j = 1; j < vocab; ++j) row_max = std::max(row_max, row[j]);
    double sum = 0.0;
    for (int j = 0; j < vocab; ++j) {
      sum += std::exp(static_cast<double>(row[j] - row_max));
    }
    expected_losses[r] =
        static_cast<float>(std::log(sum)) + row_max - row[targets[r]];
  }
  const std::vector<float> got_losses = FromStorage(losses_dev.Download());
  CheckVectorClose(got_losses, expected_losses, 1e-5, "oracle classifier fwd");

  double mean = 0.0;
  for (float value : got_losses) mean += value;
  mean /= rows;
  CheckVectorClose(std::vector<float>{static_cast<float>(mean)},
                   std::vector<float>{loss.scalar_f32()}, 1e-5,
                   "oracle classifier mean loss");

  DevBuf<ComputeType> dlogits_dev(rows * padded);
  nanochat::kernels::ClassifierBackward(p, logits_dev.ptr, targets,
                                        dlogits_dev.ptr);
  nanochat::kernels::Synchronize();
  // ClassifierBackward produces the per-row (sum) gradient; the fixture stores
  // the batch-mean gradient, so scale by the row count.
  std::vector<float> expected_grad(static_cast<std::size_t>(rows) * padded,
                                   0.0f);
  for (std::size_t i = 0; i < expected_grad.size(); ++i) {
    expected_grad[i] = grad_raw.f32()[i] * static_cast<float>(rows);
  }
  CheckVectorClose(FromStorage(dlogits_dev.Download()), expected_grad, 1e-5,
                   "oracle classifier bwd");
}

void CheckEmbedding(const Fixture& fixture) {
  const Tensor& tokens = fixture.Get("input/tokens");
  const Tensor& wte = fixture.Get("param/transformer.wte.weight");
  const int tokens_count = static_cast<int>(tokens.numel());
  const int vocab = static_cast<int>(wte.shape[0]);
  const int dim = static_cast<int>(wte.shape[1]);
  const std::int32_t* ids = tokens.i32();

  const std::vector<float> table(wte.f32(), wte.f32() + wte.numel());
  DevBuf<ComputeType> table_dev(ToStorage(table));
  DevBuf<ComputeType> out_dev(tokens_count * dim);
  nanochat::kernels::EmbeddingForward(tokens_count, dim, ids, table_dev.ptr,
                                      out_dev.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(
      FromStorage(out_dev.Download()),
      EmbeddingForward(tokens_count, dim,
                       std::vector<int>(ids, ids + tokens_count), table),
      1e-6, "oracle embedding fwd");

  // Scatter-add backward on the same oracle data, against the host reference.
  Rng rng;
  const std::vector<float> dout0 = RandomVec(tokens_count * dim, &rng);
  const std::vector<float> dtable_init(vocab * dim, 0.0f);
  DevBuf<ComputeType> dout_dev(ToStorage(dout0));
  DevBuf<ComputeType> dtable_dev(ToStorage(dtable_init));
  nanochat::kernels::EmbeddingBackward(tokens_count, dim, ids, dout_dev.ptr,
                                       dtable_dev.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(dtable_dev.Download()),
                   EmbeddingBackward(tokens_count, dim,
                                     std::vector<int>(ids, ids + tokens_count),
                                     dout0, dtable_init),
                   1e-6, "oracle embedding bwd");
}

void CheckAttention(const Fixture& fixture) {
  const int batch = static_cast<int>(fixture.Get("config/batch").scalar_int());
  const int seq = static_cast<int>(fixture.Get("config/seq").scalar_int());
  const int heads = static_cast<int>(fixture.Get("config/heads").scalar_int());
  const int kv_heads =
      static_cast<int>(fixture.Get("config/kv_heads").scalar_int());
  const int embd = static_cast<int>(fixture.Get("config/embd").scalar_int());
  const int head_dim = embd / heads;

  AttentionParams p;
  p.batch = batch;
  p.seq = seq;
  p.num_heads = heads;
  p.num_kv_heads = kv_heads;
  p.head_dim = head_dim;
  p.causal = true;
  p.window_left = -1;
  p.window_right = 0;
  p.kv_len = 0;
  p.scale = 0.0f;

  const int q_count = batch * seq * heads * head_dim;
  const int kv_count = batch * seq * kv_heads * head_dim;
  Rng rng;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(kv_count, &rng);
  const std::vector<float> v0 = RandomVec(kv_count, &rng);
  const std::vector<float> w = RandomVec(q_count, &rng);

  DevBuf<ComputeType> q(ToStorage(q0));
  DevBuf<ComputeType> k(ToStorage(k0));
  DevBuf<ComputeType> v(ToStorage(v0));
  DevBuf<ComputeType> out(q_count);
  DevBuf<float> stats(nanochat::AttentionStatsCount(p));
  nanochat::kernels::AttentionForward(p, q.ptr, k.ptr, v.ptr, out.ptr,
                                      stats.ptr);
  nanochat::kernels::Synchronize();
  std::vector<float> ref_stats;
  CheckVectorClose(FromStorage(out.Download()),
                   AttentionForward(p, q0, k0, v0, &ref_stats), 1e-4,
                   "oracle attention fwd");
  CheckVectorClose(stats.Download(), ref_stats, 1e-4, "oracle attention stats");

  DevBuf<ComputeType> dout(ToStorage(w));
  DevBuf<ComputeType> dq(q_count);
  DevBuf<ComputeType> dk(kv_count);
  DevBuf<ComputeType> dv(kv_count);
  nanochat::kernels::AttentionBackward(p, q.ptr, k.ptr, v.ptr, stats.ptr,
                                       dout.ptr, dq.ptr, dk.ptr, dv.ptr);
  nanochat::kernels::Synchronize();
  std::vector<float> ref_dq;
  std::vector<float> ref_dk;
  std::vector<float> ref_dv;
  AttentionBackward(p, q0, k0, v0, ref_stats, w, &ref_dq, &ref_dk, &ref_dv);
  CheckVectorClose(FromStorage(dq.Download()), ref_dq, 1e-4,
                   "oracle attention dq");
  CheckVectorClose(FromStorage(dk.Download()), ref_dk, 1e-4,
                   "oracle attention dk");
  CheckVectorClose(FromStorage(dv.Download()), ref_dv, 1e-4,
                   "oracle attention dv");
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("test");
  std::printf("sequence oracle test\n");

  Fixture fixture;
  try {
    fixture = Fixture::Load(LocateFixture(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  CheckClassifier(fixture);
  CheckEmbedding(fixture);
  CheckAttention(fixture);

  if (Failures() != 0) {
    std::printf("sequence oracle: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("sequence oracle: all checks passed\n");
  return 0;
}
