// T1 GPU correctness + finite-difference test for the Embedding family:
// indexed gather forward and scatter-add backward over a persistent dense
// gradient buffer, including duplicate ids. Tiny shapes only. See
// docs/testing.md.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using nanochat::dev::seqref::EmbeddingBackward;
using nanochat::dev::seqref::EmbeddingForward;

void RunCase(int vocab, int dim, const std::vector<int>& ids,
             const char* name) {
  const int tokens = static_cast<int>(ids.size());

  Rng rng;
  const std::vector<float> table0 = RandomVec(vocab * dim, &rng);
  const std::vector<float> dout0 = RandomVec(tokens * dim, &rng);
  const std::vector<float> dtable_init = RandomVec(vocab * dim, &rng);

  // Forward gather.
  DevBuf<ComputeType> table(ToStorage(table0));
  DevBuf<ComputeType> out(tokens * dim);
  nanochat::kernels::EmbeddingForward(tokens, dim, ids.data(), table.ptr,
                                      out.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(out.Download()),
                   EmbeddingForward(tokens, dim, ids, table0), 1e-4,
                   (std::string(name) + " fwd").c_str());

  // Backward over a persistent, non-zero gradient buffer: only the touched rows
  // may change.
  DevBuf<ComputeType> dout(ToStorage(dout0));
  DevBuf<ComputeType> dtable(ToStorage(dtable_init));
  nanochat::kernels::EmbeddingBackward(tokens, dim, ids.data(), dout.ptr,
                                       dtable.ptr);
  nanochat::kernels::Synchronize();
  CheckVectorClose(FromStorage(dtable.Download()),
                   EmbeddingBackward(tokens, dim, ids, dout0, dtable_init),
                   1e-4, (std::string(name) + " bwd persistent").c_str());

#if !defined(NANOCHAT_PRECISION_FP16)
  // Finite-difference the gather with a zero-initialised gradient buffer so the
  // analytic gradient is exact on untouched rows as well.
  DevBuf<ComputeType> dtable_zero(
      ToStorage(std::vector<float>(vocab * dim, 0.0f)));
  nanochat::kernels::EmbeddingBackward(tokens, dim, ids.data(), dout.ptr,
                                       dtable_zero.ptr);
  nanochat::kernels::Synchronize();
  const std::vector<float> analytic = FromStorage(dtable_zero.Download());
  auto loss_at = [&](const std::vector<float>& tv) {
    const std::vector<float> o = EmbeddingForward(tokens, dim, ids, tv);
    double loss = 0.0;
    for (int i = 0; i < tokens * dim; ++i) loss += o[i] * dout0[i];
    return loss;
  };
  CheckFiniteDifference(table0, analytic, loss_at, 3e-3,
                        (std::string(name) + " dtable").c_str());
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("embedding gpu test\n");

  RunCase(8, 4, {0, 3, 7, 3, 1, 3}, "duplicates");
  RunCase(5, 3, {4, 0, 2}, "simple");

  if (Failures() != 0) {
    std::printf("embedding: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("embedding: all checks passed\n");
  return 0;
}
