// T1 GPU correctness + finite-difference test for the Classifier family:
// logit softcap + cross-entropy over the real vocabulary slice, with a padded
// tail. Tiny shapes only. See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"

namespace {

using nanochat::ClassifierParams;
using nanochat::ComputeType;
using nanochat::dev::CheckFiniteDifference;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;
using namespace nanochat::dev::seqref;

void RunCase(int rows, int vocab, int padded, bool use_ignore,
             const char* name) {
  ClassifierParams p;
  p.rows = rows;
  p.vocab_size = vocab;
  p.padded_vocab_size = padded;
  p.softcap = 15.0f;
  p.ignore_index = -1;

  Rng rng;
  // Fill the padded tail with values that would change the loss if the kernel
  // read them; a correct implementation never does.
  std::vector<float> logits = RandomVec(rows * padded, &rng);
  for (int r = 0; r < rows; ++r) {
    for (int j = vocab; j < padded; ++j) {
      logits[static_cast<std::size_t>(r) * padded + j] = 100.0f + j;
    }
  }
  std::vector<int> targets(rows);
  for (int r = 0; r < rows; ++r) targets[r] = r % vocab;
  if (use_ignore) targets[rows / 2] = p.ignore_index;

  DevBuf<ComputeType> dlogits_in(ToStorage(logits));
  DevBuf<ComputeType> losses(rows);
  nanochat::kernels::ClassifierForward(p, dlogits_in.ptr, targets.data(),
                                       losses.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> ref_losses = ClassifierForward(p, logits, targets);
  CheckVectorClose(FromStorage(losses.Download()), ref_losses, 1e-4,
                   (std::string(name) + " fwd").c_str());

  DevBuf<ComputeType> dlogits(rows * padded);
  nanochat::kernels::ClassifierBackward(p, dlogits_in.ptr, targets.data(),
                                        dlogits.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> ref_dlogits = ClassifierBackward(p, logits, targets);
  CheckVectorClose(FromStorage(dlogits.Download()), ref_dlogits, 1e-4,
                   (std::string(name) + " bwd").c_str());

  // The padded tail must be zeroed even when it held garbage on entry.
  const std::vector<float> got = FromStorage(dlogits.Download());
  for (int r = 0; r < rows; ++r) {
    for (int j = vocab; j < padded; ++j) {
      if (got[static_cast<std::size_t>(r) * padded + j] != 0.0f) {
        std::printf("FAIL: %s padded tail not zeroed at row %d col %d\n", name,
                    r, j);
        ++Failures();
      }
    }
  }

#if !defined(NANOCHAT_PRECISION_FP16)
  auto loss_at = [&](const std::vector<float>& lv) {
    const std::vector<float> l = ClassifierForward(p, lv, targets);
    double loss = 0.0;
    for (float value : l) loss += value;
    return loss;
  };
  CheckFiniteDifference(logits, got, loss_at, 3e-3,
                        (std::string(name) + " dlogits").c_str());
#endif
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("classifier gpu test\n");

  RunCase(6, 10, 10, false, "no-pad");
  RunCase(6, 10, 16, false, "padded");
  RunCase(6, 10, 16, true, "ignore-index");

  if (Failures() != 0) {
    std::printf("classifier: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("classifier: all checks passed\n");
  return 0;
}
