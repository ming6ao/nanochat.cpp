// T1 GPU correctness + finite-difference test for the smear family. Tiny shapes
// only. The device result is compared against the host reference in
// testing/smear_ref.h, and the backward is checked by central differences of
// that reference forward, so the kernel answers to the definition and not to a
// second kernel. See docs/testing.md.

#include <cstdio>
#include <utility>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/smear_ref.h"
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
using nanochat::kernels::SmearParams;
using nanochat::kernels::testing::SmearBackwardRef;
using nanochat::kernels::testing::SmearForwardRef;

// (batch, seq, hidden, channels). The third shape has a width that is not a
// multiple of the gate width, which is the ordinary case.
const SmearParams kShapes[] = {
    {2, 4, 8, 4},
    {1, 6, 12, 3},
    {3, 3, 5, 5},
};

void TestForwardBackward() {
  Rng rng;
  for (const SmearParams& shape : kShapes) {
    const int rows = shape.rows();
    const int hidden = shape.hidden;
    SmearParams params = shape;
    params.lambda = 0.7f;

    const std::vector<float> emb = RandomVec(rows * hidden, &rng);
    const std::vector<float> gate_w = RandomVec(shape.channels, &rng);
    const std::vector<float> dy = RandomVec(rows * hidden, &rng);

    DevBuf<ComputeType> d_emb(ToStorage(emb));
    DevBuf<ComputeType> d_w(ToStorage(gate_w));
    DevBuf<ComputeType> d_x0(rows * hidden);
    DevBuf<float> d_sig(rows);
    nanochat::kernels::SmearForward(params, d_emb.ptr, d_w.ptr, d_x0.ptr,
                                    d_sig.ptr);
    nanochat::kernels::Synchronize();

    std::vector<float> ref_x0(rows * hidden, 0.0f);
    std::vector<float> ref_sig(rows, 0.0f);
    SmearForwardRef(params, emb, gate_w, ref_x0.data(), ref_sig.data());
    CheckVectorClose(FromStorage(d_x0.Download()), ref_x0, 1e-4, "fwd x0");
    CheckVectorClose(d_sig.Download(), ref_sig, 1e-4, "fwd sig");

    // The backward accumulates the two parameter gradients, so seed them with a
    // value the kernel has to add to, exactly as the chain does.
    const std::vector<float> w_seed = RandomVec(shape.channels, &rng);
    const std::vector<float> l_seed = RandomVec(1, &rng);
    DevBuf<float> d_wgrad(shape.channels);
    DevBuf<float> d_lgrad(1);
    d_wgrad.Upload(w_seed);
    d_lgrad.Upload(l_seed);
    DevBuf<ComputeType> d_de(rows * hidden);
    DevBuf<ComputeType> d_dy(ToStorage(dy));
    nanochat::kernels::SmearBackward(params, d_emb.ptr, d_w.ptr, d_sig.ptr,
                                     d_dy.ptr, d_wgrad.ptr, d_lgrad.ptr,
                                     d_de.ptr);
    nanochat::kernels::Synchronize();

    std::vector<float> ref_de(rows * hidden, 0.0f);
    std::vector<float> ref_wgrad = w_seed;
    std::vector<float> ref_lgrad = l_seed;
    SmearBackwardRef(params, emb, gate_w, ref_sig.data(), dy, ref_wgrad.data(),
                     ref_lgrad.data(), ref_de.data());
    CheckVectorClose(FromStorage(d_de.Download()), ref_de, 1e-4, "bwd de");
    CheckVectorClose(d_wgrad.Download(), ref_wgrad, 1e-4, "bwd gate_w_grad");
    CheckVectorClose(d_lgrad.Download(), ref_lgrad, 1e-4, "bwd lambda_grad");

    // Central differences of the reference forward, against the reference
    // backward: the definition has to be its own gradient before the kernel is
    // allowed to claim the same answer.
    const auto loss_at = [&](const std::vector<float>& emb_v) {
      std::vector<float> out(rows * hidden, 0.0f);
      SmearForwardRef(params, emb_v, gate_w, out.data(), nullptr);
      double total = 0.0;
      for (std::size_t i = 0; i < out.size(); ++i) total += out[i] * dy[i];
      return total;
    };
    CheckFiniteDifference(emb, ref_de, loss_at, 2e-3, "de");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("smear gpu test\n");
  TestForwardBackward();
  if (Failures() != 0) {
    std::printf("smear: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("smear: all checks passed\n");
  return 0;
}
