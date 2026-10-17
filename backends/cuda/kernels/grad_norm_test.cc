// T1 GPU correctness test for the GradNorm family: the L2 norm over a set of
// gradient spans and the published clip scale. The buffers must be left
// untouched, because the caller folds the scale into its own update. Tiny
// shapes, one multi-block shape, and a multi-span shape; no backward. Compared
// against the host reference in row_ref.h. See docs/testing.md.

#include <cstdio>
#include <memory>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/row_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::dev::CheckClose;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;

// Runs one case over a set of host spans and compares the norm, the scale, and
// every span's contents against the reference.
void RunCase(const std::vector<std::vector<float>>& spans, float clip,
             const char* name) {
  const int num_spans = static_cast<int>(spans.size());
  std::vector<std::unique_ptr<DevBuf<float>>> buffers;
  std::vector<const float*> pointers;
  std::vector<int> counts;
  buffers.reserve(spans.size());
  for (const std::vector<float>& span : spans) {
    buffers.push_back(std::make_unique<DevBuf<float>>(span));
    pointers.push_back(buffers.back()->ptr);
    counts.push_back(static_cast<int>(span.size()));
  }
  // The span table is device memory because the kernel reads it.
  DevBuf<const float*> grad_table(pointers);
  DevBuf<int> count_table(counts);
  DevBuf<float> norm(1);
  DevBuf<float> scale(1);
  nanochat::kernels::GradNorm(grad_table.ptr, count_table.ptr, num_spans, clip,
                              norm.ptr, scale.ptr);
  nanochat::kernels::Synchronize();

  float ref_scale = 1.0f;
  const float ref_norm =
      nanochat::dev::rowref::GradNorm(spans, clip, &ref_scale);
  CheckClose(norm.Download()[0], ref_norm, 1e-4, name);
  CheckClose(scale.Download()[0], ref_scale, 1e-4, name);
  // Nothing is rewritten: the scale is applied by the caller.
  for (std::size_t s = 0; s < spans.size(); ++s) {
    CheckVectorClose(buffers[s]->Download(), spans[s], 1e-6, name);
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("grad_norm gpu test\n");

  // {3, 4} has norm 5: clipped when the threshold is below it, untouched when
  // above it, and untouched when the threshold disables clipping.
  RunCase({{3.0f, 4.0f}}, 2.5f, "clip below");
  RunCase({{3.0f, 4.0f}}, 10.0f, "clip above");
  RunCase({{3.0f, 4.0f}}, 0.0f, "clip disabled");
  RunCase({{3.0f, 4.0f}}, -1.0f, "clip negative");

  // A multi-block reduction: 1000 elements span more than one 256-thread block.
  Rng rng;
  const std::vector<float> big = RandomVec(1000, &rng);
  RunCase({big}, 1e-3f, "large clipped");
  RunCase({big}, 1e9f, "large unclipped");

  // Several spans of different lengths, including a short one and an empty one:
  // the norm is over the concatenation and every span is covered.
  const std::vector<float> mid = RandomVec(517, &rng);
  const std::vector<float> small = RandomVec(3, &rng);
  RunCase({big, small, mid}, 1e9f, "multi-span unclipped");
  RunCase({big, small, mid}, 0.5f, "multi-span clipped");
  RunCase({small, {}, mid}, 1e9f, "empty span");
  RunCase({{1.0f}, {2.0f}, {3.0f}}, 100.0f, "many small spans");

  // An empty span set is a no-op and must not launch or read either buffer.
  DevBuf<float> norm(1);
  DevBuf<float> scale(1);
  std::vector<const float*> none;
  std::vector<int> no_counts;
  DevBuf<const float*> empty_grads(none);
  DevBuf<int> empty_counts(no_counts);
  nanochat::kernels::GradNorm(empty_grads.ptr, empty_counts.ptr, 0, 1.0f,
                              norm.ptr, scale.ptr);
  nanochat::kernels::GradNorm(nullptr, nullptr, 3, 1.0f, norm.ptr, scale.ptr);
  nanochat::kernels::Synchronize();

  // A null output pointer is accepted for either output.
  const std::vector<float> one = {3.0f, 4.0f};
  DevBuf<float> one_dev(one);
  DevBuf<const float*> one_grads(std::vector<const float*>{one_dev.ptr});
  DevBuf<int> one_counts(std::vector<int>{2});
  nanochat::kernels::GradNorm(one_grads.ptr, one_counts.ptr, 1, 1.0f, nullptr,
                              scale.ptr);
  nanochat::kernels::GradNorm(one_grads.ptr, one_counts.ptr, 1, 1.0f, norm.ptr,
                              nullptr);
  nanochat::kernels::Synchronize();

  if (Failures() != 0) {
    std::printf("grad_norm: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("grad_norm: all checks passed\n");
  return 0;
}
