// T1 GPU correctness for SpanReduce, the device-wide ragged reduction of the
// primitive tier (include/nanochat/kernels.h, seam 2). The span table and the
// counts are device memory, because the kernel reads them itself and the caller
// never round-trips them to the host.
//
// Covered: several ragged spans, an empty span, a span wider than one block,
// the plain sum and the dot form, the scale, and both `accumulate` settings.
// The reference is the entry point's definition in double, so the comparison is
// against what the seam says and not against a second guess at the kernel.
// See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::SpanReduceParams;
using nanochat::dev::CheckClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// A span of `n` random values in [-scale, scale). The shared helper generates
// in [-1, 1); the scale keeps a long span's double accumulation from swamping
// the shorter ones.
std::vector<float> RandomSpan(int n, Rng* rng, float scale) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    // The same xorshift stream `RandomVec` uses, scaled here.
    v[static_cast<std::size_t>(i)] = scale * rng->Next();
  }
  return v;
}

// The definition: the sum of the spans, or the dot of the two ragged lists,
// scaled by `scale` and written to or added to `out`. Cross-span order is
// unspecified, so the reference sums in double and the gate is a tolerance.
double RefSpanReduce(const std::vector<std::vector<float>>& spans,
                     const std::vector<std::vector<float>>* other, float scale,
                     float initial, bool accumulate) {
  double total = 0.0;
  for (std::size_t s = 0; s < spans.size(); ++s) {
    for (std::size_t i = 0; i < spans[s].size(); ++i) {
      const double v = spans[s][i];
      total += other != nullptr ? v * (*other)[s][i] : v;
    }
  }
  const double scaled = static_cast<double>(scale) * total;
  return accumulate ? static_cast<double>(initial) + scaled : scaled;
}

// One case: `spans` reduced on the device against the reference. `other` is
// null for the plain sum. `initial` seeds the output buffer, which is what
// `accumulate` adds to.
void RunCase(const std::vector<std::vector<float>>& spans,
             const std::vector<std::vector<float>>* other, float scale,
             bool accumulate, const char* name, double tol = 2e-5) {
  const int num_spans = static_cast<int>(spans.size());
  std::vector<std::unique_ptr<DevBuf<ComputeType>>> buffers;
  std::vector<const ComputeType*> pointers;
  std::vector<std::unique_ptr<DevBuf<ComputeType>>> other_buffers;
  std::vector<const ComputeType*> other_pointers;
  std::vector<int> counts;
  for (const std::vector<float>& span : spans) {
    buffers.push_back(std::make_unique<DevBuf<ComputeType>>(ToStorage(span)));
    pointers.push_back(buffers.back()->ptr);
    counts.push_back(static_cast<int>(span.size()));
  }
  if (other != nullptr) {
    for (const std::vector<float>& span : *other) {
      other_buffers.push_back(
          std::make_unique<DevBuf<ComputeType>>(ToStorage(span)));
      other_pointers.push_back(other_buffers.back()->ptr);
    }
  }

  // The span table and the counts live on the device: `GradNorm`'s and
  // `ScalarDot`'s callers follow the same rule.
  DevBuf<const ComputeType*> span_table(pointers);
  DevBuf<const ComputeType*> other_table(
      other != nullptr ? other_pointers : std::vector<const ComputeType*>());
  DevBuf<int> count_table(counts);
  const float initial = 0.25f;
  DevBuf<float> out(1);
  out.Upload({initial});

  SpanReduceParams params;
  params.num_spans = num_spans;
  params.counts = count_table.ptr;
  params.scale = scale;
  params.accumulate = accumulate;
  nanochat::kernels::SpanReduce(params, span_table.ptr,
                                other != nullptr ? other_table.ptr : nullptr,
                                out.ptr);
  nanochat::kernels::Synchronize();

  const double want = RefSpanReduce(spans, other, scale, initial, accumulate);
  CheckClose(out.Download()[0], want, tol, name);
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("span_reduce gpu test\n");
  Rng rng;

  // --- the plain sum over ragged spans -------------------------------------
  // //
  {
    const std::vector<std::vector<float>> spans = {
        RandomSpan(7, &rng, 1.0f),
        RandomSpan(1, &rng, 1.0f),  // the shortest non-empty span
        RandomSpan(13, &rng,
                   1.0f),  // and a longer one, so no length is assumed
    };
    RunCase(spans, nullptr, 1.0f, false, "ragged sum");
    RunCase(spans, nullptr, -1.0f, false, "ragged sum scaled", 1e-5);
    RunCase(spans, nullptr, 2.5f, true, "ragged sum accumulated", 1e-5);
  }

  // --- an empty span contributes zero and does not abort the block ---------
  // //
  {
    const std::vector<std::vector<float>> spans = {
        RandomSpan(4, &rng, 1.0f),
        std::vector<float>(),  // a parameter with no element
        RandomSpan(5, &rng, 1.0f),
        std::vector<float>(),
    };
    RunCase(spans, nullptr, 1.0f, false, "empty spans");
  }

  // --- a span wider than one 256-thread block ------------------------------
  // //
  {
    const std::vector<std::vector<float>> spans = {
        RandomSpan(5000, &rng, 0.5f),
        RandomSpan(3, &rng, 0.5f),
    };
    RunCase(spans, nullptr, 1.0f, false, "wide span", 1e-4);
    const std::vector<std::vector<float>> other = {
        RandomSpan(5000, &rng, 0.5f),
        RandomSpan(3, &rng, 0.5f),
    };
    RunCase(spans, &other, 1.0f, false, "wide dot", 1e-4);
  }

  // --- the dot form: one entry point serves a sum and a dot ----------------
  // //
  {
    const std::vector<std::vector<float>> spans = {
        RandomSpan(9, &rng, 1.0f),
        RandomSpan(6, &rng, 1.0f),
    };
    const std::vector<std::vector<float>> other = {
        RandomSpan(9, &rng, 1.0f),
        RandomSpan(6, &rng, 1.0f),
    };
    RunCase(spans, &other, 1.0f, false, "dot");
    RunCase(spans, &other, 0.5f, true, "dot accumulated", 1e-5);
  }

  // --- a long span list: the chunked launch ------------------------------- //
  {
    // More spans than one block budget covers, so the entry point reduces the
    // list in more than one chunk into the same accumulator.
    std::vector<std::vector<float>> spans;
    spans.reserve(300);
    for (int s = 0; s < 300; ++s) {
      spans.push_back(RandomSpan(1 + s % 17, &rng, 0.25f));
    }
    RunCase(spans, nullptr, 1.0f, false, "many spans", 1e-4);
  }

  // --- nothing to reduce -------------------------------------------------- //
  {
    DevBuf<float> out(1);
    out.Upload({3.5f});
    SpanReduceParams params;
    params.num_spans = 0;
    params.counts = nullptr;
    nanochat::kernels::SpanReduce(params, nullptr, nullptr, out.ptr);
    nanochat::kernels::Synchronize();
    bool ok = out.Download()[0] == 3.5f;
    // A null span table or a null count table is a no-op too: the entry point
    // never dereferences what it was not given.
    DevBuf<int> counts(1);
    counts.Upload({4});
    params.num_spans = 1;
    params.counts = counts.ptr;
    nanochat::kernels::SpanReduce(params, nullptr, nullptr, out.ptr);
    nanochat::kernels::Synchronize();
    ok = ok && out.Download()[0] == 3.5f;
    nanochat::dev::Check(ok, "no spans leaves the output buffer untouched");
  }

  if (Failures() != 0) {
    std::printf("span_reduce: %d failure(s)\n", Failures());
    return 1;
  }
  std::printf("span_reduce: ok\n");
  return 0;
}
