// T1 GPU correctness for RowReduce, the per-segment scan of the primitive tier
// (include/nanochat/kernels.h, seam 2). Tiny shapes, one multi-block shape, the
// interleaved descriptor the stride field allows, and the two shapes the three
// consumers need: a contiguous row (RmsNorm, Classifier with an offset) and a
// head_dim slice inside a packed row (QkPrep).
//
// The reference below is the descriptor read literally, written once in double
// so the comparison is against the seam's definition and not against a second
// guess at the indexing. See docs/testing.md.

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::ReduceOp;
using nanochat::RowReduceParams;
using nanochat::dev::CheckClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// The descriptor's definition, element by element: segment `s` starts at
// `offset + s * count * stride` and holds `count` elements spaced by `stride`.
// The accumulation is double, which is what the CPU reference does.
std::vector<double> RefRowReduce(const RowReduceParams& p,
                                 const std::vector<float>& a) {
  std::vector<double> out(static_cast<std::size_t>(p.n_out), 0.0);
  for (int s = 0; s < p.n_out; ++s) {
    const long long base =
        p.offset + static_cast<long long>(s) * p.count * p.stride;
    if (p.count <= 0) {
      out[static_cast<std::size_t>(s)] =
          (p.op == ReduceOp::kSumSq || p.op == ReduceOp::kSum)
              ? 0.0
              : -std::numeric_limits<double>::infinity();
      continue;
    }
    // Two passes for the max-stable log-sum-exp; one pass for the rest.
    double running = 0.0;
    double maximum = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < p.count; ++j) {
      const double v = a[static_cast<std::size_t>(base + j * p.stride)];
      switch (p.op) {
        case ReduceOp::kSumSq:
          running += v * v;
          break;
        case ReduceOp::kSum:
          running += v;
          break;
        case ReduceOp::kMax:
        case ReduceOp::kLogSumExp:
          if (v > maximum) maximum = v;
          break;
      }
    }
    switch (p.op) {
      case ReduceOp::kSumSq:
      case ReduceOp::kSum:
        out[static_cast<std::size_t>(s)] = running;
        break;
      case ReduceOp::kMax:
        out[static_cast<std::size_t>(s)] = maximum;
        break;
      case ReduceOp::kLogSumExp: {
        double sum = 0.0;
        for (int j = 0; j < p.count; ++j) {
          const double v = a[static_cast<std::size_t>(base + j * p.stride)];
          sum += std::exp(v - maximum);
        }
        out[static_cast<std::size_t>(s)] = maximum + std::log(sum);
        break;
      }
    }
  }
  return out;
}

// Runs one descriptor over one host buffer and compares every output.
void RunCase(const RowReduceParams& params, const std::vector<float>& a,
             const char* name, double tol = 2e-5) {
  DevBuf<ComputeType> dev_a(ToStorage(a));
  DevBuf<float> dev_out(params.n_out > 0 ? params.n_out : 1);
  // A sentinel no reference value equals, so a segment the kernel forgets to
  // write is caught instead of comparing equal to a zeroed buffer.
  dev_out.Upload(
      std::vector<float>(static_cast<std::size_t>(dev_out.count), -1234.5f));
  nanochat::kernels::RowReduce(params, dev_a.ptr, dev_out.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> got = dev_out.Download();
  const std::vector<double> want = RefRowReduce(params, a);
  for (int s = 0; s < params.n_out; ++s) {
    const char* op_name = "?";
    switch (params.op) {
      case ReduceOp::kSumSq:
        op_name = "sumsq";
        break;
      case ReduceOp::kSum:
        op_name = "sum";
        break;
      case ReduceOp::kMax:
        op_name = "max";
        break;
      case ReduceOp::kLogSumExp:
        op_name = "lse";
        break;
    }
    char label[160];
    std::snprintf(label, sizeof(label), "%s %s seg %d", name, op_name, s);
    const double expected = want[static_cast<std::size_t>(s)];
    if (std::isinf(expected)) {
      nanochat::dev::Check(std::isinf(got[static_cast<std::size_t>(s)]) &&
                               got[static_cast<std::size_t>(s)] < 0.0f,
                           label);
    } else {
      CheckClose(got[static_cast<std::size_t>(s)], expected, tol, label);
    }
  }
}

std::vector<float> RandomVec(int n, Rng* rng, float scale) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    v[static_cast<std::size_t>(i)] = rng->Next() * scale;
  return v;
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("row_reduce gpu test\n");
  Rng rng;

  // --- the contiguous row: RmsNorm and the Classifier's shape --------------
  // //
  {
    const int rows = 5;
    const int dim = 24;
    const std::vector<float> a = RandomVec(rows * dim, &rng, 1.0f);
    for (ReduceOp op : {ReduceOp::kSumSq, ReduceOp::kSum, ReduceOp::kMax}) {
      RowReduceParams p;
      p.n_out = rows;
      p.count = dim;
      p.stride = 1;
      p.offset = 0;
      p.op = op;
      RunCase(p, a, "rows");
    }
  }

  // --- the log-sum-exp: a large offset makes the shift necessary -----------
  // //
  {
    const int rows = 3;
    const int dim = 40;
    std::vector<float> a = RandomVec(rows * dim, &rng, 1.0f);
    for (float& v : a) v += 30.0f;  // exp(30) overflows float without a shift
    RowReduceParams p;
    p.n_out = rows;
    p.count = dim;
    p.op = ReduceOp::kLogSumExp;
    RunCase(p, a, "lse shifted");
  }

  // --- the QkPrep shape: one head_dim slice of a packed row ----------------
  // //
  {
    const int rows = 4;
    const int heads = 3;
    const int head_dim = 8;  // the packed row is heads * head_dim wide
    const std::vector<float> a = RandomVec(rows * heads * head_dim, &rng, 1.0f);
    RowReduceParams p;
    p.n_out = rows * heads;  // the packed heads make the segments consecutive
    p.count = head_dim;
    p.stride = 1;
    p.offset = 0;
    p.op = ReduceOp::kSumSq;
    RunCase(p, a, "head slice");
  }

  // --- the descriptor's offset and stride ----------------------------------
  // //
  {
    // Every third element, starting at one: the interleaved case the descriptor
    // allows and the one an offset row slice uses.
    const int n_out = 3;
    const int count = 5;
    const std::int64_t stride = 2;
    const std::int64_t offset = 1;
    const std::vector<float> a = RandomVec(
        static_cast<int>(offset) + n_out * count * static_cast<int>(stride) + 4,
        &rng, 1.0f);
    for (ReduceOp op : {ReduceOp::kSumSq, ReduceOp::kSum, ReduceOp::kMax}) {
      RowReduceParams p;
      p.n_out = n_out;
      p.count = count;
      p.stride = stride;
      p.offset = offset;
      p.op = op;
      RunCase(p, a, "strided", 1e-5);
    }
  }

  // --- an offset slice of a wider row, the Classifier's vocabulary window --
  // //
  {
    const int rows = 2;
    const int padded = 32;
    const int window = 12;  // the vocabulary slice, starting at column 4
    const std::vector<float> a = RandomVec(rows * padded, &rng, 1.0f);
    RowReduceParams p;
    p.count = window;
    p.stride = 1;
    p.op = ReduceOp::kLogSumExp;
    // The descriptor's segment pitch is `count * stride`, and a row pitch of
    // `padded` is wider than that, so a caller whose rows are not the segment
    // pitch makes one call per row. That is the shape the Classifier uses when
    // it reduces only the real vocabulary columns of a padded row.
    for (int r = 0; r < rows; ++r) {
      RowReduceParams row = p;
      row.n_out = 1;
      row.offset = static_cast<std::int64_t>(r) * padded + 4;
      RunCase(row, a, "vocab window", 2e-5);
    }
  }

  // --- a segment wider than one block ------------------------------------- //
  {
    const int n_out = 2;
    const int count = 4096;  // the block caps at 256, so the scan strides
    const std::vector<float> a = RandomVec(n_out * count, &rng, 0.5f);
    for (ReduceOp op :
         {ReduceOp::kSumSq, ReduceOp::kSum, ReduceOp::kLogSumExp}) {
      RowReduceParams p;
      p.n_out = n_out;
      p.count = count;
      p.op = op;
      RunCase(p, a, "wide segment", 1e-4);
    }
  }

  // --- degenerate segments -------------------------------------------------
  // //
  {
    // One element: the block still reduces, and the answer is that element
    // squared, itself, or itself again.
    const std::vector<float> a = RandomVec(4, &rng, 1.0f);
    RowReduceParams p;
    p.n_out = 1;
    p.count = 1;
    p.op = ReduceOp::kSumSq;
    RunCase(p, a, "single element", 1e-6);

    // No element at all: a sum is zero and a maximum (or log-sum-exp) is minus
    // infinity, never a NaN and never a stale buffer value.
    for (ReduceOp op : {ReduceOp::kSumSq, ReduceOp::kSum, ReduceOp::kMax,
                        ReduceOp::kLogSumExp}) {
      RowReduceParams empty = p;
      empty.n_out = 3;
      empty.count = 0;
      empty.op = op;
      RunCase(empty, a, "empty segment");
    }
  }

  // --- no output requested -------------------------------------------------
  // //
  {
    RowReduceParams p;
    p.n_out = 0;
    p.count = 4;
    DevBuf<float> out(1);
    out.Upload({7.0f});
    nanochat::kernels::RowReduce(p, nullptr, out.ptr);
    nanochat::kernels::Synchronize();
    const std::vector<float> got = out.Download();
    nanochat::dev::Check(got[0] == 7.0f,
                         "n_out == 0 leaves the output buffer untouched");
  }

  if (Failures() != 0) {
    std::printf("row_reduce: %d failure(s)\n", Failures());
    return 1;
  }
  std::printf("row_reduce: ok\n");
  return 0;
}
