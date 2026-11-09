// The measurement rows of the GEMM candidate sweep (docs/kernels.md section
// 11.5).
//
// The 29 rows answer a question about a kernel and not about a model. The five
// families measure the things a production sample cannot: the launch floor, the
// work scaling, the aspect ratios, the batch, the ragged extents, and the four
// operand forms. `align:offset` is the general path, whose pointer leaves the
// sixteen-byte boundary that the CUTLASS candidates need.
//
// Only the closed list of `dispatch.cc` feeds the decision table of
// `cutlass_rules.h`. These rows decide whether a candidate is fit to enter a
// sweep, and they are the record a later tuning session reads.
//
// Every derived quantity is computed here from the row's own fields, so the
// host test `bench_shapes_test.cc` checks the list against section 11.5's
// arithmetic without a card.

#include "backends/cuda/gemm/bench_shapes.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

namespace {

// The rows, in the order section 11.5 lists them. The work column of that table
// and `BenchShapeFlops` must agree, and `bench_shapes_test.cc` states the
// agreement for every row.
constexpr BenchShape kRows[] = {
    // The scaling family: one square probe for each decade of work.
    {"scale:16", 16, 16, 16, 1, false, false, false, 0, 0, "the launch floor"},
    {"scale:64", 64, 64, 64, 1, false, false, false, 0, 0,
     "a quarter of one tile"},
    {"scale:256", 256, 256, 256, 1, false, false, false, 0, 0,
     "below the 128 by 256 tile"},
    {"scale:1024", 1024, 1024, 1024, 1, false, false, false, 0, 0,
     "one partial wave"},
    {"scale:2048", 2048, 2048, 2048, 1, false, false, false, 0, 0,
     "one full wave"},
    {"scale:4096", 4096, 4096, 4096, 1, false, false, false, 0, 0,
     "four waves"},
    {"scale:8192", 8192, 8192, 8192, 1, false, false, false, 0, 0,
     "an operand above the second-level cache"},
    {"scale:16384", 16384, 16384, 16384, 1, false, false, false, 0, 0,
     "a large matrix, 512 MiB an operand"},
    {"scale:32768", 32768, 32768, 32768, 1, false, false, false, 0, 0,
     "the device memory ceiling"},
    // The aspect family: one extent far from the other two.
    {"aspect:thin-m", 128, 4096, 4096, 1, false, false, false, 0, 0,
     "too few tiles in M"},
    {"aspect:thin-n", 4096, 128, 4096, 1, false, false, false, 0, 0,
     "too few tiles in N"},
    {"aspect:deep-k", 1024, 1024, 32768, 1, false, false, false, 0, 0,
     "a long k loop and few tiles"},
    {"aspect:thin-k", 4096, 4096, 64, 1, false, false, false, 0, 0,
     "one k slice"},
    {"aspect:tall", 16384, 256, 2048, 1, false, false, false, 0, 0,
     "a decoder-like shape"},
    // The batch family: one slice repeated.
    {"batch:1", 512, 512, 512, 1, false, false, false, 0, 0, "one slice"},
    {"batch:8", 512, 512, 512, 8, false, false, false, 0, 0, "a few slices"},
    {"batch:40", 512, 512, 512, 40, false, false, false, 0, 0,
     "the Muon batch"},
    {"batch:128", 512, 512, 512, 128, false, false, false, 0, 0, "many slices"},
    {"batch:muon-tall", 4096, 4096, 1024, 4, true, false, false, 0, 0,
     "the transposed-A Muon form"},
    {"batch:muon-wide", 1024, 4096, 4096, 4, false, false, false, 0, 0,
     "the plain Muon and B X forms"},
    // The ragged family: an extent that is not a multiple of a tile.
    {"ragged:m", 1000, 1000, 1024, 1, false, false, false, 0, 0,
     "a partial M tile"},
    {"ragged:k", 1024, 1024, 1000, 1, false, false, false, 0, 0,
     "a partial k slice"},
    {"ragged:lda", 1024, 1024, 1024, 1, false, false, false, 1, 0,
     "a leading dimension off the tile"},
    // The form family: the four operand orientations and the fp32 store.
    {"form:nn", 2048, 2048, 2048, 1, false, false, false, 0, 0,
     "the first operand form"},
    {"form:tn", 2048, 2048, 2048, 1, true, false, false, 0, 0,
     "the second operand form"},
    {"form:nt", 2048, 2048, 2048, 1, false, true, false, 0, 0,
     "the third operand form"},
    {"form:tt", 2048, 2048, 2048, 1, true, true, false, 0, 0,
     "the fourth operand form"},
    {"form:fp32-c", 2048, 2048, 2048, 1, false, false, true, 0, 0,
     "the weight-gradient store"},
    // The alignment row: one element off the boundary.
    {"align:offset", 1024, 1024, 1024, 1, false, false, false, 0, 1,
     "the general path"},
};

constexpr int kRowCount = static_cast<int>(sizeof(kRows) / sizeof(kRows[0]));

// The rows of A and B, which a transposed operand takes from its other extent.
int RowsA(const BenchShape& shape) {
  return shape.transpose_a ? shape.k : shape.m;
}
int RowsB(const BenchShape& shape) {
  return shape.transpose_b ? shape.n : shape.k;
}

// The tight leading dimension of A, before the row's extra columns.
int TightLda(const BenchShape& shape) {
  return shape.transpose_a ? shape.m : shape.k;
}
int TightLdb(const BenchShape& shape) {
  return shape.transpose_b ? shape.k : shape.n;
}

// The width of one element of C. The operand width is
// `kBenchOperandElementBytes`, which the header states and explains.
std::size_t OutputElementBytes(bool fp32_c) { return fp32_c ? 4 : 2; }

}  // namespace

int BenchShapeCount() { return kRowCount; }

const BenchShape& BenchShapeAt(int index) {
  static const BenchShape kEmpty{"", 0, 0, 0, 0, false, false, false, 0, 0, ""};
  if (index < 0 || index >= kRowCount) return kEmpty;
  return kRows[index];
}

ResolvedGemm BenchShapeGemm(int index) {
  const BenchShape& shape = BenchShapeAt(index);
  GemmParams params;
  params.m = shape.m;
  params.n = shape.n;
  params.k = shape.k;
  params.batch_count = shape.batch_count;
  params.transpose_a = shape.transpose_a;
  params.transpose_b = shape.transpose_b;
  params.lda = TightLda(shape) + shape.lda_extra;
  params.ldb = TightLdb(shape);
  // A zero leading dimension of C asks `ResolveGemm` for the tight one, which
  // is the row major width of the result.
  params.ldc = 0;
  return ResolveGemm(params, shape.fp32_c);
}

std::string BenchShapeExtent(const BenchShape& shape) {
  std::string text = std::to_string(shape.m) + "x" + std::to_string(shape.n) +
                     "x" + std::to_string(shape.k);
  if (shape.batch_count != 1) {
    text += " batch " + std::to_string(shape.batch_count);
  }
  if (shape.transpose_a || shape.transpose_b) {
    text += shape.transpose_a ? (shape.transpose_b ? " tt" : " tn") : " nt";
  }
  if (shape.fp32_c) text += " fp32 out";
  if (shape.lda_extra != 0) {
    text += " lda+=" + std::to_string(shape.lda_extra);
  }
  if (shape.pointer_offset != 0) {
    text += " offset " + std::to_string(shape.pointer_offset);
  }
  return text;
}

double BenchShapeFlops(const BenchShape& shape) {
  return 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
         static_cast<double>(shape.k) * static_cast<double>(shape.batch_count);
}

std::size_t BenchShapeOperandABytes(const BenchShape& shape) {
  const std::size_t elements =
      static_cast<std::size_t>(shape.batch_count) *
      static_cast<std::size_t>(RowsA(shape)) *
      static_cast<std::size_t>(TightLda(shape) + shape.lda_extra);
  return elements * kBenchOperandElementBytes;
}

std::size_t BenchShapeOperandBBytes(const BenchShape& shape) {
  const std::size_t elements = static_cast<std::size_t>(shape.batch_count) *
                               static_cast<std::size_t>(RowsB(shape)) *
                               static_cast<std::size_t>(TightLdb(shape));
  return elements * kBenchOperandElementBytes;
}

std::size_t BenchShapeOperandBytes(const BenchShape& shape) {
  return BenchShapeOperandABytes(shape) + BenchShapeOperandBBytes(shape);
}

std::size_t BenchShapeOutputElements(const BenchShape& shape) {
  // C is row major in every row, so its leading dimension is the width of the
  // product and not the leading dimension of B.
  return static_cast<std::size_t>(shape.batch_count) *
         static_cast<std::size_t>(shape.m) * static_cast<std::size_t>(shape.n);
}

std::size_t BenchShapeOutputBytes(const BenchShape& shape) {
  return BenchShapeOutputElements(shape) * OutputElementBytes(shape.fp32_c);
}

std::size_t BenchShapeBytes(const BenchShape& shape) {
  // The operands, the result, and the reference copy the check reads. The
  // pointer offset costs one element of slack.
  return BenchShapeOperandBytes(shape) + 2 * BenchShapeOutputBytes(shape) +
         2 * static_cast<std::size_t>(shape.pointer_offset) *
             kBenchOperandElementBytes;
}

long long BenchShapeTileCount(const BenchShape& shape) {
  const long long tiles_m = (shape.m + 127) / 128;
  const long long tiles_n = (shape.n + 255) / 256;
  return tiles_m * tiles_n * static_cast<long long>(shape.batch_count);
}

std::size_t BenchShapeFillElements(const BenchShape& shape) {
  const std::size_t operands =
      BenchShapeOperandBytes(shape) / kBenchOperandElementBytes;
  return operands + BenchShapeOutputElements(shape);
}

std::size_t BenchDeviceBudget(std::size_t free_bytes) {
  const double share = 1.0 - kBenchDeviceReserve;
  if (!(share > 0.0)) return 0;
  return static_cast<std::size_t>(static_cast<double>(free_bytes) * share);
}

std::string BenchSizeSkipReason(std::size_t bytes, std::size_t budget_bytes) {
  if (bytes <= budget_bytes) return std::string();
  return "needs " + std::to_string(bytes) + " B, the budget is " +
         std::to_string(budget_bytes) + " B";
}

std::string BenchShapeSkipReason(const BenchShape& shape,
                                 std::size_t budget_bytes) {
  return BenchSizeSkipReason(BenchShapeBytes(shape), budget_bytes);
}

int BenchIterationsFor(double flops, double peak_flops) {
  if (!(peak_flops > 0.0) || !(flops > 0.0)) return kBenchMinIters;
  const double ideal_seconds = flops / peak_flops;
  if (!std::isfinite(ideal_seconds) || ideal_seconds <= 0.0) {
    return kBenchMinIters;
  }
  const double count = std::ceil(kBenchRoundSeconds / ideal_seconds);
  if (count < static_cast<double>(kBenchMinIters)) return kBenchMinIters;
  if (count > static_cast<double>(kBenchMaxIters)) return kBenchMaxIters;
  return static_cast<int>(count);
}

double CandidateRelativeError(double diff_sumsq, double ref_sumsq) {
  if (!(ref_sumsq > 0.0)) {
    return diff_sumsq > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  }
  if (!(diff_sumsq > 0.0)) return 0.0;
  return std::sqrt(diff_sumsq / ref_sumsq);
}

bool CandidateMatches(double diff_sumsq, double ref_sumsq) {
  if (!(diff_sumsq >= 0.0)) return false;  // a reduction that did not run
  return CandidateRelativeError(diff_sumsq, ref_sumsq) <=
         kCandidateMaxRelativeError;
}

}  // namespace kernels
}  // namespace nanochat
