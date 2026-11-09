// The host gate of the measurement rows (docs/kernels.md section 11.7 step 1).
//
// The rows state extents and nothing else, and every other quantity is derived.
// This test checks the derivation against section 11.5's own arithmetic, so a
// mistyped row fails here and not on a card. Four things are checked:
//
//   1. The list holds the rows of that section, by name, in its order.
//   2. Each row's operation count is the figure the section's work column
//      states, to the precision it states it.
//   3. The derived tile count, operand bytes, output elements, and fill size
//   are
//      consistent with the row's own extents.
//   4. The budget skip, the iteration count, and the check's predicate behave
//   as
//      the section describes, including the case that must be rejected.
//
// Hermetic and CPU only: the header names no CUDA type, so this rides the
// default `-gpu` test leg.

#include "backends/cuda/gemm/bench_shapes.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using nanochat::kernels::BenchDeviceBudget;
using nanochat::kernels::BenchIterationsFor;
using nanochat::kernels::BenchShape;
using nanochat::kernels::BenchShapeAt;
using nanochat::kernels::BenchShapeBytes;
using nanochat::kernels::BenchShapeCount;
using nanochat::kernels::BenchShapeExtent;
using nanochat::kernels::BenchShapeFillElements;
using nanochat::kernels::BenchShapeFlops;
using nanochat::kernels::BenchShapeGemm;
using nanochat::kernels::BenchShapeOperandABytes;
using nanochat::kernels::BenchShapeOperandBBytes;
using nanochat::kernels::BenchShapeOperandBytes;
using nanochat::kernels::BenchShapeOutputBytes;
using nanochat::kernels::BenchShapeOutputElements;
using nanochat::kernels::BenchShapeSkipReason;
using nanochat::kernels::BenchShapeTileCount;
using nanochat::kernels::BenchSizeSkipReason;
using nanochat::kernels::CandidateMatches;
using nanochat::kernels::CandidateRelativeError;
using nanochat::kernels::kBenchMaxIters;
using nanochat::kernels::kBenchMinIters;
using nanochat::kernels::kBenchOperandElementBytes;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

// Section 11.5's own work column, in FLOP, as the section rounds it.
struct RowWork {
  const char* name;
  double flops;
};

constexpr RowWork kExpected[] = {
    {"scale:16", 8.2e3},          {"scale:64", 5.24e5},
    {"scale:256", 3.35e7},        {"scale:1024", 2.15e9},
    {"scale:2048", 1.72e10},      {"scale:4096", 1.37e11},
    {"scale:8192", 1.10e12},      {"scale:16384", 8.80e12},
    {"scale:32768", 7.04e13},     {"aspect:thin-m", 4.30e9},
    {"aspect:thin-n", 4.30e9},    {"aspect:deep-k", 6.87e10},
    {"aspect:thin-k", 2.15e9},    {"aspect:tall", 1.72e10},
    {"batch:1", 2.68e8},          {"batch:8", 2.15e9},
    {"batch:40", 1.07e10},        {"batch:128", 3.44e10},
    {"batch:muon-tall", 1.37e11}, {"batch:muon-wide", 1.37e11},
    {"ragged:m", 2.05e9},         {"ragged:k", 2.10e9},
    {"ragged:lda", 2.15e9},       {"form:nn", 1.72e10},
    {"form:tn", 1.72e10},         {"form:nt", 1.72e10},
    {"form:tt", 1.72e10},         {"form:fp32-c", 1.72e10},
    {"align:offset", 2.15e9},
};

constexpr int kExpectedCount =
    static_cast<int>(sizeof(kExpected) / sizeof(kExpected[0]));

// True when `value` is `expected` within the two significant figures of the
// section's work column.
bool SameWork(double value, double expected) {
  if (!(expected > 0.0)) return value == expected;
  return std::fabs(value - expected) <= 0.01 * expected;
}

// The row family's name: the part of the row name before the colon.
std::string FamilyOf(const char* name) {
  const std::string text(name);
  const std::size_t colon = text.find(':');
  return colon == std::string::npos ? std::string() : text.substr(0, colon);
}

void CheckList() {
  Check(BenchShapeCount() == kExpectedCount,
        "the list holds the 29 rows of section 11.5, it holds " +
            std::to_string(BenchShapeCount()));
  for (int index = 0; index < kExpectedCount && index < BenchShapeCount();
       ++index) {
    Check(std::string(BenchShapeAt(index).name) == kExpected[index].name,
          "row " + std::to_string(index) + " is " + kExpected[index].name);
  }
  // The five families, and the one alignment row beside them.
  std::vector<std::string> families;
  for (int index = 0; index < BenchShapeCount(); ++index) {
    const std::string family = FamilyOf(BenchShapeAt(index).name);
    if (families.empty() || families.back() != family) {
      families.push_back(family);
    }
  }
  const std::vector<std::string> want = {"scale",  "aspect", "batch",
                                         "ragged", "form",   "align"};
  Check(families == want, "the six families are in the section's order");
}

void CheckWork() {
  for (int index = 0; index < kExpectedCount && index < BenchShapeCount();
       ++index) {
    const BenchShape& shape = BenchShapeAt(index);
    Check(SameWork(BenchShapeFlops(shape), kExpected[index].flops),
          std::string("the work of ") + shape.name + " is " +
              std::to_string(BenchShapeFlops(shape)));
  }
  // The derived count is exactly the definition, at one row that states both
  // factors plainly.
  const BenchShape& square = BenchShapeAt(3);  // scale:1024
  Check(BenchShapeFlops(square) == 2.0 * 1024.0 * 1024.0 * 1024.0,
        "the operation count is two for each element of the product");
}

void CheckDerived() {
  for (int index = 0; index < BenchShapeCount(); ++index) {
    const BenchShape& shape = BenchShapeAt(index);
    const std::string name(shape.name);
    const std::size_t element = kBenchOperandElementBytes;

    // The operand bytes are the two operands at their own leading dimensions.
    const int rows_a = shape.transpose_a ? shape.k : shape.m;
    const int rows_b = shape.transpose_b ? shape.n : shape.k;
    const int lda = (shape.transpose_a ? shape.m : shape.k) + shape.lda_extra;
    const int ldb = shape.transpose_b ? shape.k : shape.n;
    const std::size_t want =
        static_cast<std::size_t>(shape.batch_count) *
        (static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(lda) +
         static_cast<std::size_t>(rows_b) * static_cast<std::size_t>(ldb)) *
        element;
    Check(BenchShapeOperandBytes(shape) == want,
          name + ": the operand bytes are the two operands");
    Check(BenchShapeOperandABytes(shape) + BenchShapeOperandBBytes(shape) ==
              BenchShapeOperandBytes(shape),
          name + ": the operand bytes are A and B");
    Check(BenchShapeOperandABytes(shape) ==
              static_cast<std::size_t>(shape.batch_count) *
                  static_cast<std::size_t>(rows_a) *
                  static_cast<std::size_t>(lda) * element,
          name + ": A is one element for each of its own rows");
    Check(BenchShapeOperandBBytes(shape) ==
              static_cast<std::size_t>(shape.batch_count) *
                  static_cast<std::size_t>(rows_b) *
                  static_cast<std::size_t>(ldb) * element,
          name + ": B is one element for each of its own rows");

    // The output is the batch times the product's own extents, at the store
    // type the row states.
    const std::size_t out_elements =
        static_cast<std::size_t>(shape.m) * static_cast<std::size_t>(shape.n) *
        static_cast<std::size_t>(shape.batch_count);
    Check(BenchShapeOutputElements(shape) == out_elements,
          name + ": the output elements are the product's own extents");
    Check(
        BenchShapeOutputBytes(shape) == out_elements * (shape.fp32_c ? 4u : 2u),
        name + ": the output bytes follow the store type");

    // The workspace is the operands, the result, and the reference copy.
    Check(BenchShapeBytes(shape) ==
              BenchShapeOperandBytes(shape) + 2 * BenchShapeOutputBytes(shape) +
                  2 * static_cast<std::size_t>(shape.pointer_offset) * element,
          name + ": the workspace is the operands and two results");

    // The fill covers the operands and one result.
    Check(BenchShapeFillElements(shape) ==
              BenchShapeOperandBytes(shape) / element + out_elements,
          name + ": the fill covers the operands and one result");

    // The resolved call names the row's own extents, and its stride rule is the
    // one the row states.
    const nanochat::kernels::ResolvedGemm gemm = BenchShapeGemm(index);
    Check(gemm.m == shape.m && gemm.n == shape.n && gemm.k == shape.k &&
              gemm.batch_count == shape.batch_count,
          name + ": the resolved call holds the row's extents");
    Check(gemm.lda == lda && gemm.ldb == ldb && gemm.ldc == shape.n,
          name + ": the resolved call holds the row's leading dimensions");
    Check(gemm.transpose_a == shape.transpose_a &&
              gemm.transpose_b == shape.transpose_b &&
              gemm.fp32_c == shape.fp32_c,
          name + ": the resolved call holds the row's form");
    Check(!BenchShapeExtent(shape).empty(),
          name + ": the extent has a spelling");
  }

  // The tile count of a square probe is the coarse tile grid of one slice.
  const BenchShape& tile_grid = BenchShapeAt(5);  // scale:4096
  Check(BenchShapeTileCount(tile_grid) ==
            static_cast<long long>((4096 + 127) / 128) *
                static_cast<long long>((4096 + 255) / 256),
        "the tile count is the coarse grid of one slice");
  // A ragged row rounds up: 1000 by 1000 is 8 by 4 tiles.
  Check(BenchShapeTileCount(BenchShapeAt(20)) == 8 * 4,
        "a ragged M and N round up to whole tiles");
  // The batch multiplies the grid.
  Check(BenchShapeTileCount(BenchShapeAt(15)) == 4 * 2 * 8,
        "the batch multiplies the tile count");

  // The two rows whose form must break a rule of the CUTLASS candidates.
  const nanochat::kernels::ResolvedGemm ragged_lda = BenchShapeGemm(22);
  Check(ragged_lda.lda == ragged_lda.k + 1,
        "ragged:lda takes a leading dimension one element past the tight one");
  Check(BenchShapeAt(28).pointer_offset == 1,
        "align:offset moves the operand pointers by one element");
}

void CheckBudget() {
  // The largest row is 8 GiB, as section 11.5 states.
  const BenchShape& largest = BenchShapeAt(8);  // scale:32768
  Check(BenchShapeBytes(largest) >= std::size_t{8} * 1024 * 1024 * 1024 - 2,
        "the largest row needs at least 8 GiB: it needs " +
            std::to_string(BenchShapeBytes(largest)));
  const std::size_t big = std::size_t{8} * 1024 * 1024 * 1024;
  Check(BenchShapeSkipReason(largest, big).empty(),
        "the largest row fits a budget of its own size");
  Check(!BenchShapeSkipReason(largest, big / 2).empty(),
        "the largest row is skipped below its size");
  Check(BenchShapeSkipReason(BenchShapeAt(0), big).empty(),
        "a tiny row fits every budget");
  // The byte form answers the same question for a row that states its own
  // bytes, which is how a production row asks it.
  Check(BenchSizeSkipReason(BenchShapeBytes(largest), big).empty(),
        "the byte form accepts a row of its own size");
  Check(!BenchSizeSkipReason(BenchShapeBytes(largest) + 2, big).empty(),
        "the byte form skips a row past the budget");

  // The budget keeps a reserve of the free bytes.
  Check(BenchDeviceBudget(std::size_t{10} * 1024 * 1024 * 1024) ==
            std::size_t{8} * 1024 * 1024 * 1024,
        "the budget is four fifths of the free bytes");
}

void CheckIterations() {
  // A round runs at least 50 ms. The H100 declares 989 TFLOP/s at bf16, so a
  // 2.1 GFLOP row's call is 2 us and the ceiling bounds its count, while a
  // 70 TFLOP row is 71 ms and the floor bounds its count.
  const double peak = 989e12;
  Check(BenchIterationsFor(2.1e9, peak) == kBenchMaxIters,
        "a row whose call is far under the round takes the ceiling");
  Check(BenchIterationsFor(70.4e12, peak) == kBenchMinIters,
        "a row longer than the round takes the floor");
  // Between the two bounds the count follows the work: a 17 GFLOP row is 17 us,
  // so 50 ms of it is 2,900 calls, which the ceiling bounds.
  const int medium = BenchIterationsFor(17.2e9, peak);
  Check(medium >= kBenchMinIters && medium <= kBenchMaxIters,
        "the count stays inside its floor and ceiling");
  Check(medium <= BenchIterationsFor(2.1e9, peak),
        "a larger row takes no more iterations than a smaller one");
  // The count is the round divided by the ideal time, at a rate where the
  // answer falls between the bounds: 1e-4 s a call at 1 TFLOP/s.
  Check(BenchIterationsFor(1e8, 1e12) == 500,
        "the count is the round divided by the row's ideal time");
  // No denominator states the floor.
  Check(BenchIterationsFor(2.1e9, 0.0) == kBenchMinIters,
        "no declared peak takes the floor");
  Check(BenchIterationsFor(0.0, peak) == kBenchMinIters,
        "a row with no work takes the floor");
}

void CheckPredicate() {
  // An exact match passes and a one percent error does not.
  Check(CandidateMatches(0.0, 4.0), "an exact result passes the check");
  Check(CandidateMatches(1e-8, 4.0),
        "a rounding difference of the last place passes the check");
  Check(!CandidateMatches(0.04, 4.0), "a one percent error fails the check");
  Check(!CandidateMatches(4.0 * 0.04, 4.0),
        "a two percent error fails the check");
  // A deliberately wrong candidate: the whole result scaled by 25 percent.
  const double want = 16.0;
  const double diff = want * 0.25 * 0.25;
  Check(!CandidateMatches(diff, want),
        "a result scaled by a quarter is rejected");
  Check(std::fabs(CandidateRelativeError(diff, want) - 0.25) < 1e-12,
        "the reported error of the rejected candidate is the scale it moved");
  // A reference of zero norm admits only an exact result.
  Check(CandidateMatches(0.0, 0.0), "an exact zero passes a zero reference");
  Check(!CandidateMatches(1e-9, 0.0),
        "a non-zero difference fails a zero "
        "reference");
  // A reduction that did not run states a negative sum, which must not pass.
  Check(!CandidateMatches(-1.0, 4.0),
        "a reduction that did not run is refused");
}

}  // namespace

int main() {
  CheckList();
  CheckWork();
  CheckDerived();
  CheckBudget();
  CheckIterations();
  CheckPredicate();
  if (g_failures != 0) {
    std::fprintf(stderr, "bench shapes test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("bench shapes test: %d rows, %d families, all derived\n",
              BenchShapeCount(), 6);
  return 0;
}
