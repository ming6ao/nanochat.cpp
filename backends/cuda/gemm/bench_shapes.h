#ifndef NANOCHAT_BACKENDS_CUDA_GEMM_BENCH_SHAPES_H_
#define NANOCHAT_BACKENDS_CUDA_GEMM_BENCH_SHAPES_H_

// The measurement rows of the GEMM candidate sweep, and the arithmetic every
// one of them derives (docs/kernels.md section 11.5).
//
// Two row families share the one benchmark. The **production rows** are the 28
// shapes of the closed list in `dispatch.cc`: they are the shapes the one model
// issues, and only they may select a candidate for a production shape. These
// **measurement rows** say how a kernel scales. They answer a question about a
// kernel and not about a model, so a row here never selects a candidate; it
// decides whether a candidate is fit to enter a sweep.
//
// Every row states extents, a batch, two transpose flags, and a store type, and
// nothing else. The flop count, the operand bytes, the output elements, and the
// tile count are all derived here, so a mistyped row fails this file's host
// test and not a card run.
//
// Precision free and device free: a host test drives every function below.

#include <cstddef>
#include <cstdint>
#include <string>

#include "backends/cuda/gemm/gemm_shape.h"
#include "nanochat/kernels.h"

namespace nanochat {
namespace kernels {

// One measurement row.
struct BenchShape {
  // The report name, in the `family:variant` spelling of section 11.5.
  const char* name;
  int m = 0;
  int n = 0;
  int k = 0;
  int batch_count = 1;
  bool transpose_a = false;
  bool transpose_b = false;
  // True when C is stored as fp32, which is the weight-gradient form.
  bool fp32_c = false;
  // Extra elements in A's leading dimension. A zero states the tight leading
  // dimension; one states the `lda = k + 1` row, whose stride leaves the
  // sixteen-byte boundary.
  int lda_extra = 0;
  // The offset, in elements, of the operand pointers from their allocation.
  // One states the row whose pointer leaves the sixteen-byte boundary.
  int pointer_offset = 0;
  // What the row measures, in the words of section 11.5.
  const char* measures;
};

// The rows of section 11.5, in the order that section lists them.
int BenchShapeCount();
const BenchShape& BenchShapeAt(int index);

// The full resolved call of a row, by the one resolution every other part of
// the tree uses.
ResolvedGemm BenchShapeGemm(int index);

// One row's extent as `M x N x K`, with the batch and the store spelled out.
std::string BenchShapeExtent(const BenchShape& shape);

// The necessary operation count: two floating-point operations for each element
// of the product, over every batch slice.
double BenchShapeFlops(const BenchShape& shape);

// The width of one operand element. The sweep is a bf16 measurement, because
// `gemm_benchmark` compiles only in a bf16 or fp8 build and the seam's operand
// is two bytes there. The arithmetic states the width instead of
// `sizeof(ComputeType)`, so this header, its host test, and the 8 GiB figure of
// section 11.5 read the same numbers in the default fp32 build.
inline constexpr std::size_t kBenchOperandElementBytes = 2;

// The bytes of A and of B, and the bytes of the two together.
std::size_t BenchShapeOperandABytes(const BenchShape& shape);
std::size_t BenchShapeOperandBBytes(const BenchShape& shape);
std::size_t BenchShapeOperandBytes(const BenchShape& shape);
std::size_t BenchShapeOutputBytes(const BenchShape& shape);

// The count of the elements of C, which is also the count the check reduces.
std::size_t BenchShapeOutputElements(const BenchShape& shape);

// The bytes the row needs on the device: both operands, the result, and the
// reference copy the check compares against. The benchmark sizes one workspace
// from the largest row it will run, so this is the number that decides it.
std::size_t BenchShapeBytes(const BenchShape& shape);

// The tile count at the 128 by 256 tile of `dev/cutlass/README.md` section 3,
// counted before the cluster multicast halves the operand traffic.
long long BenchShapeTileCount(const BenchShape& shape);

// The count of the elements of the fill: both operands and the result.
std::size_t BenchShapeFillElements(const BenchShape& shape);

// --- The device budget ------------------------------------------------------

// The largest row of the list needs 8 GiB of a card: two 2 GiB operands, a
// 4 GiB fp32 result, and the reference copy the check compares against. The
// harness takes the card's free memory and keeps a reserve, and it reports a
// row that exceeds the budget as skipped with the byte count and the reason.
// The reserve is a fraction of the free memory, because the driver and the
// cuBLAS workspaces need a share that a fixed number cannot state.
inline constexpr double kBenchDeviceReserve = 0.20;

// The bytes a row may use, from the free bytes a card reports.
std::size_t BenchDeviceBudget(std::size_t free_bytes);

// The reason a row needs more than the budget, or an empty string when it fits.
// The byte form serves a row of either family, because a production row states
// its bytes without a `BenchShape`.
std::string BenchSizeSkipReason(std::size_t bytes, std::size_t budget_bytes);

// The same reason for a row of this list.
std::string BenchShapeSkipReason(const BenchShape& shape,
                                 std::size_t budget_bytes);

// --- The iteration count ----------------------------------------------------

// A round must run at least 50 ms, so a row's count follows from its ideal time
// at the declared peak. The floor keeps a row that the counters cannot resolve
// measurable, and the ceiling keeps a 70 TFLOP row from taking a minute.
inline constexpr double kBenchRoundSeconds = 0.050;
inline constexpr int kBenchMinIters = 3;
inline constexpr int kBenchMaxIters = 1000;

// The iteration count for one round of `flops` at `peak_flops`. A zero or
// negative peak states no denominator, and then the floor is the answer.
int BenchIterationsFor(double flops, double peak_flops);

// --- The check --------------------------------------------------------------

// Two bf16 accumulators differ in the order of the sum, so a candidate's result
// cannot equal the reference bit for bit. The check therefore compares the
// squared norms: the squared Frobenius norm of the difference over the squared
// norm of the reference. A bf16 store rounds once at the end, which is 4e-3 of
// the reference for the worst case, so 1e-2 leaves room for the accumulation
// order and rejects any error a reader would call wrong.
inline constexpr double kCandidateMaxRelativeError = 1e-2;

// True when a candidate's result matches the reference within
// `kCandidateMaxRelativeError`. A reference of zero norm matches only an exact
// zero difference.
bool CandidateMatches(double diff_sumsq, double ref_sumsq);

// The relative error the check reads, for the report. A reference of zero norm
// states no ratio, which is infinity for a non-zero difference and zero for an
// exact one.
double CandidateRelativeError(double diff_sumsq, double ref_sumsq);

}  // namespace kernels
}  // namespace nanochat

#endif  // NANOCHAT_BACKENDS_CUDA_GEMM_BENCH_SHAPES_H_
