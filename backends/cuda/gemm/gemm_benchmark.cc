// The sweep of the GEMM arms (docs/kernels.md Part 3, sections 11.4 to 11.7).
//
// One binary measures three arms over two row families:
//
//   * cuBLAS, the reference and the last fallback.
//   * The Hopper mainloop of `cutlass_bf16.cu`, one candidate at a time from
//   the
//     registry of `cutlass_candidates.h`.
//   * The hand tile of `gemm_bf16.cu`, one schedule at a time from
//     `gemm_plan.h`.
//
// Every arm runs through the seam, so a measurement covers the wiring a
// training step takes and not only the kernel. `NANOCHAT_GEMM_LIB` names the
// arm, as it does for the shipped path.
//
// **The protocol** is the one of section 11.4, and each step answers a defect
// of the measurement before it:
//
//   1. The operands are filled on the device, so a two GiB row costs no host
//      copy and no host memory.
//   2. One workspace serves every row, sized from the largest row that runs.
//   3. A row that needs more than the card's budget is reported as skipped,
//   with
//      its byte count and the reason.
//   4. The second-level cache is flushed before every timed call, so no row
//      reads what the previous iteration left warm. The classifier operands are
//      63 MB and the cache of an H100 is 50 MB.
//   5. Every arm reports the minimum and the median of the rounds.
//   6. Every candidate is checked against cuBLAS before its timing is reported.
//      A candidate that is fast and wrong must not win.
//
// **The rows.** `--rows production` measures the closed list of `dispatch.cc`,
// which is the shapes the one model issues: only these may select a candidate
// for a production shape. `--rows measurement` measures the families of section
// 11.5, which say how a kernel scales. `--rows all` runs both.
//
// **The report.** One row for each measurement, named `gemm:<row>:<arm>`, and a
// `:median` companion row. The primary row states the minimum, which is the
// meaning the hand tile's rate gate already gave that column, so a row compares
// with the committed entry of `docs/baselines.json`. The companion row states
// the median, which is the statistic a decision reads. A measured arm that the
// check rejects states no row at all, and the reason is on standard error.
//
// **The check.** A small row compares every element on the host. A big row
// reduces the squared norm of the difference and of the reference on the
// device, because a full copy of a four GiB result fits neither the card's
// budget nor the host's. Both paths answer the same predicate,
// `CandidateMatches`.
//
// **`--self-test`** proves the check itself: an identical copy must match and a
// perturbed copy must be rejected. That is the gate which states a deliberately
// wrong candidate is rejected by name, and it needs no wrong kernel.
//
// A run needs a bf16 build, because the operands are bf16. The build compiles
// this source only then; an fp32 build runs the stand-in `main`.
//
// See docs/performance.md for the measurement protocol.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "backends/cuda/gemm/bench_shapes.h"
#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/cutlass_launch.h"
#include "backends/cuda/gemm/dispatch.h"
#include "backends/cuda/gemm/gemm_fp32_plan.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "backends/cuda/kernels/testing/bench_protocol.h"
#include "backends/cuda/kernels/testing/bench_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::BenchOptions;
using nanochat::BenchReport;
using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::Fp8Format;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::GradientType;
using nanochat::QuantizeFp8Params;

using nanochat::ScaledGemmParams;
using nanochat::kernels::Alloc;
using nanochat::kernels::BenchDeviceBudget;
using nanochat::kernels::BenchIterationsFor;
using nanochat::kernels::BenchShapeAt;
using nanochat::kernels::BenchShapeCount;
using nanochat::kernels::BenchShapeExtent;
using nanochat::kernels::BenchShapeFlops;
using nanochat::kernels::BenchShapeGemm;
using nanochat::kernels::BenchShapeOperandABytes;
using nanochat::kernels::BenchShapeOperandBBytes;
using nanochat::kernels::BenchSizeSkipReason;
using nanochat::kernels::CandidateMatches;
using nanochat::kernels::CandidateRelativeError;
using nanochat::kernels::CutlassGrid;
using nanochat::kernels::CutlassRasterOrder;
using nanochat::kernels::CutlassRasterOrderFromName;
using nanochat::kernels::CutlassRasterOrderName;
using nanochat::kernels::CutlassRefusal;
using nanochat::kernels::CutlassRefusalName;
using nanochat::kernels::Fp8ScaledFormSupported;
using nanochat::kernels::Free;
using nanochat::kernels::Gemm;
using nanochat::kernels::GemmCandidateAt;
using nanochat::kernels::GemmCandidateCount;
using nanochat::kernels::GemmCandidateIndexFromName;
using nanochat::kernels::GemmCandidateSpec;
using nanochat::kernels::GemmCutlassCandidateGrid;
using nanochat::kernels::GemmCutlassCandidateOccupancy;
using nanochat::kernels::GemmCutlassRefusalCount;
using nanochat::kernels::GemmScaledFp8;
using nanochat::kernels::GemmScheduleCount;
using nanochat::kernels::GemmScheduleName;
using nanochat::kernels::GemmTileShapeCount;
using nanochat::kernels::GemmTileShapeGemm;
using nanochat::kernels::GemmTileShapeName;
using nanochat::kernels::GemmTileShapeSchedule;
using nanochat::kernels::JudgeCutlass;
using nanochat::kernels::JudgeScaledFp8;
using nanochat::kernels::Memcpy;
using nanochat::kernels::Memset;
using nanochat::kernels::QuantizeFp8;
using nanochat::kernels::ResetGemmCutlassRefusalCount;
using nanochat::kernels::ResolvedGemm;
using nanochat::kernels::Synchronize;

using nanochat::BenchRow;
using nanochat::dev::AddTiming;
using nanochat::dev::BenchDeviceName;
using nanochat::dev::Flusher;
using nanochat::dev::Gflops;
using nanochat::dev::Summarize;
using nanochat::dev::TimeFlush;
using nanochat::dev::TimeWithFlush;
using nanochat::dev::Timing;
using nanochat::dev::UseDevicePeak;
using nanochat::dev::WindowTimer;

// The sweep is a bf16 measurement on a bf16 or fp8 build: the operands are the
// seam's two-byte bf16, and the row arithmetic of `bench_shapes.h` states that
// width. An fp32 build measures the fp32 tile of `docs/kernels.md` Part 3
// section 6 phase 3 instead, whose operands are four bytes, so the width is the
// one thing the two builds state differently.
static_assert(sizeof(ComputeType) == 2 || sizeof(ComputeType) == 4,
              "the GEMM sweep measures a two-byte or a four-byte operand");
static_assert(sizeof(GradientType) == 4, "the fp32 store is four bytes");

// The bytes below which the check copies both results to the host and reads
// every element. Above it the check reduces on the device.
inline constexpr std::size_t kHostCheckBytes = 32u * 1024 * 1024;

// The blocks and threads of the fill and the reduction. The fill is
// bandwidth-bound and the reduction is a short grid-stride loop, so neither
// needs a shape-dependent grid.
inline constexpr int kFillBlocks = 512;
inline constexpr int kFillThreads = 256;
inline constexpr int kReduceBlocks = 256;
inline constexpr int kReduceThreads = 256;

// The sixteen-byte boundary every buffer of the workspace shares.
inline constexpr std::size_t kWorkspaceAlign = 16;

struct Options {
  BenchOptions bench;
  // Empty runs every row of the selected families.
  std::string row;
  // "production", "measurement", or "all".
  std::string rows = "production";
  // "all", "cublas", "cutlass", or "hand".
  std::string arms = "all";
  // Empty runs every candidate of the registry; a name runs one.
  std::string candidate;
  // Empty measures the schedule the row owns; "all" sweeps the list.
  std::string schedule;
  // Takes the iteration count of a measurement row from its work.
  bool auto_iters = false;
  // Flushes the second-level cache before every timed call.
  bool flush = true;
  // Checks every measured arm against the reference.
  bool check = true;
  // Times the fp8 arms beside cublasLt on quantized operands (docs/kernels.md
  // section 10.4). It needs an fp8 build and a Hopper device.
  bool fp8 = false;
  // The two lists of the scheduler sweep, comma separated and
  // empty for the vendor default. One entry of each makes one cell.
  std::string swizzle;
  std::string raster;
  // Proves the check itself and exits.
  bool self_test = false;
};

// One cell of the scheduler sweep: the two
// launch arguments of the CUTLASS persistent scheduler.
struct SchedulerCell {
  int max_swizzle_size = 1;
  nanochat::kernels::CutlassRasterOrder raster =
      nanochat::kernels::CutlassRasterOrder::kHeuristic;

  // The cell as a row suffix spells it. A single-cell run states the empty
  // string, so its report names are the ones the table writer reads.
  std::string Suffix() const {
    if (default_cell) return std::string();
    return "@sw" + std::to_string(max_swizzle_size) + "-" +
           nanochat::kernels::CutlassRasterOrderName(raster);
  }

  // True for the pair the shipped path states, which needs no suffix.
  bool default_cell = true;
};

// One row of either family, in the one form the harness needs.
struct SweepRow {
  std::string name;
  std::string shape;
  ResolvedGemm gemm;
  double flops = 0.0;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  // The bytes of one result buffer. Two of them exist: the result and the
  // reference the check reads.
  std::size_t out_bytes = 0;
  // The elements by which the operand pointers leave the sixteen-byte boundary.
  int pointer_offset = 0;
  // The hand tile's schedule for this shape, which is the one a promotion would
  // run. A measurement row states none, so it takes the first of the sweep.
  int own_schedule = 0;
  // True for a row of the closed list, which is the only family that may select
  // a candidate for a production shape.
  bool production = false;
};

std::size_t OutputElementBytes(bool fp32_c) {
  return fp32_c ? 4 : sizeof(ComputeType);
}

//: The row arithmetic of `bench_shapes.h` states a two-byte operand, because
//: the bf16 sweep is its reader. An fp32 build doubles both operands, and the
//: workspace and the device budget have to know it.
inline constexpr std::size_t kOperandScale =
    sizeof(ComputeType) / nanochat::kernels::kBenchOperandElementBytes;

// The candidate arm exists only in a bf16 or fp8 build, because the CUTLASS
// mainloop's operands are bf16. An fp32 build measures the fp32 tile instead,
// so the occupancy a candidate would state is zero there and the function is a
// template: a discarded `if constexpr` branch of a non-template still names its
// symbol, and that symbol is not defined in this build.
template <typename Element = ComputeType>
int CandidateOccupancy(int candidate) {
  if constexpr (sizeof(Element) == 2) {
    return GemmCutlassCandidateOccupancy(candidate);
  }
  return 0;
}

// The grid a candidate launches, and the block count of the call. Both come
// from the vendor's scheduler, so the sweep reads how the card is used instead
// of inferring it from a timing. It follows the same build rule as the
// occupancy above: only a bf16 or fp8 build defines the query.
template <typename Element = ComputeType>
CutlassGrid CandidateGrid(int candidate, const ResolvedGemm& gemm) {
  if constexpr (sizeof(Element) == 2) {
    return GemmCutlassCandidateGrid(candidate, gemm);
  }
  return CutlassGrid{};
}

// The refusal count of a timed window, on the same build rule.
template <typename Element = ComputeType>
int Refusals() {
  if constexpr (sizeof(Element) == 2) return GemmCutlassRefusalCount();
  return 0;
}

template <typename Element = ComputeType>
void ResetRefusals() {
  if constexpr (sizeof(Element) == 2) ResetGemmCutlassRefusalCount();
}

// --- The scheduler sweep -----------------------------------------------------

// One comma-separated list of a run, split and stripped. An empty string is no
// list, which leaves that dimension at the vendor default.
std::vector<std::string> SplitList(const std::string& text) {
  std::vector<std::string> items;
  std::string current;
  for (const char character : text) {
    if (character == ',') {
      if (!current.empty()) items.push_back(current);
      current.clear();
      continue;
    }
    if (character != ' ') current.push_back(character);
  }
  if (!current.empty()) items.push_back(current);
  return items;
}

// The cells one run measures. An empty pair of lists is the one shipped cell,
// and it carries no suffix, so a run that names no list writes the report names
// the decision table is written from.
std::vector<SchedulerCell> SchedulerCells(const Options& options) {
  const std::vector<std::string> swizzles = SplitList(options.swizzle);
  const std::vector<std::string> rasters = SplitList(options.raster);
  std::vector<SchedulerCell> cells;
  if (swizzles.empty() && rasters.empty()) {
    cells.push_back(SchedulerCell{});
    return cells;
  }
  const std::vector<std::string> sizes =
      swizzles.empty() ? std::vector<std::string>{"1"} : swizzles;
  const std::vector<std::string> orders =
      rasters.empty() ? std::vector<std::string>{"heuristic"} : rasters;
  for (const std::string& size : sizes) {
    for (const std::string& order : orders) {
      SchedulerCell cell;
      cell.max_swizzle_size = std::atoi(size.c_str());
      if (cell.max_swizzle_size < 1) cell.max_swizzle_size = 1;
      cell.raster = CutlassRasterOrderFromName(order.c_str());
      cell.default_cell = cell.max_swizzle_size == 1 &&
                          cell.raster == CutlassRasterOrder::kHeuristic;
      cells.push_back(cell);
    }
  }
  return cells;
}

// States one cell to the launch, which reads the two variables on each call.
void ApplyCell(const SchedulerCell& cell) {
  const std::string size = std::to_string(cell.max_swizzle_size);
  setenv("NANOCHAT_GEMM_SWIZZLE", size.c_str(), 1);
  setenv("NANOCHAT_GEMM_RASTER", CutlassRasterOrderName(cell.raster), 1);
}

// The wave count of a grid: the blocks the call holds over the blocks it
// launches, rounded up.
int WavesOf(const CutlassGrid& grid) {
  if (grid.grid_blocks <= 0) return 0;
  return (grid.problem_blocks + grid.grid_blocks - 1) / grid.grid_blocks;
}

// The grid text a sweep appends to a row's shape, because a card job returns
// the report and only the tail of the console. A single-cell run states
// nothing, so a decision report keeps the shape it always had.
std::string GridText(const CutlassGrid& grid) {
  if (grid.grid_blocks <= 0) return std::string();
  std::string text = " grid " + std::to_string(grid.grid_blocks) + "/" +
                     std::to_string(grid.problem_blocks);
  const int waves = WavesOf(grid);
  if (waves > 1) text += " waves " + std::to_string(waves);
  return text;
}

// The count of the elements of C, at the leading dimension of the call.
std::size_t OutputElements(const ResolvedGemm& gemm) {
  return static_cast<std::size_t>(gemm.batch_count) *
         static_cast<std::size_t>(gemm.m) * static_cast<std::size_t>(gemm.ldc);
}

std::size_t Align(std::size_t bytes) {
  return (bytes + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

// The workspace one row needs: both operands, the result, and the reference,
// each on the boundary, plus the operand offset the row asks for.
std::size_t RowWorkspaceBytes(const SweepRow& row) {
  const std::size_t offset =
      static_cast<std::size_t>(row.pointer_offset) * sizeof(ComputeType);
  return Align(offset + row.a_bytes) + Align(row.b_bytes) +
         2 * Align(row.out_bytes);
}

std::string ExtentText(const ResolvedGemm& gemm) {
  std::string text = std::to_string(gemm.m) + "x" + std::to_string(gemm.n) +
                     "x" + std::to_string(gemm.k);
  if (gemm.batch_count != 1) {
    text += " batch " + std::to_string(gemm.batch_count);
  }
  if (gemm.transpose_a || gemm.transpose_b) {
    text += gemm.transpose_a ? (gemm.transpose_b ? " tt" : " tn") : " nt";
  }
  if (gemm.fp32_c) text += " fp32 out";
  return text;
}

SweepRow MakeProductionRow(int index) {
  SweepRow row;
  row.production = true;
  row.name = GemmTileShapeName(index);
  row.gemm = GemmTileShapeGemm(index);
  const ResolvedGemm& gemm = row.gemm;
  const std::size_t rows_a =
      static_cast<std::size_t>(gemm.transpose_a ? gemm.k : gemm.m);
  const std::size_t rows_b =
      static_cast<std::size_t>(gemm.transpose_b ? gemm.n : gemm.k);
  row.a_bytes = static_cast<std::size_t>(gemm.batch_count) * rows_a *
                static_cast<std::size_t>(gemm.lda) * sizeof(ComputeType);
  row.b_bytes = static_cast<std::size_t>(gemm.batch_count) * rows_b *
                static_cast<std::size_t>(gemm.ldb) * sizeof(ComputeType);
  row.out_bytes = OutputElements(gemm) * OutputElementBytes(gemm.fp32_c);
  row.flops =
      2.0 * static_cast<double>(gemm.m) * gemm.n * gemm.k * gemm.batch_count;
  row.shape = ExtentText(gemm);
  row.own_schedule = GemmTileShapeSchedule(index);
  return row;
}

SweepRow MakeMeasurementRow(int index) {
  SweepRow row;
  const nanochat::kernels::BenchShape& shape = BenchShapeAt(index);
  row.name = shape.name;
  row.gemm = BenchShapeGemm(index);
  row.flops = BenchShapeFlops(shape);
  row.a_bytes = BenchShapeOperandABytes(shape) * kOperandScale;
  row.b_bytes = BenchShapeOperandBBytes(shape) * kOperandScale;
  row.out_bytes = nanochat::kernels::BenchShapeOutputBytes(shape);
  row.pointer_offset = shape.pointer_offset;
  row.shape = BenchShapeExtent(shape);
  return row;
}

// --- The device fill --------------------------------------------------------
//
// The operands are filled on the device, so a row of two GiB costs no host
// buffer. The pattern is deterministic and spread over both signs, because a
// uniform operand makes every product equal and a wrong kernel can pass.

template <class Element>
__global__ void FillKernel(Element* dst, long long count, unsigned seed) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned bits = static_cast<unsigned>(i) * 2654435761u + seed * 2246822519u;
    bits ^= bits >> 15;
    // A value in [-0.5, 0.5), which keeps a deep sum finite in bf16.
    const float unit = static_cast<float>(bits & 0xFFFFu) / 65536.0f - 0.5f;
    dst[i] = Element(unit);
  }
}

// A deliberately wrong result, for the self test: every element is scaled, so
// the error is the scale and not one element out of a million. The check states
// a relative Frobenius error, and a single element of a large row would hide
// under that bound.
template <class Element>
__global__ void ScaleKernel(Element* dst, long long count, float factor) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    dst[i] = Element(static_cast<float>(dst[i]) * factor);
  }
}

// The squared norm of the difference and of the reference, in one pass. One
// block writes two partial sums, and the host adds them, so no atomic operation
// bounds the grid.
template <class Element>
__global__ void DiffSumsqKernel(const Element* got, const Element* want,
                                long long count, double* partial) {
  __shared__ double s_diff[kReduceThreads];
  __shared__ double s_norm[kReduceThreads];
  double diff = 0.0;
  double norm = 0.0;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const double value = static_cast<double>(static_cast<float>(got[i]));
    const double expected = static_cast<double>(static_cast<float>(want[i]));
    const double delta = value - expected;
    diff += delta * delta;
    norm += expected * expected;
  }
  s_diff[threadIdx.x] = diff;
  s_norm[threadIdx.x] = norm;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_diff[threadIdx.x] += s_diff[threadIdx.x + offset];
      s_norm[threadIdx.x] += s_norm[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = s_diff[0];
    partial[gridDim.x + blockIdx.x] = s_norm[0];
  }
}

void CheckLaunch(const char* what) {
  const cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "gemm benchmark: %s: %s\n", what,
               cudaGetErrorString(status));
  std::exit(3);
}

template <class Element>
void FillDevice(void* buffer, std::size_t count, unsigned seed) {
  if (buffer == nullptr || count == 0) return;
  const std::size_t needed = (count + kFillThreads - 1) / kFillThreads;
  const int blocks =
      static_cast<int>(std::min<std::size_t>(needed, kFillBlocks));
  FillKernel<Element><<<blocks, kFillThreads>>>(
      static_cast<Element*>(buffer), static_cast<long long>(count), seed);
  CheckLaunch("device fill");
}

// --- The workspace and the flush --------------------------------------------

// One allocation for every row: the two operands, the result, and the
// reference. It grows to the largest row and stays, so a sweep pays one
// allocation.
class Workspace {
 public:
  ~Workspace() { Release(); }
  Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  bool Reserve(std::size_t bytes) {
    if (bytes <= bytes_) return true;
    if (base_ != nullptr) Free(base_);
    base_ = Alloc(bytes);
    bytes_ = base_ != nullptr ? bytes : 0;
    return base_ != nullptr;
  }

  char* base() const { return static_cast<char*>(base_); }

  void Release() {
    Free(base_);
    base_ = nullptr;
    bytes_ = 0;
  }

 private:
  void* base_ = nullptr;
  std::size_t bytes_ = 0;
};

// One buffer per row, carved from the workspace.
struct RowBuffers {
  ComputeType* a = nullptr;
  ComputeType* b = nullptr;
  void* c = nullptr;
  void* ref = nullptr;
};

RowBuffers Carve(char* base, const SweepRow& row) {
  RowBuffers buffers;
  char* cursor = base;
  const std::size_t offset =
      static_cast<std::size_t>(row.pointer_offset) * sizeof(ComputeType);
  buffers.a = reinterpret_cast<ComputeType*>(cursor + offset);
  cursor += static_cast<std::ptrdiff_t>(Align(offset + row.a_bytes));
  buffers.b = reinterpret_cast<ComputeType*>(cursor);
  cursor += static_cast<std::ptrdiff_t>(Align(row.b_bytes));
  buffers.c = cursor;
  cursor += static_cast<std::ptrdiff_t>(Align(row.out_bytes));
  buffers.ref = cursor;
  return buffers;
}

// --- The check --------------------------------------------------------------

struct CheckOutcome {
  bool ran = false;
  bool matched = false;
  double relative_error = 0.0;
  // The worst element on the host path, or npos on the device path.
  std::size_t worst_index = 0;
  bool element_wise = false;
};

inline constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

// Every element, on the host. The row is small enough that both results fit the
// sandbox's host budget.
template <class Element>
CheckOutcome CompareOnHost(const void* got, const void* want,
                           std::size_t count) {
  CheckOutcome outcome;
  outcome.ran = true;
  outcome.element_wise = true;
  std::vector<Element> host_got(count);
  std::vector<Element> host_want(count);
  Memcpy(host_got.data(), got, count * sizeof(Element), CopyDir::kDeviceToHost);
  Memcpy(host_want.data(), want, count * sizeof(Element),
         CopyDir::kDeviceToHost);
  double diff_sumsq = 0.0;
  double ref_sumsq = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = static_cast<double>(static_cast<float>(host_got[i]));
    const double expected =
        static_cast<double>(static_cast<float>(host_want[i]));
    const double delta = value - expected;
    diff_sumsq += delta * delta;
    ref_sumsq += expected * expected;
    const double scaled = std::fabs(delta) / (1.0 + std::fabs(expected));
    if (scaled > worst) {
      worst = scaled;
      outcome.worst_index = i;
    }
  }
  outcome.matched = CandidateMatches(diff_sumsq, ref_sumsq);
  outcome.relative_error = CandidateRelativeError(diff_sumsq, ref_sumsq);
  return outcome;
}

// The squared norms, on the device. A four GiB result needs this path.
template <class Element>
CheckOutcome CompareOnDevice(const void* got, const void* want,
                             std::size_t count) {
  CheckOutcome outcome;
  outcome.ran = true;
  outcome.worst_index = kNoIndex;
  if (count == 0) return outcome;
  const std::size_t partials = 2 * kReduceBlocks;
  double* device_partial =
      static_cast<double*>(Alloc(partials * sizeof(double)));
  if (device_partial == nullptr) return outcome;
  DiffSumsqKernel<Element><<<kReduceBlocks, kReduceThreads>>>(
      static_cast<const Element*>(got), static_cast<const Element*>(want),
      static_cast<long long>(count), device_partial);
  CheckLaunch("device reduction");
  std::vector<double> host_partial(partials, 0.0);
  Memcpy(host_partial.data(), device_partial, partials * sizeof(double),
         CopyDir::kDeviceToHost);
  Free(device_partial);
  double diff_sumsq = 0.0;
  double ref_sumsq = 0.0;
  for (int block = 0; block < kReduceBlocks; ++block) {
    diff_sumsq += host_partial[block];
    ref_sumsq += host_partial[kReduceBlocks + block];
  }
  outcome.matched = CandidateMatches(diff_sumsq, ref_sumsq);
  outcome.relative_error = CandidateRelativeError(diff_sumsq, ref_sumsq);
  return outcome;
}

CheckOutcome CheckOutput(const SweepRow& row, const void* got, const void* want,
                         std::size_t count, bool have_reference) {
  CheckOutcome outcome;
  if (!have_reference || count == 0) return outcome;
  const std::size_t bytes = count * OutputElementBytes(row.gemm.fp32_c);
  const bool on_host = bytes <= kHostCheckBytes;
  if (row.gemm.fp32_c) {
    return on_host ? CompareOnHost<float>(got, want, count)
                   : CompareOnDevice<float>(got, want, count);
  }
  return on_host ? CompareOnHost<ComputeType>(got, want, count)
                 : CompareOnDevice<ComputeType>(got, want, count);
}

// One report row of the sweep, in full. `AddTiming` takes the row in full and
// states the two rows of one measurement from it, so the sweep names its family
// and its phase here as well as on the report. `BenchReport::Add` inherits the
// precision alone, and not the family.
BenchRow TimedRow(const std::string& name, const std::string& shape,
                  int occupancy = 0) {
  BenchRow row;
  row.name = name;
  row.shape = shape;
  row.family = "gemm";
  row.phase = "forward";
  row.occupancy = occupancy;
  return row;
}

void PrintMeasurement(const std::string& row, const std::string& arm,
                      const Timing& timing, int occupancy,
                      const CheckOutcome& check,
                      const CutlassGrid& grid = CutlassGrid()) {
  std::fprintf(stderr,
               "gemm benchmark: %-24s %-10s min %9.4f ms  median %9.4f ms  "
               "iters %4d",
               row.c_str(), arm.c_str(), timing.min_ms, timing.median_ms,
               timing.iters);
  if (occupancy > 0) std::fprintf(stderr, "  occ %d", occupancy);
  if (grid.grid_blocks > 0) {
    // The two numbers the vendor's scheduler answers: the blocks it launches
    // and the blocks the call holds. A call with more work than launch runs
    // more than one wave, and a launch below the multiprocessor count leaves
    // the card idle.
    std::fprintf(stderr, "  grid %d/%d", grid.grid_blocks, grid.problem_blocks);
    const int waves = WavesOf(grid);
    if (waves > 1) std::fprintf(stderr, "  waves %d", waves);
  }
  if (check.ran) {
    std::fprintf(stderr, "  rel err %.2e %s", check.relative_error,
                 check.matched ? "yes" : "NO");
  }
  std::fprintf(stderr, "\n");
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--rows production|measurement|all] [--row NAME] "
               "[--all|cublas|cutlass|hand|seam] [--arms LIST] "
               "[--candidate NAME|all] "
               "[--schedule NAME|all] [--swizzle LIST] [--raster LIST] "
               "[--auto-iters] [--no-flush] [--no-check] [--self-test] "
               "[--fp8] [--iters N] [--rounds N] [--warmup N] [--json] "
               "[--out PATH]\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  nanochat::RequireSandboxOrDie("bench");

  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_int = [&](int* out) {
      if (i + 1 >= argc) {
        Usage(argv[0]);
        std::exit(2);
      }
      *out = std::atoi(argv[++i]);
    };
    if (arg == "--row" && i + 1 < argc) {
      options.row = argv[++i];
    } else if (arg == "--rows" && i + 1 < argc) {
      options.rows = argv[++i];
    } else if (arg == "--arms" && i + 1 < argc) {
      // A list, so a sweep can measure cuBLAS beside the candidates without the
      // hand tile's five schedules. The single-arm flags below still state one.
      options.arms = argv[++i];
    } else if (arg == "--hand") {
      options.arms = "hand";
    } else if (arg == "--cublas") {
      options.arms = "cublas";
    } else if (arg == "--cutlass") {
      options.arms = "cutlass";
    } else if (arg == "--seam") {
      // The shipped path: no override, so the seam reads the decision table.
      options.arms = "seam";
    } else if (arg == "--all" || arg == "--both") {
      // `--both` is the older spelling of the two-side measurement, and it
      // still means every arm.
      options.arms = "all";
    } else if (arg == "--candidate" && i + 1 < argc) {
      options.candidate = argv[++i];
    } else if (arg == "--schedule" && i + 1 < argc) {
      options.schedule = argv[++i];
    } else if (arg == "--swizzle" && i + 1 < argc) {
      options.swizzle = argv[++i];
    } else if (arg == "--raster" && i + 1 < argc) {
      options.raster = argv[++i];
    } else if (arg == "--auto-iters") {
      options.auto_iters = true;
    } else if (arg == "--no-flush") {
      options.flush = false;
    } else if (arg == "--no-check") {
      options.check = false;
    } else if (arg == "--self-test") {
      options.self_test = true;
    } else if (arg == "--fp8") {
      options.fp8 = true;
    } else if (arg == "--iters") {
      next_int(&options.bench.iters);
    } else if (arg == "--rounds") {
      next_int(&options.bench.rounds);
    } else if (arg == "--warmup") {
      next_int(&options.bench.warmup);
    } else if (arg == "--json") {
      options.bench.json = true;
    } else if (arg == "--out" && i + 1 < argc) {
      options.bench.out = argv[++i];
    } else {
      Usage(argv[0]);
      return 2;
    }
  }

  // The arms of the run, by membership, so `--arms cublas,cutlass` measures a
  // ratio without the hand tile and `--all` measures every arm.
  const std::vector<std::string> arm_names = SplitList(options.arms);
  const auto names_arm = [&arm_names](const char* name) {
    return std::find(arm_names.begin(), arm_names.end(), name) !=
           arm_names.end();
  };
  const bool run_all = names_arm("all");
  const bool run_cublas = run_all || names_arm("cublas");
  const bool run_cutlass = run_all || names_arm("cutlass");
  const bool run_hand = run_all || names_arm("hand");
  // The seam arm is separate from `all`, because it measures the shipped path
  // and not a candidate: a report of the three measured arms is what the table
  // writer reads, and a seam report must not be mistaken for one.
  // The seam arm is the shipped path, so it is measured alone by default. It
  // combines with cuBLAS on request, because a fraction needs a denominator
  // from the same session: the two classifier rows moved 7 and 12 percent
  // between two runs of one binary, and a fraction across sessions cannot
  // separate that drift from a difference the change made.
  const bool run_seam = names_arm("seam");

  const nanochat::Caps caps = nanochat::kernels::GetCaps();
  if (!caps.is_device) {
    std::fprintf(stderr, "gemm benchmark: no device\n");
    return 1;
  }
  // The candidate arm needs a compute capability 9.0 part, because the cubin is
  // `sm_90a`. A measurement on another card states that instead of timing the
  // fallback under the candidate's name: the seam refuses a call it cannot run,
  // and the arm would report the hand tile's time.
  const bool is_hopper = caps.compute_major == 9 && caps.compute_minor == 0;
  if (run_cutlass && !is_hopper) {
    std::fprintf(stderr,
                 "gemm benchmark: the card is %d.%d, so the candidate arm is "
                 "not measured; it needs 9.0\n",
                 caps.compute_major, caps.compute_minor);
  }
  const bool measure_candidates = run_cutlass && is_hopper;
  if (run_cutlass && !caps.has_cublas) {
    std::fprintf(stderr,
                 "gemm benchmark: no cuBLAS, so the sweep has no reference\n");
    return 1;
  }

  // The rows.
  std::vector<SweepRow> rows;
  if (options.rows != "measurement") {
    const int count = GemmTileShapeCount();
    for (int index = 0; index < count; ++index) {
      rows.push_back(MakeProductionRow(index));
    }
  }
  if (options.rows != "production") {
    const int count = BenchShapeCount();
    for (int index = 0; index < count; ++index) {
      rows.push_back(MakeMeasurementRow(index));
    }
  }
  if (!options.row.empty()) {
    std::vector<SweepRow> selected;
    for (const SweepRow& row : rows) {
      if (row.name == options.row) selected.push_back(row);
    }
    if (selected.empty()) {
      std::fprintf(stderr, "gemm benchmark: no row named %s\n",
                   options.row.c_str());
      return 2;
    }
    rows = std::move(selected);
  }

  // The candidates the sweep measures.
  std::vector<int> candidates;
  if (measure_candidates) {
    if (options.candidate.empty() || options.candidate == "all") {
      for (int index = 0; index < GemmCandidateCount(); ++index) {
        candidates.push_back(index);
      }
    } else {
      const int index = GemmCandidateIndexFromName(options.candidate.c_str());
      if (index < 0) {
        std::fprintf(stderr, "gemm benchmark: no candidate named %s\n",
                     options.candidate.c_str());
        return 2;
      }
      candidates.push_back(index);
    }
  }

  // The scheduler cells of the run: one, unless a caller names a list. A run
  // with more than one cell is the scheduler sweep, and its report rows
  // carry the cell, so the table writer does not read them.
  const std::vector<SchedulerCell> cells = SchedulerCells(options);
  if (cells.size() > 1) {
    std::fprintf(stderr,
                 "gemm benchmark: %zu scheduler cells; each report row carries "
                 "its cell, so this report is a diagnostic and not a decision "
                 "input\n",
                 cells.size());
  }

  // The budget, and the rows it drops.
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  const std::size_t budget = BenchDeviceBudget(free_bytes);

  std::size_t largest = 0;
  std::vector<std::string> skip_reasons(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const std::size_t needed = RowWorkspaceBytes(rows[index]);
    skip_reasons[index] = BenchSizeSkipReason(needed, budget);
    if (skip_reasons[index].empty()) largest = std::max(largest, needed);
  }

  BenchReport report;
  UseDevicePeak(&report);
  report.SetRowIdentity("gemm", "forward");
  report.SetExtraInt("shape_rows", static_cast<long long>(rows.size()));
  report.SetExtraText("mode", options.arms.c_str());
  // The report's own `rows` key holds the measurements, so the family of the
  // rows is a separate field. `tools/gemm_rules.py` reads it and refuses a
  // report of another family.
  report.SetExtraText("row_family", options.rows.c_str());
  report.SetExtraInt("candidates", static_cast<long long>(candidates.size()));
  report.SetExtraInt("scheduler_cells", static_cast<long long>(cells.size()));
  report.SetExtraInt("device_budget_bytes", static_cast<long long>(budget));
  report.SetExtraInt("checked", options.check ? 1 : 0);

  Workspace workspace;
  if (largest > 0 && !workspace.Reserve(largest)) {
    std::fprintf(stderr,
                 "gemm benchmark: cannot allocate the %zu B workspace\n",
                 largest);
    return 1;
  }

  Flusher flusher;
  if (options.flush && !flusher.Init()) {
    std::fprintf(stderr,
                 "gemm benchmark: cannot allocate the flush buffer; the sweep "
                 "runs without a flush\n");
    options.flush = false;
  }
  report.SetExtraInt("flush_bytes", static_cast<long long>(flusher.bytes()));

  WindowTimer timer;
  const double flush_ms = options.flush ? TimeFlush(&timer, flusher, 5) : 0.0;
  report.SetExtra("flush_ms", flush_ms);
  if (options.flush) {
    std::fprintf(stderr, "gemm benchmark: flush %zu B, %.4f ms a call\n",
                 flusher.bytes(), flush_ms);
  }

  int measured = 0;
  int skipped = 0;
  int rejected = 0;
  // Arms that refused a call inside the timing window. Their row is dropped,
  // because the time is the fallback's and not theirs.
  int refused = 0;

  // The self test proves the check, and no arm runs when it fails.
  if (options.self_test) {
    bool ok = true;
    for (std::size_t index = 0; index < rows.size() && ok; ++index) {
      const SweepRow& row = rows[index];
      if (!skip_reasons[index].empty()) continue;
      RowBuffers buffers = Carve(workspace.base(), row);
      const std::size_t count = OutputElements(row.gemm);
      if (row.gemm.fp32_c) {
        FillDevice<float>(buffers.ref, count, 3);
      } else {
        FillDevice<ComputeType>(buffers.ref, count, 3);
      }
      Memcpy(buffers.c, buffers.ref, row.out_bytes, CopyDir::kDeviceToDevice);
      Synchronize();
      const CheckOutcome same =
          CheckOutput(row, buffers.c, buffers.ref, count, true);
      if (!same.ran || !same.matched) {
        std::fprintf(stderr,
                     "FAIL: the check rejects an identical copy of %s\n",
                     row.name.c_str());
        ok = false;
        break;
      }
      // Every element five percent low: the check must name the row and refuse
      // it. That is the candidate the sweep would otherwise call a winner.
      if (row.gemm.fp32_c) {
        ScaleKernel<float><<<kFillBlocks, kFillThreads>>>(
            static_cast<float*>(buffers.c), static_cast<long long>(count),
            0.95f);
      } else {
        ScaleKernel<ComputeType><<<kFillBlocks, kFillThreads>>>(
            static_cast<ComputeType*>(buffers.c), static_cast<long long>(count),
            0.95f);
      }
      CheckLaunch("self test perturbation");
      Synchronize();
      const CheckOutcome wrong =
          CheckOutput(row, buffers.c, buffers.ref, count, true);
      if (!wrong.ran || wrong.matched) {
        std::fprintf(stderr,
                     "FAIL: the check accepts a perturbed result of %s\n",
                     row.name.c_str());
        ok = false;
        break;
      }
      std::fprintf(stderr,
                   "gemm benchmark: self test: an identical copy of %s matches "
                   "and a five percent error reads %.2e, which is refused\n",
                   row.name.c_str(), wrong.relative_error);
    }
    workspace.Release();
    flusher.Release();
    if (!ok) return 1;
    std::fprintf(stderr, "gemm benchmark: self test passed\n");
    return 0;
  }

  for (std::size_t index = 0; index < rows.size(); ++index) {
    const SweepRow& row = rows[index];
    if (!skip_reasons[index].empty()) {
      std::fprintf(stderr, "gemm benchmark: %-24s skipped: %s\n",
                   row.name.c_str(), skip_reasons[index].c_str());
      ++skipped;
      continue;
    }
    RowBuffers buffers = Carve(workspace.base(), row);
    const std::size_t out_elements = OutputElements(row.gemm);
    const std::size_t a_elements = row.a_bytes / sizeof(ComputeType);
    const std::size_t b_elements = row.b_bytes / sizeof(ComputeType);
    FillDevice<ComputeType>(buffers.a, a_elements, 11);
    FillDevice<ComputeType>(buffers.b, b_elements, 23);

    // The result starts from a known pattern on every arm, so two arms are
    // compared from the same state.
    auto reset_c = [&] {
      if (row.gemm.fp32_c) {
        FillDevice<float>(buffers.c, out_elements, 37);
      } else {
        FillDevice<ComputeType>(buffers.c, out_elements, 37);
      }
    };

    GemmParams params;
    params.m = row.gemm.m;
    params.n = row.gemm.n;
    params.k = row.gemm.k;
    params.batch_count = row.gemm.batch_count;
    params.lda = row.gemm.lda;
    params.ldb = row.gemm.ldb;
    params.ldc = row.gemm.ldc;
    params.stride_a = row.gemm.stride_a;
    params.stride_b = row.gemm.stride_b;
    params.stride_c = row.gemm.stride_c;
    params.alpha = row.gemm.alpha;
    params.beta = row.gemm.beta;
    params.transpose_a = row.gemm.transpose_a;
    params.transpose_b = row.gemm.transpose_b;

    const ComputeType* a = buffers.a;
    const ComputeType* b = buffers.b;

    // One call of the arm the environment names. The seam reads the variable,
    // so an arm is one `setenv` and nothing else. The `seam` arm clears it
    // instead, which leaves the seam with the decision table: that is the arm a
    // training step runs, and it is what promotion is judged on.
    auto call_arm = [&](const char* arm) {
      if (std::string(arm) == "seam") {
        unsetenv("NANOCHAT_GEMM_LIB");
      } else {
        setenv("NANOCHAT_GEMM_LIB", arm, 1);
      }
      if (row.gemm.fp32_c) {
        nanochat::kernels::GemmWgrad(params, a, b,
                                     static_cast<GradientType*>(buffers.c));
      } else {
        Gemm(GemmMode::kForward, params, a, b,
             static_cast<ComputeType*>(buffers.c));
      }
    };

    // A measurement row takes its iteration count from its own work, so a small
    // row runs to the ceiling and a large row to the floor.
    const int row_iters =
        (options.auto_iters || !row.production)
            ? BenchIterationsFor(row.flops, report.peak_flops())
            : options.bench.iters;
    const int iters = std::max(1, row_iters);

    auto measure = [&](const char* arm) {
      return TimeWithFlush(
          &timer, [&] { call_arm(arm); }, iters,
          std::max(1, options.bench.rounds), std::max(0, options.bench.warmup),
          options.flush ? &flusher : nullptr, flush_ms);
    };

    bool have_reference = false;
    auto take_reference = [&] {
      Memcpy(buffers.ref, buffers.c, row.out_bytes, CopyDir::kDeviceToDevice);
      Synchronize();
      have_reference = true;
    };

    // cuBLAS is the reference. A run that does not measure it still calls it
    // once, so every arm has something correct to be compared with.
    if (run_cublas) {
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
      unsetenv("NANOCHAT_GEMM_CANDIDATE");
      reset_c();
      const Timing timing = measure("cublas");
      take_reference();
      const std::string name = "gemm:" + row.name + ":cublas";
      AddTiming(&report, TimedRow(name, row.shape), timing, row.flops);
      PrintMeasurement(row.name, "cublas", timing, 0, CheckOutcome());
      ++measured;
    } else {
      // The reference must be produced the way a measured arm produces its
      // result, or a row that accumulates is compared across a different number
      // of calls. A weight-gradient row states `beta = 1`, so eleven calls
      // leave eleven products in C and one call leaves one: the first
      // single-arm run of section 11.8 read exactly ten times the reference for
      // that reason. The arm is measured here and its timing is dropped,
      // because a run that does not measure cuBLAS still needs a reference of
      // the same shape.
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
      unsetenv("NANOCHAT_GEMM_CANDIDATE");
      reset_c();
      (void)measure("cublas");
      take_reference();
    }

    if (measure_candidates) {
      for (int candidate : candidates) {
        const GemmCandidateSpec& spec = GemmCandidateAt(candidate);
        const CutlassRefusal refusal =
            JudgeCutlass(row.gemm, buffers.a, buffers.b, buffers.c);
        if (refusal != CutlassRefusal::kNone) {
          std::fprintf(stderr, "gemm benchmark: %-24s %-10s refuses it: %s\n",
                       row.name.c_str(), spec.name,
                       CutlassRefusalName(refusal));
          continue;
        }
        // The refusal follows the shape and not the scheduler, so it is tested
        // once for every cell below.
        for (const SchedulerCell& cell : cells) {
          ApplyCell(cell);
          setenv("NANOCHAT_GEMM_CANDIDATE", spec.name, 1);
          unsetenv("NANOCHAT_GEMM_SCHEDULE");
          reset_c();
          ResetRefusals();
          const Timing timing = measure("cutlass");
          // An arm that refused a call inside the window did not run: the seam
          // handed the call to the hand tile and then cuBLAS, so the time is
          // another arm's. The row is dropped instead of reported.
          if (const int refusals = Refusals(); refusals > 0) {
            std::fprintf(stderr,
                         "gemm benchmark: %-24s %-10s refused %d call(s) at "
                         "run time, so the timing is not this arm's\n",
                         (row.name + cell.Suffix()).c_str(), spec.name,
                         refusals);
            ++refused;
            continue;
          }
          const CheckOutcome check =
              CheckOutput(row, buffers.c, buffers.ref, out_elements,
                          options.check && have_reference);
          if (check.ran && !check.matched) {
            std::fprintf(stderr,
                         "gemm benchmark: %-24s %-10s REJECTED by the check: "
                         "relative error %.2e\n",
                         row.name.c_str(), spec.name, check.relative_error);
            ++rejected;
            continue;
          }
          const int occupancy = CandidateOccupancy(candidate);
          const CutlassGrid grid = CandidateGrid(candidate, row.gemm);
          // The cell's suffix is part of the row name, so a sweep report holds
          // one row for each cell. A single-cell run states no suffix, so its
          // names stay the ones the table writer reads.
          const std::string variant = row.name + cell.Suffix();
          const std::string label =
              "gemm:" + variant + ":" + spec.name + ":cutlass";
          // A sweep states the grid on the row it measured, because a card job
          // returns the report and only the tail of the console. A run of one
          // cell states the shape it always did, so a decision report is
          // unchanged and the table writer still reads it.
          const std::string shape =
              cells.size() > 1 ? row.shape + GridText(grid) : row.shape;
          AddTiming(&report, TimedRow(label, shape, occupancy), timing,
                    row.flops);
          PrintMeasurement(variant, spec.name, timing, occupancy, check, grid);
          ++measured;
        }
      }
      unsetenv("NANOCHAT_GEMM_CANDIDATE");
      unsetenv("NANOCHAT_GEMM_SWIZZLE");
      unsetenv("NANOCHAT_GEMM_RASTER");
    }

    if (run_hand) {
      // The bf16 sweep and the fp32 sweep are two lists, and each has its own
      // environment variable: a name of one list indexes the other wrongly
      // (`docs/kernels.md` Part 3 section 6 phase 3).
      constexpr bool kFp32Tile = sizeof(ComputeType) == 4;
      const char* const kScheduleEnv =
          kFp32Tile ? "NANOCHAT_GEMM_FP32_SCHEDULE" : "NANOCHAT_GEMM_SCHEDULE";
      const int count = kFp32Tile ? nanochat::kernels::gemm_fp32::kScheduleCount
                                  : GemmScheduleCount();
      const int own = kFp32Tile ? nanochat::kernels::gemm_fp32::kDefaultSchedule
                                : row.own_schedule;
      const int first = options.schedule.empty() ? own : 0;
      const int last = options.schedule.empty() ? first + 1 : count;
      for (int schedule = first; schedule < last; ++schedule) {
        const char* schedule_name =
            kFp32Tile ? nanochat::kernels::gemm_fp32::ScheduleName(schedule)
                      : GemmScheduleName(schedule);
        if (options.schedule != "all" && !options.schedule.empty() &&
            options.schedule != schedule_name) {
          continue;
        }
        setenv(kScheduleEnv, schedule_name, 1);
        unsetenv("NANOCHAT_GEMM_CANDIDATE");
        reset_c();
        const Timing timing = measure("hand");
        if (!have_reference) take_reference();
        const CheckOutcome check =
            CheckOutput(row, buffers.c, buffers.ref, out_elements,
                        options.check && have_reference);
        if (check.ran && !check.matched) {
          std::fprintf(stderr,
                       "gemm benchmark: %-24s %-10s REJECTED by the check: "
                       "relative error %.2e\n",
                       row.name.c_str(), schedule_name, check.relative_error);
          ++rejected;
          continue;
        }
        const std::string label =
            std::string("gemm:") + row.name + ":" + schedule_name + ":hand";
        AddTiming(&report, TimedRow(label, row.shape), timing, row.flops);
        PrintMeasurement(row.name, schedule_name, timing, 0, check);
        ++measured;
      }
      unsetenv(kScheduleEnv);
    }

    // The shipped seam over one row: whatever the decision table names, which
    // is a candidate on a promoted row and cuBLAS on every other one. The check
    // compares it with the reference like any other arm, so a promoted row is
    // gated on a card before it ships.
    if (run_seam) {
      unsetenv("NANOCHAT_GEMM_SCHEDULE");
      unsetenv("NANOCHAT_GEMM_CANDIDATE");
      reset_c();
      const Timing timing = measure("seam");
      if (!have_reference) take_reference();
      const CheckOutcome check =
          CheckOutput(row, buffers.c, buffers.ref, out_elements,
                      options.check && have_reference);
      if (check.ran && !check.matched) {
        std::fprintf(stderr,
                     "gemm benchmark: %-24s %-10s REJECTED by the check: "
                     "relative error %.2e\n",
                     row.name.c_str(), "seam", check.relative_error);
        ++rejected;
      } else {
        const std::string label = std::string("gemm:") + row.name + ":seam";
        // No occupancy: the seam is cuBLAS on most rows and a candidate on a
        // promoted one, so one residency number would name neither arm.
        AddTiming(&report, TimedRow(label, row.shape), timing, row.flops);
        PrintMeasurement(row.name, "seam", timing, 0, check);
        ++measured;
      }
    }

    // The fp8 arms (docs/kernels.md section 10.4). The seam states one
    // dequantization value for each operand, so this arm quantizes the row's
    // own operands and times the same call twice: on cublasLt, which is the
    // shipped path, and on the Hopper candidate.
    //
    // The printed error is information and not a gate. The correctness gate of
    // the fp8 tier is `//tests:fp8_gemm_test` with the precision fixtures, and
    // this binary answers a rate. Nothing is promoted either, because
    // `NANOCHAT_GEMM_FP8` is unset in a training run.
    if (options.fp8 && is_hopper) {
      auto* a8 = static_cast<std::uint8_t*>(Alloc(a_elements));
      auto* b8 = static_cast<std::uint8_t*>(Alloc(b_elements));
      auto* a_scale = static_cast<float*>(Alloc(sizeof(float)));
      auto* b_scale = static_cast<float*>(Alloc(sizeof(float)));
      if (a8 != nullptr && b8 != nullptr && a_scale != nullptr &&
          b_scale != nullptr) {
        // The card proved one form, so a row outside it is reported and not
        // called. Three facts bound it: `GemmScaledFp8` exits on a batch, the
        // library answers status 15 for every form but the K-major pair, and
        // the arm faulted on the card at the first backward row
        // (docs/kernels.md section 10.4). A fatal here would lose the sweep
        // instead of one row.
        const bool kMajorPair = !params.transpose_a && params.transpose_b;
        const CutlassRefusal fp8_refusal =
            params.batch_count == 1 && kMajorPair
                ? JudgeScaledFp8(row.gemm, a8, b8, buffers.c)
                : CutlassRefusal::kExtent;
        if (fp8_refusal != CutlassRefusal::kNone) {
          std::fprintf(stderr, "gemm benchmark: %-24s fp8 arms skipped: %s\n",
                       row.name.c_str(), CutlassRefusalName(fp8_refusal));
        } else {
          // The forward's pair, E4M3 for both operands (docs/precision.md
          // section 7).
          QuantizeFp8Params quant_a;
          quant_a.count = static_cast<int>(a_elements);
          quant_a.format = Fp8Format::kE4M3;
          QuantizeFp8(quant_a, a, a8, a_scale);
          QuantizeFp8Params quant_b;
          quant_b.count = static_cast<int>(b_elements);
          QuantizeFp8(quant_b, b, b8, b_scale);

          ScaledGemmParams scaled;
          scaled.m = params.m;
          scaled.n = params.n;
          scaled.k = params.k;
          scaled.batch_count = 1;
          scaled.alpha = params.alpha;
          scaled.beta = params.beta;
          scaled.transpose_a = params.transpose_a;
          scaled.transpose_b = params.transpose_b;
          scaled.format_a = Fp8Format::kE4M3;
          scaled.format_b = Fp8Format::kE4M3;
          scaled.scale_a = a_scale;
          scaled.scale_b = b_scale;

          auto call_scaled = [&] {
            if (row.gemm.fp32_c) {
              GemmScaledFp8(GemmMode::kWgrad, scaled, a8, b8,
                            static_cast<GradientType*>(buffers.c));
            } else {
              GemmScaledFp8(GemmMode::kForward, scaled, a8, b8,
                            static_cast<ComputeType*>(buffers.c));
            }
          };
          if (!have_reference) take_reference();
          for (const char* selector : {"cublaslt", "cutlass"}) {
            if (std::string(selector) == "cutlass") {
              setenv("NANOCHAT_GEMM_FP8", "cutlass", 1);
            } else {
              unsetenv("NANOCHAT_GEMM_FP8");
            }
            reset_c();
            const Timing timing = TimeWithFlush(
                &timer, [&] { call_scaled(); }, iters,
                std::max(1, options.bench.rounds),
                std::max(0, options.bench.warmup),
                options.flush ? &flusher : nullptr, flush_ms);
            const CheckOutcome check =
                CheckOutput(row, buffers.c, buffers.ref, out_elements,
                            options.check && have_reference);
            const std::string label =
                std::string("gemm:") + row.name + ":fp8:" + selector;
            AddTiming(&report, TimedRow(label, row.shape), timing, row.flops);
            PrintMeasurement(row.name, (std::string("fp8:") + selector).c_str(),
                             timing, 0, check);
            ++measured;
          }
          unsetenv("NANOCHAT_GEMM_FP8");
        }
      }
      Free(a8);
      Free(b8);
      Free(a_scale);
      Free(b_scale);
    }
    Synchronize();
  }

  unsetenv("NANOCHAT_GEMM_LIB");
  flusher.Release();
  workspace.Release();

  if (measured == 0 && skipped == 0) {
    std::fprintf(stderr, "gemm benchmark: no row matched\n");
    return 2;
  }
  if (skipped > 0) {
    report.SetExtraInt("skipped_rows", static_cast<long long>(skipped));
  }
  if (rejected > 0) {
    report.SetExtraInt("rejected_arms", static_cast<long long>(rejected));
  }
  if (refused > 0) {
    report.SetExtraInt("refused_arms", static_cast<long long>(refused));
  }
  // The rate gate compares each arm's fraction with the cuBLAS row of the same
  // shape, and the fractions are on standard error beside the timings.
  report.Print(options.bench);
  return (rejected > 0 || refused > 0) ? 1 : 0;
}
