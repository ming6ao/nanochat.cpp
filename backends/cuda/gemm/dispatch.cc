// The closed shape list of the hand tensor-core GEMM, and the verdict
// (docs/tensor-core-gemm-plan.md section B.6).
//
// The Muon rows are derived from `muon_shape.h`, which `muon.cu` also uses, so
// the list and the call sites cannot drift. The Linear rows of the profiles
// join it the same way once the benchmark has measured them: a row is added by
// a measurement and not by a guess.
//
// Setting `NANOCHAT_GEMM_LIB` to "hand" or "cublas" overrides the table for an
// A/B measurement and for nothing else.

#include "backends/cuda/gemm/dispatch.h"

#include <cstdlib>
#include <utility>
#include <vector>

#include "backends/cuda/kernels/muon_gemm_shape.h"

namespace nanochat {
namespace kernels {

namespace {

// One Muon group of docs/model.md section The Muon step: a stacked set of
// `rows` by `cols` matrices, `batch` of them.
struct PolarGroup {
  const char* name;
  int rows;
  int cols;
  int batch;
};

// The five groups of the twenty-layer profile. `muon_benchmark` builds the same
// five, so its rows and the table's rows name one shape.
constexpr PolarGroup kPolarGroups[] = {
    {"5x12", 5, 12, 10},           {"640x1280", 640, 1280, 40},
    {"1280x1280", 1280, 1280, 40}, {"1280x5120", 1280, 5120, 20},
    {"5120x1280", 5120, 1280, 20},
};

struct Row {
  std::string name;
  ResolvedGemm gemm;
  bool enabled = false;
};

std::vector<Row>& Rows() {
  // Built once. The table is static, so a bit that a measurement turns on stays
  // on for the process, and a reader of the source sees every row and its bit.
  //
  // Two products of the five groups share a shape: the square of a group and
  // the apply product of a square group are one key, and the `1280x5120` and
  // `5120x1280` squares are the same `1280` square. A key is one call, so a
  // shared shape is one row and its name lists every product it serves.
  static std::vector<Row>* rows = [] {
    auto* table = new std::vector<Row>();
    auto add = [table](const std::string& name, const char* leaf,
                       const ResolvedGemm& gemm) {
      for (Row& row : *table) {
        if (SameShape(row.gemm, gemm)) {
          row.name += std::string("+") + leaf;
          return;
        }
      }
      table->push_back({name, gemm});
    };
    for (const PolarGroup& group : kPolarGroups) {
      const std::string base = std::string("muon:") + group.name + ":";
      add(base + "A", "A",
          ResolveGemm(
              muon_shape::SquareGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false));
      add(base + "A2", "A2",
          ResolveGemm(
              muon_shape::SquareSquareGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false));
      add(base + "prod", "prod",
          ResolveGemm(
              muon_shape::ApplyGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false));
    }
    return table;
  }();
  return *rows;
}

}  // namespace

const char* GemmTileModeName(GemmTileMode mode) {
  switch (mode) {
    case GemmTileMode::kHand:
      return "hand";
    case GemmTileMode::kCublas:
      return "cublas";
    case GemmTileMode::kAuto:
    default:
      return "auto";
  }
}

GemmTileMode GemmTileModeFromEnv() {
  const char* value = std::getenv("NANOCHAT_GEMM_LIB");
  if (value == nullptr) return GemmTileMode::kAuto;
  const std::string name(value);
  if (name == "hand") return GemmTileMode::kHand;
  if (name == "cublas") return GemmTileMode::kCublas;
  return GemmTileMode::kAuto;
}

bool GemmTileSelected(const ResolvedGemm& gemm, GemmTileMode mode) {
  if (!IsTileCallable(gemm)) return false;
  if (mode == GemmTileMode::kCublas) return false;
  if (mode == GemmTileMode::kHand) return true;
  for (const Row& row : Rows()) {
    if (row.enabled && SameShape(row.gemm, gemm)) return true;
  }
  return false;
}

int GemmTileShapeCount() { return static_cast<int>(Rows().size()); }

const char* GemmTileShapeName(int index) {
  if (index < 0 || index >= GemmTileShapeCount()) return "";
  return Rows()[static_cast<std::size_t>(index)].name.c_str();
}

const ResolvedGemm& GemmTileShapeGemm(int index) {
  static const ResolvedGemm kEmpty;
  if (index < 0 || index >= GemmTileShapeCount()) return kEmpty;
  return Rows()[static_cast<std::size_t>(index)].gemm;
}

bool GemmTileShapeEnabled(int index) {
  if (index < 0 || index >= GemmTileShapeCount()) return false;
  return Rows()[static_cast<std::size_t>(index)].enabled;
}

void SetGemmTileShapeEnabled(int index, bool enabled) {
  if (index < 0 || index >= GemmTileShapeCount()) return;
  Rows()[static_cast<std::size_t>(index)].enabled = enabled;
}

}  // namespace kernels
}  // namespace nanochat
