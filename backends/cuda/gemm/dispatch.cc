// The closed shape list of the hand tensor-core GEMM, and the verdict
// (docs/tensor-core-gemm-plan.md section B.6).
//
// The Muon rows are derived from `muon_shape.h`, which `muon.cu` also uses, so
// the list and the call sites cannot drift. The Linear rows come from
// `linear_shape.h`, which fills the same `GemmParams` that `src/ops.cc` fills,
// at the token count of one micro-batch of the throughput shape.
//
// Every row is inert. The sweep measured each shape against cuBLAS and no shape
// reached the promotion gate, so no bit is on:
// `backends/cuda/gemm/README.md` section 8 records why. A row also names the
// schedule its measurement chose, so a promotion is one bit and not a second
// sweep.
//
// Setting `NANOCHAT_GEMM_LIB` to "cutlass", "hand", or "cublas" overrides the
// table for an A/B measurement and for nothing else.

#include "backends/cuda/gemm/dispatch.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/linear_shape.h"
#include "backends/cuda/kernels/muon_gemm_shape.h"
#include "nanochat/config.h"

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

// The token positions of one micro-batch of the throughput shape: sixteen
// sequences of 1024, which is the batch `docs/model.md` section The cost on the
// H100 bounds and the batch a `big` job runs. A Linear call's m is that
// product, so it is part of the shape's key.
constexpr int kLinearRows = 16 * 1024;

// The model shape whose Linear calls the table holds: the `big` profile of
// `python/nanochat_cpp/pipeline.py` PROFILES, which is the twenty-layer shape
// of `docs/model.md`. The layer widths then come from `Config`, so the table
// and the call sites cannot disagree about which layers a step issues.
Config LinearModel() {
  Config config;
  config.num_layers = 20;
  config.num_heads = 10;
  config.num_kv_heads = 5;
  config.hidden_dim = 1280;
  config.seq_len = 1024;
  config.vocab_size = 32768;
  config.padded_vocab_size = 32768;
  config.window_pattern = "SSL";
  return config;
}

struct Row {
  std::string name;
  ResolvedGemm gemm;
  // The schedule the sweep chose for this shape
  // (docs/tensor-core-gemm-plan.md section B.4). The bit is off for every row,
  // so this is the record of the measurement: a promotion is one bit and no
  // second sweep.
  int schedule = gemm_tile::kDefaultSchedule;
  bool enabled = false;
};

// The sweep left three rules. The square tile with a 64-deep slice and the
// swizzled order wins every model shape that stores the compute type, a
// 96-deep slice wins the weight gradients, whose fp32 store is twice the bytes,
// and the 5 by 12 group takes the small tile because its whole call is launch
// overhead.
constexpr int kWideShapeSchedule = gemm_tile::kScheduleKDeepSwizzle;
constexpr int kWeightGradientSchedule = gemm_tile::kScheduleKDeeper;
constexpr int kTinyShapeSchedule = gemm_tile::kScheduleSmall;

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
                       const ResolvedGemm& gemm, int schedule) {
      for (Row& row : *table) {
        if (SameShape(row.gemm, gemm)) {
          row.name += std::string("+") + leaf;
          return;
        }
      }
      table->push_back({name, gemm, schedule});
    };
    for (const PolarGroup& group : kPolarGroups) {
      const std::string base = std::string("muon:") + group.name + ":";
      // The 5 by 12 group is a handful of microseconds on either path, so its
      // fraction is launch overhead and not a rate (section B.8, gate G4).
      const int schedule = group.rows == 5 && group.cols == 12
                               ? kTinyShapeSchedule
                               : kWideShapeSchedule;
      add(base + "A", "A",
          ResolveGemm(
              muon_shape::SquareGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false),
          schedule);
      add(base + "A2", "A2",
          ResolveGemm(
              muon_shape::SquareSquareGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false),
          schedule);
      add(base + "prod", "prod",
          ResolveGemm(
              muon_shape::ApplyGemm(group.rows, group.cols, group.batch),
              /*fp32_c=*/false),
          schedule);
    }
    // The Linear calls of the same model shape. A row's name puts the call
    // first, so a shape two layers share reads `linear:fwd:q+proj`.
    for (const linear_shape::Layer& layer :
         linear_shape::Layers(LinearModel())) {
      const std::string base = "linear:";
      add(base + "fwd:" + layer.name, layer.name,
          ResolveGemm(linear_shape::Forward(kLinearRows, layer.in, layer.out),
                      /*fp32_c=*/false),
          kWideShapeSchedule);
      add(base + "dgrad:" + layer.name, layer.name,
          ResolveGemm(linear_shape::Dgrad(kLinearRows, layer.in, layer.out),
                      /*fp32_c=*/false),
          kWideShapeSchedule);
      add(base + "wgrad:" + layer.name, layer.name,
          ResolveGemm(linear_shape::Wgrad(kLinearRows, layer.in, layer.out),
                      /*fp32_c=*/true),
          kWeightGradientSchedule);
    }
    return table;
  }();
  return *rows;
}

}  // namespace

const char* GemmTileModeName(GemmTileMode mode) {
  switch (mode) {
    case GemmTileMode::kCutlass:
      return "cutlass";
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
  if (name == "cutlass") return GemmTileMode::kCutlass;
  if (name == "hand") return GemmTileMode::kHand;
  if (name == "cublas") return GemmTileMode::kCublas;
  return GemmTileMode::kAuto;
}

int GemmScheduleCount() { return gemm_tile::kScheduleCount; }

const char* GemmScheduleName(int index) {
  if (index < 0 || index >= GemmScheduleCount()) return "";
  return gemm_tile::kScheduleNames[index];
}

int GemmScheduleIndexFromEnv() {
  const char* value = std::getenv("NANOCHAT_GEMM_SCHEDULE");
  if (value == nullptr) return -1;
  for (int index = 0; index < GemmScheduleCount(); ++index) {
    if (std::string(value) == GemmScheduleName(index)) return index;
  }
  return -1;
}

int GemmTileScheduleFor(const ResolvedGemm& gemm, GemmTileMode mode) {
  if (!IsTileCallable(gemm)) return -1;
  // The tile serves neither the reference nor the Hopper mainloop: those are
  // two other libraries, and a call they take is not this tile's.
  if (mode == GemmTileMode::kCublas || mode == GemmTileMode::kCutlass)
    return -1;
  // The environment overrides the row's schedule, so a measurement can sweep
  // one shape over the whole list.
  const int forced = GemmScheduleIndexFromEnv();
  if (mode == GemmTileMode::kHand) {
    return forced >= 0 ? forced : gemm_tile::kDefaultSchedule;
  }
  for (const Row& row : Rows()) {
    if (row.enabled && SameShape(row.gemm, gemm)) {
      return forced >= 0 ? forced : row.schedule;
    }
  }
  return -1;
}

bool GemmTileSelected(const ResolvedGemm& gemm, GemmTileMode mode) {
  return GemmTileScheduleFor(gemm, mode) >= 0;
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

int GemmTileShapeSchedule(int index) {
  if (index < 0 || index >= GemmTileShapeCount())
    return gemm_tile::kDefaultSchedule;
  return Rows()[static_cast<std::size_t>(index)].schedule;
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
