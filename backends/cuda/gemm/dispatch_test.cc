// The gate of the hand GEMM's shape table and its verdict
// (docs/tensor-core-gemm-plan.md section B.6).
//
// The table is the closed shape list, and every row starts inert, so the checks
// are: the list matches the shapes `muon.cu` issues, a row selects only its own
// key, and a bit turned on is the only way a shape changes hands. Every row is
// inert in the shipped table, so a row is turned on here and turned off again.
//
// It also re-checks the seam's default rules, because `ResolveGemm` and
// `GemmImpl` must agree and only one of the two runs on a card, and it checks
// the Hopper mainloop's own rule and the fp8 arm's, which have no other gate
// off a card.
//
// CPU only: no device, no driver, no toolkit.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "backends/cuda/gemm/cutlass_candidates.h"
#include "backends/cuda/gemm/cutlass_rules.h"
#include "backends/cuda/gemm/dispatch.h"
#include "backends/cuda/gemm/gemm_plan.h"
#include "backends/cuda/gemm/gemm_shape.h"
#include "backends/cuda/gemm/linear_shape.h"
#include "backends/cuda/kernels/muon_gemm_shape.h"
#include "nanochat/kernels.h"

namespace {

using nanochat::GemmParams;
using nanochat::kernels::CutlassCandidateFor;
using nanochat::kernels::CutlassRefusal;
using nanochat::kernels::CutlassRefusalName;
using nanochat::kernels::GemmCandidateCount;
using nanochat::kernels::GemmRoute;
using nanochat::kernels::GemmRuleFor;
using nanochat::kernels::GemmRuleRow;
using nanochat::kernels::GemmScheduleCount;
using nanochat::kernels::GemmScheduleIndexFromEnv;
using nanochat::kernels::GemmScheduleName;
using nanochat::kernels::GemmTileMode;
using nanochat::kernels::GemmTileModeFromEnv;
using nanochat::kernels::GemmTileModeName;
using nanochat::kernels::GemmTileScheduleFor;
using nanochat::kernels::GemmTileSelected;
using nanochat::kernels::GemmTileShapeCount;
using nanochat::kernels::GemmTileShapeEnabled;
using nanochat::kernels::GemmTileShapeGemm;
using nanochat::kernels::GemmTileShapeName;
using nanochat::kernels::GemmTileShapeSchedule;
using nanochat::kernels::IsCutlassCallable;
using nanochat::kernels::IsTileCallable;
using nanochat::kernels::JudgeCutlass;
using nanochat::kernels::JudgeScaledFp8;
using nanochat::kernels::kCutlassMinOutputElements;
using nanochat::kernels::kFp8Tile;
using nanochat::kernels::kGemmRuleCount;
using nanochat::kernels::kGemmRules;
using nanochat::kernels::ResolvedGemm;
using nanochat::kernels::ResolveGemm;
using nanochat::kernels::SameShape;
using nanochat::kernels::SetGemmTileShapeEnabled;

int g_failures = 0;

void Fail(const std::string& what, const char* case_name) {
  std::fprintf(stderr, "FAIL: %s (%s)\n", what.c_str(), case_name);
  ++g_failures;
}

void Expect(bool condition, const std::string& what, const char* case_name) {
  if (!condition) Fail(what, case_name);
}

// The five Muon groups of docs/model.md section The Muon step. These are the
// literals the table is checked against.
struct Group {
  int rows;
  int cols;
  int batch;
};

constexpr Group kGroups[] = {
    {5, 12, 10},      {640, 1280, 40},  {1280, 1280, 40},
    {1280, 5120, 20}, {5120, 1280, 20},
};

// The Linear layers of the throughput shape, written out: the twenty-layer
// profile of `python/nanochat_cpp/pipeline.py` PROFILES, whose hidden width is
// 1280, whose key and value width is 640, whose feed-forward width is 5120, and
// whose classifier width is the padded vocabulary.
struct LinearLayer {
  const char* name;
  int in;
  int out;
};

constexpr LinearLayer kLinearLayers[] = {
    {"q", 1280, 1280},  {"kv", 1280, 640},     {"proj", 1280, 1280},
    {"fc", 1280, 5120}, {"cproj", 5120, 1280}, {"cls", 1280, 32768},
};

// The token positions of one micro-batch: sixteen sequences of 1024.
constexpr int kLinearRows = 16 * 1024;

// The table is one row per shape. The five groups contribute fifteen products,
// and two pairs of them are one shape each: `1280x1280:A2` is also its product,
// and the `1280x5120` square is the `5120x1280` square. The Linear side holds
// three calls of six layers, and the query projection shares its shape with the
// attention output projection in all three. Thirteen rows and fifteen rows.
// The decision table of `cutlass_rules.h` (docs/kernels.md section 11.4 item
// 5).
//
// The table is written by `tools/gemm_rules.py` from a card run, so this gate
// states the two things a written row must satisfy. A row must key a shape of
// the closed list, because a row no call can match is a decision that never
// applies. And a row must agree with itself: a promoted row names a candidate
// of the registry, and every other route names none.
//
// The shipped table holds no row, so both checks are vacuous until a card run
// writes one. The gate is here for that run.
void CheckRuleTable() {
  const char* kCase = "rule table";
  for (int index = 0; index < kGemmRuleCount; ++index) {
    const GemmRuleRow& rule = kGemmRules[index];
    bool found = false;
    for (int row = 0; row < GemmTileShapeCount() && !found; ++row) {
      const ResolvedGemm& gemm = GemmTileShapeGemm(row);
      found = rule.m == gemm.m && rule.n == gemm.n && rule.k == gemm.k &&
              rule.batch_count == gemm.batch_count &&
              rule.transpose_a == gemm.transpose_a &&
              rule.transpose_b == gemm.transpose_b &&
              rule.fp32_c == gemm.fp32_c;
    }
    Expect(found,
           "rule " + std::to_string(index) + " keys a shape of the closed list",
           kCase);
    const bool promoted = rule.route == GemmRoute::kCandidate;
    Expect(promoted == (rule.candidate >= 0),
           "rule " + std::to_string(index) +
               " names a candidate exactly when it promotes one",
           kCase);
    Expect(rule.candidate < GemmCandidateCount(),
           "rule " + std::to_string(index) +
               " names a candidate the registry holds",
           kCase);
    // The lookup answers the row the table states, and the seam's question is
    // the same answer.
    for (int row = 0; row < GemmTileShapeCount(); ++row) {
      const ResolvedGemm& gemm = GemmTileShapeGemm(row);
      if (rule.m != gemm.m || rule.n != gemm.n || rule.k != gemm.k ||
          rule.batch_count != gemm.batch_count ||
          rule.transpose_a != gemm.transpose_a ||
          rule.transpose_b != gemm.transpose_b || rule.fp32_c != gemm.fp32_c) {
        continue;
      }
      Expect(GemmRuleFor(gemm) == &rule,
             "rule " + std::to_string(index) + " is the rule of its shape",
             kCase);
      Expect(CutlassCandidateFor(gemm) == rule.candidate,
             "rule " + std::to_string(index) +
                 " answers the seam's candidate question",
             kCase);
    }
  }
}

void CheckTableShape() {
  const char* kCase = "table shape";
  Expect(GemmTileShapeCount() == 28, "the closed list holds twenty-eight rows",
         kCase);
  bool shared = false;
  for (int i = 0; i < GemmTileShapeCount(); ++i) {
    if (std::string(GemmTileShapeName(i)).find('+') != std::string::npos) {
      shared = true;
    }
  }
  Expect(shared, "a shared shape names both of its products", kCase);

  std::set<std::string> names;
  for (int i = 0; i < GemmTileShapeCount(); ++i) {
    const std::string name = GemmTileShapeName(i);
    Expect(!name.empty(), "every row has a name", kCase);
    Expect(names.insert(name).second, "every name is unique: " + name, kCase);
    Expect(!GemmTileShapeEnabled(i), "the shipped row is inert: " + name,
           kCase);
    Expect(IsTileCallable(GemmTileShapeGemm(i)),
           "every row is a callable shape: " + name, kCase);
  }
}

// Every shape `muon.cu` issues is one row, and the row matches it. A change to
// the Polar Express shapes therefore fails here until the table follows.
void CheckTableMatchesTheCallSites() {
  const char* kCase = "table matches the call sites";
  for (const Group& group : kGroups) {
    const GemmParams issued[] = {
        nanochat::kernels::muon_shape::SquareGemm(group.rows, group.cols,
                                                  group.batch),
        nanochat::kernels::muon_shape::SquareSquareGemm(group.rows, group.cols,
                                                        group.batch),
        nanochat::kernels::muon_shape::ApplyGemm(group.rows, group.cols,
                                                 group.batch),
    };
    for (const GemmParams& params : issued) {
      const ResolvedGemm gemm = ResolveGemm(params, /*fp32_c=*/false);
      int matches = 0;
      for (int i = 0; i < GemmTileShapeCount(); ++i) {
        if (SameShape(GemmTileShapeGemm(i), gemm)) ++matches;
      }
      const std::string label = std::to_string(group.rows) + "x" +
                                std::to_string(group.cols) + " batch " +
                                std::to_string(group.batch);
      Expect(matches == 1,
             "one row for " + label + ", found " + std::to_string(matches),
             kCase);
    }
  }
}

// Every Linear call of the throughput shape is one row, and the row matches it.
// The three calls are the literals `src/ops.cc` issues: `LinearForward`
// transposes its weight, `LinearDgrad` transposes nothing, and `LinearWgrad`
// transposes its first operand and stores fp32.
void CheckTableMatchesTheLinearCalls() {
  const char* kCase = "table matches the Linear calls";
  for (const LinearLayer& layer : kLinearLayers) {
    GemmParams forward;
    forward.m = kLinearRows;
    forward.n = layer.out;
    forward.k = layer.in;
    forward.transpose_b = true;

    GemmParams dgrad;
    dgrad.m = kLinearRows;
    dgrad.n = layer.in;
    dgrad.k = layer.out;

    GemmParams wgrad;
    wgrad.m = layer.out;
    wgrad.n = layer.in;
    wgrad.k = kLinearRows;
    wgrad.transpose_a = true;

    struct Wanted {
      const char* call;
      GemmParams params;
      bool fp32_c;
    };
    const Wanted wanted[] = {{"fwd", forward, false},
                             {"dgrad", dgrad, false},
                             {"wgrad", wgrad, true}};
    for (const Wanted& want : wanted) {
      const ResolvedGemm gemm = ResolveGemm(want.params, want.fp32_c);
      int matches = 0;
      std::string name;
      for (int i = 0; i < GemmTileShapeCount(); ++i) {
        if (SameShape(GemmTileShapeGemm(i), gemm)) {
          ++matches;
          name = GemmTileShapeName(i);
        }
      }
      const std::string label = std::string(want.call) + " " + layer.name;
      Expect(matches == 1,
             "one row for " + label + ", found " + std::to_string(matches),
             kCase);
      Expect(name.find(layer.name) != std::string::npos,
             "the row of " + label + " names its layer: " + name, kCase);
      Expect(name.find(want.call) != std::string::npos,
             "the row of " + label + " names its call: " + name, kCase);
    }
  }

  // The layer widths come from one model shape, so the derivation in
  // `linear_shape.h` must give these six layers and no other.
  nanochat::Config config;
  config.num_layers = 20;
  config.num_heads = 10;
  config.num_kv_heads = 5;
  config.hidden_dim = 1280;
  config.seq_len = 1024;
  config.vocab_size = 32768;
  config.padded_vocab_size = 32768;
  config.window_pattern = "SSL";
  const std::vector<nanochat::kernels::linear_shape::Layer> layers =
      nanochat::kernels::linear_shape::Layers(config);
  Expect(layers.size() == sizeof(kLinearLayers) / sizeof(kLinearLayers[0]),
         "the model shape holds six Linear layers", kCase);
  for (std::size_t i = 0; i < layers.size() && i < 6; ++i) {
    Expect(std::string(layers[i].name) == kLinearLayers[i].name,
           "layer " + std::to_string(i) + " has the right name", kCase);
    Expect(layers[i].in == kLinearLayers[i].in &&
               layers[i].out == kLinearLayers[i].out,
           "layer " + std::string(layers[i].name) + " has the right widths",
           kCase);
  }
}

// A row selects its own key, and only after a measurement turns it on.
void CheckSelection() {
  const char* kCase = "selection";
  const int row = 3;
  const ResolvedGemm key = GemmTileShapeGemm(row);

  Expect(!GemmTileSelected(key, GemmTileMode::kAuto),
         "an inert row selects nothing", kCase);

  SetGemmTileShapeEnabled(row, true);
  Expect(GemmTileSelected(key, GemmTileMode::kAuto),
         "an enabled row selects its own key", kCase);
  Expect(GemmTileShapeEnabled(row), "the row reports its own bit", kCase);

  // A near miss on any part of the key is a different shape.
  ResolvedGemm shifted = key;
  shifted.ldb += 1;
  Expect(!GemmTileSelected(shifted, GemmTileMode::kAuto),
         "a different leading dimension does not match", kCase);
  shifted = key;
  shifted.batch_count += 1;
  Expect(!GemmTileSelected(shifted, GemmTileMode::kAuto),
         "a different batch count does not match", kCase);
  shifted = key;
  shifted.fp32_c = !key.fp32_c;
  Expect(!GemmTileSelected(shifted, GemmTileMode::kAuto),
         "a different store type does not match", kCase);

  SetGemmTileShapeEnabled(row, false);
  Expect(!GemmTileSelected(key, GemmTileMode::kAuto),
         "the row goes back to inert", kCase);
}

// The measurement switch overrides the table, and the switch never selects a
// call the tile refuses.
void CheckMode() {
  const char* kCase = "mode";
  const ResolvedGemm key = GemmTileShapeGemm(0);
  Expect(!GemmTileSelected(key, GemmTileMode::kCublas),
         "the cublas mode selects nothing", kCase);
  Expect(GemmTileSelected(key, GemmTileMode::kHand),
         "the hand mode selects a callable shape", kCase);
  Expect(!GemmTileSelected(key, GemmTileMode::kCutlass),
         "the cutlass mode is not the tile's", kCase);

  ResolvedGemm empty = key;
  empty.k = 0;
  Expect(!GemmTileSelected(empty, GemmTileMode::kHand),
         "the hand mode still refuses an empty extent", kCase);
  empty = key;
  empty.batch_count = 0;
  Expect(!GemmTileSelected(empty, GemmTileMode::kHand),
         "the hand mode refuses an empty batch", kCase);

  Expect(std::string(GemmTileModeName(GemmTileMode::kAuto)) == "auto",
         "the auto mode names itself", kCase);
  Expect(std::string(GemmTileModeName(GemmTileMode::kHand)) == "hand",
         "the hand mode names itself", kCase);
  Expect(std::string(GemmTileModeName(GemmTileMode::kCublas)) == "cublas",
         "the cublas mode names itself", kCase);
  Expect(std::string(GemmTileModeName(GemmTileMode::kCutlass)) == "cutlass",
         "the cutlass mode names itself", kCase);
}

// The sweep is a closed list of its own, and every row names one of it.
void CheckSchedules() {
  const char* kCase = "schedules";
  Expect(GemmScheduleCount() >= 2, "the sweep holds more than one schedule",
         kCase);

  std::set<std::string> names;
  for (int i = 0; i < GemmScheduleCount(); ++i) {
    const std::string name = GemmScheduleName(i);
    Expect(!name.empty(), "every schedule has a name", kCase);
    Expect(names.insert(name).second, "every schedule name is unique: " + name,
           kCase);
  }
  Expect(std::string(GemmScheduleName(-1)).empty() &&
             std::string(GemmScheduleName(GemmScheduleCount())).empty(),
         "an index outside the sweep names nothing", kCase);

  for (int i = 0; i < GemmTileShapeCount(); ++i) {
    const int schedule = GemmTileShapeSchedule(i);
    Expect(schedule >= 0 && schedule < GemmScheduleCount(),
           "every row names a schedule in the sweep: " +
               std::string(GemmTileShapeName(i)),
           kCase);
  }
}

void CheckScheduleFromEnv() {
  const char* kCase = "schedule from the environment";
  unsetenv("NANOCHAT_GEMM_SCHEDULE");
  Expect(GemmScheduleIndexFromEnv() == -1, "unset names no schedule", kCase);
  setenv("NANOCHAT_GEMM_SCHEDULE", GemmScheduleName(1), 1);
  Expect(GemmScheduleIndexFromEnv() == 1, "a name selects its index", kCase);
  setenv("NANOCHAT_GEMM_SCHEDULE", "no such schedule", 1);
  Expect(GemmScheduleIndexFromEnv() == -1, "an unknown name names nothing",
         kCase);

  // The override reaches the verdict, which is how a card run sweeps a row.
  const int row = 2;
  SetGemmTileShapeEnabled(row, true);
  setenv("NANOCHAT_GEMM_SCHEDULE", GemmScheduleName(2), 1);
  Expect(GemmTileScheduleFor(GemmTileShapeGemm(row), GemmTileMode::kAuto) == 2,
         "the environment overrides the row's schedule", kCase);
  unsetenv("NANOCHAT_GEMM_SCHEDULE");
  Expect(GemmTileScheduleFor(GemmTileShapeGemm(row), GemmTileMode::kAuto) ==
             GemmTileShapeSchedule(row),
         "without the variable the row owns its schedule", kCase);
  SetGemmTileShapeEnabled(row, false);
}

void CheckModeFromEnv() {
  const char* kCase = "mode from the environment";
  unsetenv("NANOCHAT_GEMM_LIB");
  Expect(GemmTileModeFromEnv() == GemmTileMode::kAuto, "unset is auto", kCase);
  setenv("NANOCHAT_GEMM_LIB", "hand", 1);
  Expect(GemmTileModeFromEnv() == GemmTileMode::kHand, "hand is hand", kCase);
  setenv("NANOCHAT_GEMM_LIB", "cublas", 1);
  Expect(GemmTileModeFromEnv() == GemmTileMode::kCublas, "cublas is cublas",
         kCase);
  setenv("NANOCHAT_GEMM_LIB", "cutlass", 1);
  Expect(GemmTileModeFromEnv() == GemmTileMode::kCutlass, "cutlass is cutlass",
         kCase);
  setenv("NANOCHAT_GEMM_LIB", "something else", 1);
  Expect(GemmTileModeFromEnv() == GemmTileMode::kAuto,
         "an unknown value is auto", kCase);
  unsetenv("NANOCHAT_GEMM_LIB");
}

// The Hopper mainloop's own rule. It is precision free and device free, so this
// is the only gate it has off a card: `cutlass_bf16.cu` has no emulated leg.
void CheckCutlassVerdict() {
  const char* kCase = "cutlass verdict";
  // Four aligned, distinct host addresses, so the pointer test is a test.
  alignas(64) static char storage[256];
  const void* a = storage;
  const void* b = storage + 64;
  void* c = storage + 128;

  // A shape the sweep measured, resolved by the same rule `GemmImpl` applies.
  GemmParams params;
  params.m = 1280;
  params.n = 5120;
  params.k = 1280;
  params.batch_count = 20;
  params.transpose_a = false;
  params.transpose_b = true;
  const ResolvedGemm model = ResolveGemm(params, false);
  Expect(JudgeCutlass(model, a, b, c) == CutlassRefusal::kNone,
         "a model shape that meets every rule is taken", kCase);
  Expect(IsCutlassCallable(model, a, b, c), "the same shape is callable",
         kCase);

  // Extent first: an empty product is refused before anything else is tested.
  ResolvedGemm empty = model;
  empty.k = 0;
  Expect(JudgeCutlass(empty, a, b, c) == CutlassRefusal::kExtent,
         "an empty extent is refused for its extent", kCase);

  // Then size: the 5 by 12 group is 60 outputs, under one 64 by 64 tile.
  GemmParams tiny;
  tiny.m = 5;
  tiny.n = 12;
  tiny.k = 12;
  tiny.batch_count = 10;
  const ResolvedGemm small = ResolveGemm(tiny, false);
  Expect(JudgeCutlass(small, a, b, c) == CutlassRefusal::kSize,
         "the 5 by 12 group stays on the hand tile", kCase);
  Expect(static_cast<std::int64_t>(5) * 12 < kCutlassMinOutputElements,
         "the group really is under the crossover", kCase);

  // Then the stride. A bf16 row needs eight elements of the sixteen bytes, and
  // an fp32 row needs four.
  ResolvedGemm odd = model;
  odd.lda = 1179;
  Expect(JudgeCutlass(odd, a, b, c) == CutlassRefusal::kStride,
         "an operand stride off the boundary is refused", kCase);
  // 1284 elements is a whole number of fp32 sixteen-byte rows and not a whole
  // number of bf16 ones, so it separates the two store alignments.
  odd = model;
  odd.ldc = 1284;
  odd.stride_c = static_cast<std::int64_t>(1280) * 1284;
  Expect(JudgeCutlass(odd, a, b, c) == CutlassRefusal::kStride,
         "a bf16 store row off the boundary is refused", kCase);
  ResolvedGemm fp32_odd = ResolveGemm(params, true);
  fp32_odd.ldc = 1284;
  fp32_odd.stride_c = static_cast<std::int64_t>(1280) * 1284;
  Expect(JudgeCutlass(fp32_odd, a, b, c) == CutlassRefusal::kNone,
         "four fp32 elements share the boundary where eight bf16 do", kCase);
  odd = model;
  odd.ldc = 1282;
  odd.stride_c = static_cast<std::int64_t>(1280) * 1282;
  Expect(JudgeCutlass(odd, a, b, c) == CutlassRefusal::kStride,
         "a store row off both boundaries is refused", kCase);

  // Then the pointer, which is a different refusal from the stride so a report
  // can name it.
  const void* misaligned = storage + 2;
  Expect(JudgeCutlass(model, misaligned, b, c) == CutlassRefusal::kPointer,
         "a pointer off the boundary is refused for its pointer", kCase);
  Expect(JudgeCutlass(model, nullptr, b, c) == CutlassRefusal::kPointer,
         "a missing operand is refused for its pointer", kCase);

  // Batch strides only matter past one batch.
  ResolvedGemm batched = model;
  batched.stride_b = 3;
  Expect(JudgeCutlass(batched, a, b, c) == CutlassRefusal::kStride,
         "a batch stride off the boundary is refused", kCase);
  batched.batch_count = 1;
  Expect(JudgeCutlass(batched, a, b, c) == CutlassRefusal::kNone,
         "a batch stride is not tested at one batch", kCase);

  Expect(std::string(CutlassRefusalName(CutlassRefusal::kNone)) == "none",
         "the taken verdict names itself", kCase);
  Expect(std::string(CutlassRefusalName(CutlassRefusal::kPointer)) == "pointer",
         "the pointer verdict names itself", kCase);
}

// The fp8 arm's own rule (docs/kernels.md section 10.4). `cutlass_fp8.cu`
// includes the vendor headers, so only an fp8 build compiles it and only a card
// runs it; this verdict is the whole of the arm that a host can check.
void CheckScaledFp8Verdict() {
  const char* kCase = "fp8 verdict";
  alignas(64) static char storage[256];
  const void* a = storage;
  const void* b = storage + 64;
  void* c = storage + 128;

  // A shape the sweep measured, held to one batch because the fp8 seam states
  // no batch. The operands are K-major, which is the only pair the SM90 fp8
  // GMMA has; `CallSupported` states that rule and this test states the rest.
  GemmParams params;
  params.m = 1280;
  params.n = 5120;
  params.k = 1280;
  params.batch_count = 1;
  params.transpose_a = false;
  params.transpose_b = true;
  const ResolvedGemm gemm = ResolveGemm(params, false);
  Expect(JudgeScaledFp8(gemm, a, b, c) == CutlassRefusal::kNone,
         "a dense aligned fp8 call is taken", kCase);

  // The refusal order is the bf16 one, so a single table explains both arms.
  ResolvedGemm empty = gemm;
  empty.k = 0;
  Expect(JudgeScaledFp8(empty, a, b, c) == CutlassRefusal::kExtent,
         "an empty fp8 extent is refused for its extent", kCase);

  GemmParams tiny;
  tiny.m = 5;
  tiny.n = 12;
  tiny.k = 12;
  const ResolvedGemm small = ResolveGemm(tiny, false);
  Expect(JudgeScaledFp8(small, a, b, c) == CutlassRefusal::kSize,
         "the 5 by 12 group stays off the fp8 arm", kCase);

  // One fp8 element is one byte, so a row shares the sixteen-byte boundary
  // every sixteen elements and not every eight. 1272 is a whole number of bf16
  // rows and is not a whole number of fp8 ones, so it separates the two rules.
  ResolvedGemm odd = gemm;
  odd.lda = 1272;
  Expect(JudgeCutlass(odd, a, b, c) == CutlassRefusal::kNone,
         "1272 bf16 elements share the boundary", kCase);
  Expect(JudgeScaledFp8(odd, a, b, c) == CutlassRefusal::kStride,
         "1272 fp8 elements do not share it", kCase);

  // The store keeps the call's own element, so the C alignment is the bf16
  // rule on one call and the fp32 rule on a weight gradient.
  odd = gemm;
  odd.ldc = 1284;
  odd.stride_c = static_cast<std::int64_t>(1280) * 1284;
  Expect(JudgeScaledFp8(odd, a, b, c) == CutlassRefusal::kStride,
         "a bf16 fp8 store row off the boundary is refused", kCase);
  const ResolvedGemm fp32_gemm = ResolveGemm(params, true);
  Expect(JudgeScaledFp8(fp32_gemm, a, b, c) == CutlassRefusal::kNone,
         "four fp32 elements share the store boundary", kCase);

  // A batch is an extent refusal here and not a stride one, because the fp8
  // seam carries no batch stride at all.
  ResolvedGemm batched = gemm;
  batched.batch_count = 2;
  Expect(JudgeScaledFp8(batched, a, b, c) == CutlassRefusal::kExtent,
         "a batch is refused by the fp8 arm", kCase);

  Expect(JudgeScaledFp8(gemm, storage + 2, b, c) == CutlassRefusal::kPointer,
         "a pointer off the boundary is refused for its pointer", kCase);
  Expect(JudgeScaledFp8(gemm, nullptr, b, c) == CutlassRefusal::kPointer,
         "a missing fp8 operand is refused for its pointer", kCase);

  // The tile is the probe's `balanced` row, at half the N of the bf16 `wide-n`
  // row, because the fp8 accumulation path keeps two fp32 accumulators. The
  // fp8 operand halves the staged bytes: 64 * (128 + 128) * 5 stages is
  // 81920 B of the 227 KiB window.
  Expect(kFp8Tile.tile_m == 128 && kFp8Tile.tile_n == 128 &&
             kFp8Tile.tile_k == 64 && kFp8Tile.cluster_m == 2 &&
             kFp8Tile.cluster_n == 1 && kFp8Tile.stages == 5,
         "the fp8 tile is the one the probe measured", kCase);
  Expect(kFp8Tile.tile_m >= 128,
         "a tile narrower than 128 in M has no cooperative schedule", kCase);
  const long long one_stage =
      static_cast<long long>(kFp8Tile.tile_m + kFp8Tile.tile_n) *
      kFp8Tile.tile_k;
  const long long staged = one_stage * kFp8Tile.stages;
  Expect(staged == 81920, "the fp8 pipeline stages 81920 B", kCase);
  Expect(staged < 227LL * 1024,
         "the fp8 pipeline leaves the epilogue its share", kCase);
}

// The seam's default rules, which `ResolveGemm` and `GemmImpl` must share.
void CheckResolve() {
  const char* kCase = "resolve";
  struct Case {
    bool ta;
    bool tb;
    int m;
    int n;
    int k;
  };
  const Case cases[] = {
      {false, false, 3, 4, 5},
      {false, true, 3, 4, 5},
      {true, false, 3, 4, 5},
      {true, true, 3, 4, 5},
  };
  for (const Case& c : cases) {
    GemmParams params;
    params.m = c.m;
    params.n = c.n;
    params.k = c.k;
    params.batch_count = 2;
    params.transpose_a = c.ta;
    params.transpose_b = c.tb;
    const ResolvedGemm gemm = ResolveGemm(params, /*fp32_c=*/false);
    Expect(gemm.lda == (c.ta ? c.m : c.k), "lda follows the transpose", kCase);
    Expect(gemm.ldb == (c.tb ? c.k : c.n), "ldb follows the transpose", kCase);
    Expect(gemm.ldc == c.n, "ldc defaults to n", kCase);
    Expect(
        gemm.stride_a == static_cast<std::int64_t>(c.ta ? c.k : c.m) * gemm.lda,
        "stride_a is the operand row count times its leading dimension", kCase);
    Expect(
        gemm.stride_b == static_cast<std::int64_t>(c.tb ? c.n : c.k) * gemm.ldb,
        "stride_b is the operand row count times its leading dimension", kCase);
    Expect(gemm.stride_c == static_cast<std::int64_t>(c.m) * gemm.ldc,
           "stride_c is the output row count times its leading dimension",
           kCase);
    Expect(IsTileCallable(gemm), "a dense call is callable", kCase);
  }

  // An explicit value wins over the default.
  GemmParams explicit_params;
  explicit_params.m = 3;
  explicit_params.n = 4;
  explicit_params.k = 5;
  explicit_params.lda = 7;
  explicit_params.ldb = 9;
  explicit_params.ldc = 11;
  explicit_params.stride_a = 100;
  explicit_params.stride_b = 200;
  explicit_params.stride_c = 300;
  const ResolvedGemm gemm = ResolveGemm(explicit_params, /*fp32_c=*/true);
  Expect(gemm.lda == 7 && gemm.ldb == 9 && gemm.ldc == 11,
         "an explicit leading dimension wins", kCase);
  Expect(gemm.stride_a == 100 && gemm.stride_b == 200 && gemm.stride_c == 300,
         "an explicit stride wins", kCase);
  Expect(gemm.fp32_c, "the store type is carried", kCase);

  // The scalars are not part of the key.
  ResolvedGemm scaled = gemm;
  scaled.alpha = 0.5f;
  scaled.beta = 0.25f;
  Expect(SameShape(gemm, scaled), "the scalars are not part of the shape",
         kCase);
}

// The order a block walks the tile grid in is a bijection, whatever the grid
// and whatever the group size. A duplicate loses a tile, and a gap leaves one
// at the value C had. A tile the map puts outside the grid leaves a gap, so the
// one count covers both defects.
template <typename PlanT>
void CheckOrder(const char* what) {
  const char* kCase = "tile order";
  const int kGrids[][2] = {{1, 1},  {2, 3},  {3, 2},  {8, 1}, {9, 4},
                           {11, 7}, {16, 3}, {17, 5}, {1, 9}, {5, 1}};
  for (const auto& grid : kGrids) {
    const int grid_m = grid[0];
    const int grid_n = grid[1];
    const int tiles = grid_m * grid_n;
    std::vector<int> seen(static_cast<std::size_t>(tiles), 0);
    for (int index = 0; index < tiles; ++index) {
      int m_tile = 0;
      int n_tile = 0;
      nanochat::kernels::gemm_tile::TileOf<PlanT>(index, grid_m, grid_n,
                                                  &m_tile, &n_tile);
      if (m_tile >= 0 && m_tile < grid_m && n_tile >= 0 && n_tile < grid_n) {
        ++seen[static_cast<std::size_t>(m_tile * grid_n + n_tile)];
      }
    }
    const bool once = std::all_of(seen.begin(), seen.end(),
                                  [](int count) { return count == 1; });
    Expect(once,
           std::string(what) + " covers a " + std::to_string(grid_m) + " by " +
               std::to_string(grid_n) + " grid once",
           kCase);
  }
}

// The seven schedules of the sweep, each with its own order.
void CheckOrders() {
  CheckOrder<nanochat::kernels::gemm_tile::PlanSmall>("the small schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanBase>("the base schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanWide>("the wide schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanKDeep>("the square schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanKDeepSwizzle>(
      "the swizzled schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanKDeeper>(
      "the deeper swizzled schedule");
  CheckOrder<nanochat::kernels::gemm_tile::PlanTall>("the tall schedule");
}

}  // namespace

int main() {
  CheckTableShape();
  CheckTableMatchesTheCallSites();
  CheckTableMatchesTheLinearCalls();
  CheckSelection();
  CheckMode();
  CheckModeFromEnv();
  CheckCutlassVerdict();
  CheckScaledFp8Verdict();
  CheckRuleTable();
  CheckSchedules();
  CheckScheduleFromEnv();
  CheckOrders();
  CheckResolve();

  if (g_failures != 0) {
    std::fprintf(stderr, "dispatch: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("dispatch: ok, %d rows\n", GemmTileShapeCount());
  return 0;
}
