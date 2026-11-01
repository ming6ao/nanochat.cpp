// The fused-override gate: one trace comparison per fused family, at the T1
// tiny shapes (docs/model-and-device-bringup.md section 7.4 condition 1 and
// section 7.6).
//
// The composition defines the semantics and the fused family is an
// optimization, so both paths must satisfy the same golden trace within the
// committed tolerance. This test pins that, family by family: it runs the fused
// kernel's forward, its saved slots, and its backward on the device, and it
// runs the composition's arithmetic on the host, and it compares the two traces
// element by element. A family may not be promoted to an override until this
// comparison holds, and a fused family may not be *removed* until the
// composition's backward matches the fused backward's trace -- which is the
// gate section 7.6 requires, written down as a test rather than as an
// intention.
//
// The trace of a family is three things, not one, because a backward that reads
// a saved slot can be right about the gradients while the slot it saved is
// wrong:
//
//   * the forward outputs,
//   * the saved slots the family's kind declares (`block_kinds.md` section 2:
//     `rstd` for `rms_norm`; `q_pre`/`k_pre` plus the tables for `rope`;
//     `ve_gate` for `value_gated_residual`; the pre-absence of a slot for
//     `grad_norm`),
//   * the backward gradients.
//
// The composition side of each comparison is the reference implementation of
// the composition's arithmetic, which is the same code the CPU oracle runs
// (`backends/cuda/kernels/testing/row_ref.h`, `sequence_ref.h`); the value-gate
// composition has no shared reference, so it is written out below from the kind
// definition in `spec/block_kinds.md`.
//
// The second half of this file is the registration cross-check. It reads
// `RegisteredFusedOverrides` and holds the table below and the registrations to
// each other: a registration with no trace comparison here is a family that was
// promoted without its gate, and a trace marked `registered` with no entry in
// the table is a family that lost its registration and left dead code behind.
//
// See docs/testing.md and backends/cuda/fused_overrides.cc.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/kernels/testing/gpu_test_utils.h"
#include "backends/cuda/kernels/testing/row_ref.h"
#include "backends/cuda/kernels/testing/sequence_ref.h"
#include "nanochat/kernels.h"
#include "nanochat/plan.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ClassifierParams;
using nanochat::ComputeType;
using nanochat::FusedOverride;
using nanochat::QkPrepParams;
using nanochat::RmsNormParams;
using nanochat::dev::Check;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

// The tolerance the committed trace uses at the T1 tiny shapes: the fp32
// tolerance of `models/<name>/tolerances.json` is 1e-5 relative on the loss and
// 1e-4 on the parameter L2, and a per-element comparison of a tiny tensor sits
// inside that at 1e-4 (docs/testing.md).
constexpr double kTraceTol = 1e-4;

// ---------------------------------------------------------------------------
// rms_norm: residual add + RmsNorm
// ---------------------------------------------------------------------------
//
// Composition: reduce the row with `RowReduce(kSumSq)`, form
// `rsqrt(mean + eps)` through the transcendental leg, and scale the row. Fused
// family: `backends/cuda/kernels/rms_norm.cu`, which also carries the fused
// residual variant. Saved slot: `rstd`.
void TraceRmsNorm(const char* name) {
  Rng rng;
  for (const int dim : {8, 64}) {
    RmsNormParams p;
    p.rows = 3;
    p.dim = dim;
    p.eps = nanochat::kDefaultRmsNormEps;
    const int count = p.rows * p.dim;
    const std::vector<float> x = RandomVec(count, &rng);
    const std::vector<float> dy = RandomVec(count, &rng);

    DevBuf<ComputeType> x_dev(ToStorage(x));
    DevBuf<ComputeType> dy_dev(ToStorage(dy));
    DevBuf<ComputeType> out_dev(count);
    DevBuf<ComputeType> dx_dev(count);
    DevBuf<float> rstd_dev(p.rows);

    nanochat::kernels::RmsNormForward(p, x_dev.ptr, out_dev.ptr, rstd_dev.ptr);
    nanochat::kernels::RmsNormBackward(p, x_dev.ptr, dy_dev.ptr, rstd_dev.ptr,
                                       dx_dev.ptr);
    nanochat::kernels::Synchronize();

    std::vector<float> ref_rstd;
    const std::vector<float> ref_out =
        nanochat::dev::rowref::RmsNormForward(p, x, &ref_rstd);
    const std::vector<float> ref_dx =
        nanochat::dev::rowref::RmsNormBackward(p, x, dy, ref_rstd);

    const std::string base = std::string(name) + " dim " + std::to_string(dim);
    CheckVectorClose(FromStorage(out_dev.Download()), ref_out, kTraceTol,
                     (base + " fwd").c_str());
    CheckVectorClose(rstd_dev.Download(), ref_rstd, kTraceTol,
                     (base + " rstd slot").c_str());
    CheckVectorClose(FromStorage(dx_dev.Download()), ref_dx, kTraceTol,
                     (base + " bwd").c_str());
  }
}

// ---------------------------------------------------------------------------
// qk_prep: norm on head_dim + rope + scale
// ---------------------------------------------------------------------------
//
// Composition: `RowReduce(kSumSq)` over the `head_dim` slice, the norm scale,
// then the rotate-pair arithmetic of the `rope` kind and the qk scale. Fused
// family: `backends/cuda/kernels/qk_prep.cu`. Saved slots: `q_pre`, `k_pre`
// (the backward recovers the statistics from the saved rows) and the cos/sin
// tables, which are device memory.
//
// This is the reference case DESIGN.md section 3.1 names: the fused pair must
// carry the same golden trace as the composition.
void TraceQkPrep(const char* name) {
  Rng rng;
  QkPrepParams p;
  p.batch = 2;
  p.seq = 3;
  p.num_heads = 2;
  p.num_kv_heads = 1;  // grouped: fewer key/value heads than query heads
  p.head_dim = 8;
  p.eps = nanochat::kDefaultRmsNormEps;
  p.scale = nanochat::kDefaultQkScale;

  const int half = p.head_dim / 2;
  std::vector<float> cos(static_cast<std::size_t>(p.seq) * half, 0.0f);
  std::vector<float> sin(static_cast<std::size_t>(p.seq) * half, 0.0f);
  for (int t = 0; t < p.seq; ++t) {
    for (int d = 0; d < half; ++d) {
      const float theta =
          0.1f * static_cast<float>(t + 1) * static_cast<float>(d + 1);
      cos[static_cast<std::size_t>(t) * half + d] = std::cos(theta);
      sin[static_cast<std::size_t>(t) * half + d] = std::sin(theta);
    }
  }
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int k_count = p.batch * p.seq * p.num_kv_heads * p.head_dim;
  const std::vector<float> q0 = RandomVec(q_count, &rng);
  const std::vector<float> k0 = RandomVec(k_count, &rng);
  const std::vector<float> dq0 = RandomVec(q_count, &rng);
  const std::vector<float> dk0 = RandomVec(k_count, &rng);

  DevBuf<float> cos_dev(cos);
  DevBuf<float> sin_dev(sin);
  DevBuf<ComputeType> q_dev(ToStorage(q0));
  DevBuf<ComputeType> k_dev(ToStorage(k0));
  nanochat::kernels::QkPrepForward(p, cos_dev.ptr, sin_dev.ptr, q_dev.ptr,
                                   k_dev.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_q = q0;
  std::vector<float> ref_k = k0;
  nanochat::dev::rowref::QkPrepForward(p, cos, sin, &ref_q, &ref_k);
  CheckVectorClose(FromStorage(q_dev.Download()), ref_q, kTraceTol,
                   (std::string(name) + " fwd q").c_str());
  CheckVectorClose(FromStorage(k_dev.Download()), ref_k, kTraceTol,
                   (std::string(name) + " fwd k").c_str());

  // The backward reads the saved pre-norm rows and overwrites them with the
  // input gradients.
  DevBuf<ComputeType> dq_dev(ToStorage(dq0));
  DevBuf<ComputeType> dk_dev(ToStorage(dk0));
  DevBuf<ComputeType> q_saved(ToStorage(q0));
  DevBuf<ComputeType> k_saved(ToStorage(k0));
  nanochat::kernels::QkPrepBackward(p, cos_dev.ptr, sin_dev.ptr, dq_dev.ptr,
                                    dk_dev.ptr, q_saved.ptr, k_saved.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_dq = q0;
  std::vector<float> ref_dk = k0;
  nanochat::dev::rowref::QkPrepBackward(p, cos, sin, dq0, dk0, &ref_dq,
                                        &ref_dk);
  CheckVectorClose(FromStorage(q_saved.Download()), ref_dq, kTraceTol,
                   (std::string(name) + " bwd dq").c_str());
  CheckVectorClose(FromStorage(k_saved.Download()), ref_dk, kTraceTol,
                   (std::string(name) + " bwd dk").c_str());
}

// ---------------------------------------------------------------------------
// classifier: logit softcap + cross-entropy + the vocab slice
// ---------------------------------------------------------------------------
//
// Composition: `Pointwise(kSoftcap)` then `RowReduce(kLogSumExp)` over the
// `vocab_size` slice with the padded tail excluded, and the backward's
// `(p - onehot) * d(softcap)/d(logit)` chain. Fused family:
// `backends/cuda/kernels/classifier.cu`. The contract this family adds to the
// trace is the zeros: the backward zeroes every ignored row and the padded
// vocabulary tail, and a tolerance on the loss does not cover a wrong zero.
void TraceClassifier(const char* name) {
  Rng rng;
  ClassifierParams p;
  p.rows = 4;
  p.vocab_size = 10;
  p.padded_vocab_size = 16;
  p.softcap = nanochat::kDefaultLogitSoftcap;
  p.ignore_index = -1;
  // The per-row weights are read by the *device* kernel, so the pointer the
  // backend receives is device memory; the host reference reads the same values
  // from the host copy. `src/model.cc` stages them the same way, so the two
  // params blocks below differ only in the memory the pointer names.
  const std::vector<float> row_scale = {1.0f, 0.5f, 2.0f, 1.0f};
  DevBuf<float> row_scale_dev(row_scale);
  ClassifierParams device_params = p;
  device_params.row_scale = row_scale_dev.ptr;
  ClassifierParams host_params = p;
  host_params.row_scale = row_scale.data();

  // The padded tail holds values that would change the loss if the kernel read
  // them, so a leak is visible rather than approximately right.
  std::vector<float> logits = RandomVec(p.rows * p.padded_vocab_size, &rng);
  for (int r = 0; r < p.rows; ++r) {
    for (int j = p.vocab_size; j < p.padded_vocab_size; ++j) {
      logits[static_cast<std::size_t>(r) * p.padded_vocab_size + j] =
          100.0f + static_cast<float>(j);
    }
  }
  std::vector<int> targets(p.rows);
  for (int r = 0; r < p.rows; ++r) targets[r] = r % p.vocab_size;
  targets[2] = p.ignore_index;  // one ignored row, whose gradient must be zero

  DevBuf<ComputeType> logits_dev(ToStorage(logits));
  DevBuf<float> losses_dev(p.rows);
  nanochat::kernels::ClassifierForward(device_params, logits_dev.ptr,
                                       targets.data(), losses_dev.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> ref_losses =
      nanochat::dev::seqref::ClassifierForward(host_params, logits, targets);
  CheckVectorClose(losses_dev.Download(), ref_losses, kTraceTol,
                   (std::string(name) + " fwd").c_str());

  DevBuf<ComputeType> dlogits_dev(p.rows * p.padded_vocab_size);
  nanochat::kernels::ClassifierBackward(device_params, logits_dev.ptr,
                                        targets.data(), dlogits_dev.ptr);
  nanochat::kernels::Synchronize();

  const std::vector<float> ref_dlogits =
      nanochat::dev::seqref::ClassifierBackward(host_params, logits, targets);
  CheckVectorClose(FromStorage(dlogits_dev.Download()), ref_dlogits, kTraceTol,
                   (std::string(name) + " bwd").c_str());
}

// ---------------------------------------------------------------------------
// value_gate: the ResFormer gate on the value rows
// ---------------------------------------------------------------------------
//
// Composition (spec/block_kinds.md, `value_gated_residual`): the dot of the
// leading `G` channels of the hidden row with the gate weights, `kGateMul` for
// the sigmoid, `kScale` for the gate factor `m`, and the per-head broadcast of
// the gate over the value rows. Fused family:
// `backends/cuda/kernels/value_gate.cu`. Saved slot: `ve_gate`, and the
// backward reads it rather than recomputing it, so a gate saved under a
// different sigmoid approximation fails even when the `v` rows are right.
//
// This family has no shared host reference, so the composition's arithmetic is
// written out here from the kind definition. `G = 12` is `kVeGateChannels`,
// which the kind names as a constant of the release.
constexpr int kVeGateChannels = 12;

void RefValueGateForward(int rows, int hidden, int kv_heads, int head_dim,
                         const std::vector<float>& h,
                         const std::vector<float>& ve,
                         const std::vector<float>& gate_w,
                         std::vector<float>* v, std::vector<float>* gate) {
  for (int r = 0; r < rows; ++r) {
    for (int kh = 0; kh < kv_heads; ++kh) {
      double pre = 0.0;
      for (int j = 0; j < kVeGateChannels; ++j) {
        pre +=
            static_cast<double>(h[static_cast<std::size_t>(r) * hidden + j]) *
            gate_w[static_cast<std::size_t>(kh) * kVeGateChannels + j];
      }
      const float g = 3.0f / (1.0f + std::exp(-static_cast<float>(pre)));
      (*gate)[static_cast<std::size_t>(r) * kv_heads + kh] = g;
      for (int d = 0; d < head_dim; ++d) {
        const std::size_t off =
            (static_cast<std::size_t>(r) * kv_heads + kh) * head_dim + d;
        (*v)[off] += g * ve[off];
      }
    }
  }
}

void RefValueGateBackward(int rows, int hidden, int kv_heads, int head_dim,
                          const std::vector<float>& h,
                          const std::vector<float>& ve,
                          const std::vector<float>& gate_w,
                          const std::vector<float>& gate,
                          const std::vector<float>& dv,
                          std::vector<float>* gate_w_grad,
                          std::vector<float>* dh, std::vector<float>* dve) {
  for (int r = 0; r < rows; ++r) {
    for (int kh = 0; kh < kv_heads; ++kh) {
      const float g = gate[static_cast<std::size_t>(r) * kv_heads + kh];
      double dgate = 0.0;
      for (int d = 0; d < head_dim; ++d) {
        const std::size_t off =
            (static_cast<std::size_t>(r) * kv_heads + kh) * head_dim + d;
        dgate += static_cast<double>(dv[off]) * ve[off];
        (*dve)[off] = g * dv[off];
      }
      // gate = m * sigmoid(pre) with m = 3, so d(gate)/d(pre) = gate * (1 -
      // gate/3).
      const float dpre = static_cast<float>(dgate) * g * (1.0f - g / 3.0f);
      for (int j = 0; j < kVeGateChannels; ++j) {
        (*gate_w_grad)[static_cast<std::size_t>(kh) * kVeGateChannels + j] +=
            dpre * h[static_cast<std::size_t>(r) * hidden + j];
        (*dh)[static_cast<std::size_t>(r) * hidden + j] +=
            dpre * gate_w[static_cast<std::size_t>(kh) * kVeGateChannels + j];
      }
    }
  }
}

void TraceValueGate(const char* name) {
  Rng rng;
  const int rows = 2;
  const int hidden = 16;  // wider than the gate; only the first 12 feed it
  const int kv_heads = 2;
  const int head_dim = 4;
  const int v_count = rows * kv_heads * head_dim;

  const std::vector<float> h = RandomVec(rows * hidden, &rng);
  const std::vector<float> ve = RandomVec(v_count, &rng);
  const std::vector<float> gate_w = RandomVec(kv_heads * kVeGateChannels, &rng);
  const std::vector<float> v0 = RandomVec(v_count, &rng);
  const std::vector<float> dv = RandomVec(v_count, &rng);

  DevBuf<ComputeType> h_dev(ToStorage(h));
  DevBuf<ComputeType> ve_dev(ToStorage(ve));
  DevBuf<ComputeType> gate_w_dev(ToStorage(gate_w));
  DevBuf<ComputeType> v_dev(ToStorage(v0));
  DevBuf<ComputeType> gate_dev(rows * kv_heads);
  nanochat::kernels::ValueGateForward(rows, hidden, kv_heads, head_dim,
                                      h_dev.ptr, ve_dev.ptr, gate_w_dev.ptr,
                                      v_dev.ptr, gate_dev.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_v = v0;
  std::vector<float> ref_gate(static_cast<std::size_t>(rows) * kv_heads, 0.0f);
  RefValueGateForward(rows, hidden, kv_heads, head_dim, h, ve, gate_w, &ref_v,
                      &ref_gate);
  CheckVectorClose(FromStorage(v_dev.Download()), ref_v, kTraceTol,
                   (std::string(name) + " fwd v").c_str());
  CheckVectorClose(gate_dev.Download(), ref_gate, kTraceTol,
                   (std::string(name) + " ve_gate slot").c_str());

  // The saved gate is what the backward reads, so the backward runs against the
  // device's own gate, not against the reference one.
  const std::vector<float> gate_seen = gate_dev.Download();
  DevBuf<ComputeType> dv_dev(ToStorage(dv));
  DevBuf<float> gate_w_grad_dev(kv_heads * kVeGateChannels);
  DevBuf<ComputeType> dh_dev(rows * hidden);
  DevBuf<ComputeType> dve_dev(v_count);
  std::vector<float> gate_w_grad_zero(kv_heads * kVeGateChannels, 0.0f);
  std::vector<float> dh_zero(rows * hidden, 0.0f);
  gate_w_grad_dev.Upload(gate_w_grad_zero);
  dh_dev.Upload(ToStorage(dh_zero));
  nanochat::kernels::ValueGateBackward(
      rows, hidden, kv_heads, head_dim, h_dev.ptr, ve_dev.ptr, gate_w_dev.ptr,
      gate_dev.ptr, dv_dev.ptr, gate_w_grad_dev.ptr, dh_dev.ptr, dve_dev.ptr);
  nanochat::kernels::Synchronize();

  std::vector<float> ref_gate_w_grad = gate_w_grad_zero;
  std::vector<float> ref_dh = dh_zero;
  std::vector<float> ref_dve(v_count, 0.0f);
  RefValueGateBackward(rows, hidden, kv_heads, head_dim, h, ve, gate_w,
                       gate_seen, dv, &ref_gate_w_grad, &ref_dh, &ref_dve);
  CheckVectorClose(gate_w_grad_dev.Download(), ref_gate_w_grad, kTraceTol,
                   (std::string(name) + " bwd gate_w").c_str());
  CheckVectorClose(FromStorage(dh_dev.Download()), ref_dh, kTraceTol,
                   (std::string(name) + " bwd dh").c_str());
  CheckVectorClose(FromStorage(dve_dev.Download()), ref_dve, kTraceTol,
                   (std::string(name) + " bwd dve").c_str());
  // `v` and `dv` are inputs to the backward and must come back unchanged.
  CheckVectorClose(FromStorage(v_dev.Download()), ref_v, 1e-6,
                   (std::string(name) + " v untouched").c_str());
}

// ---------------------------------------------------------------------------
// grad_norm: the L2 norm across several spans
// ---------------------------------------------------------------------------
//
// Composition: `SpanReduce` across several spans. There is no fused *override*
// bit for it in vocabulary 1 (`spec/capability_bits.md` section 5), so nothing
// is registered -- but the trace comparison lives here all the same, so that
// the day the family is removed the composition's answer is already the pinned
// one.
void TraceGradNorm(const char* name) {
  Rng rng;
  const std::vector<std::vector<float>> spans = {
      RandomVec(3, &rng),
      RandomVec(7, &rng),
      RandomVec(1, &rng),
  };
  const float clip = 0.05f;  // below the norm, so the scale path is exercised

  std::vector<DevBuf<ComputeType>*> buffers;
  std::vector<const ComputeType*> pointers;
  std::vector<int> counts;
  for (const std::vector<float>& span : spans) {
    auto* buffer = new DevBuf<ComputeType>(ToStorage(span));
    buffers.push_back(buffer);
    pointers.push_back(buffer->ptr);
    counts.push_back(static_cast<int>(span.size()));
  }
  DevBuf<const ComputeType*> table(pointers);
  DevBuf<int> count_table(counts);
  DevBuf<float> norm_dev(1);
  DevBuf<float> scale_dev(1);
  nanochat::kernels::GradNorm(table.ptr, count_table.ptr,
                              static_cast<int>(spans.size()), clip,
                              norm_dev.ptr, scale_dev.ptr);
  nanochat::kernels::Synchronize();

  float ref_scale = 1.0f;
  const float ref_norm =
      nanochat::dev::rowref::GradNorm(spans, clip, &ref_scale);
  nanochat::dev::CheckClose(norm_dev.Download()[0], ref_norm, kTraceTol,
                            (std::string(name) + " fwd norm").c_str());
  nanochat::dev::CheckClose(scale_dev.Download()[0], ref_scale, kTraceTol,
                            (std::string(name) + " scale").c_str());
  // No span is rewritten: the caller folds the scale into its own update.
  for (std::size_t s = 0; s < spans.size(); ++s) {
    CheckVectorClose(buffers[s]->Download(), spans[s], 1e-6,
                     (std::string(name) + " span untouched").c_str());
    delete buffers[s];
  }
}

// ---------------------------------------------------------------------------
// The coverage table
// ---------------------------------------------------------------------------

// One row per fused family this backend implements.
//
//   `family`       the name the registration carries and the plan prints
//   `composition`  the `CompositionName` spelling it replaces
//   `registered`   true when `RegisteredFusedOverrides` must carry it
//   `note`         why an unregistered family is unregistered, so the follow-up
//                  is a one-line change and not an investigation
struct TraceCoverage {
  const char* family;
  const char* composition;
  bool registered;
  const char* note;
};

constexpr TraceCoverage kCoverage[] = {
    {"rms_norm", "rms_norm", true, "gated by override_rms_norm"},
    {"qk_prep", "qk_prep", true, "gated by override_qk_prep"},
    {"classifier", "classifier", true, "gated by override_classifier"},
    {"value_gate", "value_gate", false,
     "no constant for override_value_gate (vocabulary bit 15) in "
     "include/nanochat/tensor.h; the registration lands with that constant"},
    {"grad_norm", "grad_norm", false,
     "vocabulary 1 has no override_grad_norm bit (spec/capability_bits.md "
     "section 5); no registration is legal"},
};

constexpr int kCoverageCount =
    static_cast<int>(sizeof(kCoverage) / sizeof(kCoverage[0]));

const TraceCoverage* FindCoverage(const char* family) {
  for (int i = 0; i < kCoverageCount; ++i) {
    if (std::string(kCoverage[i].family) == family) return &kCoverage[i];
  }
  return nullptr;
}

// The registrations and the coverage table describe one set of families from
// two sides. Neither may move alone.
void CheckRegistrationAgreesWithCoverage() {
  int count = 0;
  const FusedOverride* registrations =
      nanochat::RegisteredFusedOverrides(&count);
  Check(registrations != nullptr, "the backend registers fused overrides");
  if (registrations == nullptr) return;

  int expected = 0;
  for (int i = 0; i < kCoverageCount; ++i) {
    if (kCoverage[i].registered) ++expected;
  }
  Check(count == expected,
        "the registration count matches the trace coverage table");
  std::printf("  %d registration(s), %d trace comparison(s)\n", count,
              kCoverageCount);

  for (int i = 0; i < count; ++i) {
    const FusedOverride& entry = registrations[i];
    Check(entry.composition != nullptr && *entry.composition != '\0',
          "a registration names a composition");
    Check(entry.capability != 0, "a registration carries a capability bit");
    const TraceCoverage* coverage = FindCoverage(entry.family);
    if (coverage == nullptr) {
      std::printf(
          "FAIL: the family \"%s\" is registered with no trace comparison in "
          "this test, so it was promoted without its gate\n",
          entry.family);
      ++Failures();
      continue;
    }
    Check(coverage->registered,
          "a registered family is marked registered in the coverage table");
    Check(std::string(coverage->composition) == entry.composition,
          "the registration and the coverage table agree on the composition");
  }

  for (int i = 0; i < kCoverageCount; ++i) {
    if (!kCoverage[i].registered) continue;
    bool found = false;
    for (int j = 0; j < count; ++j) {
      if (std::string(registrations[j].family) == kCoverage[i].family) {
        found = true;
      }
    }
    if (!found) {
      std::printf(
          "FAIL: the family \"%s\" is marked registered but has no entry\n",
          kCoverage[i].family);
      ++Failures();
    }
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("fused override trace test (T1)\n");

  CheckRegistrationAgreesWithCoverage();

  for (int i = 0; i < kCoverageCount; ++i) {
    const TraceCoverage& coverage = kCoverage[i];
    std::printf(
        "%s (%s, %s)\n", coverage.family, coverage.composition,
        coverage.registered ? "override registered" : "composition only");
    if (!coverage.registered) {
      std::printf("  unregistered: %s\n", coverage.note);
    }
    const std::string family = coverage.family;
    if (family == "rms_norm") {
      TraceRmsNorm(coverage.family);
    } else if (family == "qk_prep") {
      TraceQkPrep(coverage.family);
    } else if (family == "classifier") {
      TraceClassifier(coverage.family);
    } else if (family == "value_gate") {
      TraceValueGate(coverage.family);
    } else if (family == "grad_norm") {
      TraceGradNorm(coverage.family);
    } else {
      std::printf("FAIL: no trace comparison for the family \"%s\"\n",
                  coverage.family);
      ++Failures();
    }
  }

  if (Failures() != 0) {
    std::printf("fused override trace: %d failure(s)\n", Failures());
    return 1;
  }
  std::printf("fused override trace: ok\n");
  return 0;
}
