// Precision-independent GPU correctness for the CUDA backend: a float
// reference for the elementwise/norm, attention, and embedding families,
// exercised in both the fp32 and the fp16 build. The fp16 build runs on Pascal
// (sm_61), so this test also covers the fp16 key/value and scatter-add atomic
// fallback and the fp32 statistics / master-state contract.
//
// The device inputs are rounded to ComputeType first and the reference then
// consumes those exact stored values, so the recorded tolerance measures only
// the kernel arithmetic, not the input conversion. Tolerances (relative,
// scaled by (1 + |want|)):
//   fp32: 1e-4   (the device reduces in float, the oracle in double)
//   fp16: 2e-3   (half ulp near 1.0 is ~9.8e-4; the atomic key/value
//                 accumulation in the backward rounds once per contribution.
//                 Observed worst case over these shapes is ~6.8e-4.)
//
// Host code only: it includes no CUDA header and drives the sealed
// nanochat/kernels.h API, so it stays vendor-free. See docs/testing.md and
// docs/precision.md.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"
#include "nanochat/tensor.h"

namespace {

using nanochat::AttentionParams;
using nanochat::ComputeType;
using nanochat::CopyDir;
using nanochat::PointwiseOp;
using nanochat::RmsNormParams;
using nanochat::kernels::Alloc;
using nanochat::kernels::AttentionBackward;
using nanochat::kernels::AttentionForward;
using nanochat::kernels::EmbeddingBackward;
using nanochat::kernels::EmbeddingForward;
using nanochat::kernels::Free;
using nanochat::kernels::Memcpy;
using nanochat::kernels::PointwiseForward;
using nanochat::kernels::RmsNormForward;
using nanochat::kernels::Synchronize;

#if defined(NANOCHAT_PRECISION_FP16)
ComputeType C(float v) { return nanochat::Fp16FromFloat(v); }
float U(ComputeType v) { return nanochat::Fp16ToFloat(v); }
// Recorded fp16 tolerance; see the header comment for the justification.
constexpr double kTol = 2e-3;
#else
ComputeType C(float v) { return v; }
float U(ComputeType v) { return v; }
constexpr double kTol = 1e-4;
#endif

int g_failures = 0;
int g_checks = 0;

void Fail(const char* what, std::size_t index, double got, double want) {
  std::printf("FAIL: %s[%zu] got %.8g want %.8g\n", what, index, got, want);
  ++g_failures;
}

// Compares a device vector against the float reference, printing the worst
// absolute error so the recorded tolerance stays honest.
void CheckVectorClose(const std::vector<float>& got,
                      const std::vector<float>& want, const char* what) {
  if (got.size() != want.size()) {
    std::printf("FAIL: %s size %zu vs %zu\n", what, got.size(), want.size());
    ++g_failures;
    return;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(got[i] - want[i]);
    if (diff > worst) worst = diff;
    if (diff > kTol * (1.0 + std::fabs(want[i]))) {
      Fail(what, i, got[i], want[i]);
      return;
    }
  }
  ++g_checks;
  std::printf("  %-22s max abs error %.3g\n", what, worst);
}

// Device buffer that owns its allocation and converts on upload/download.
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  int count = 0;
  explicit DevBuf(const std::vector<T>& host) : count((int)host.size()) {
    ptr = static_cast<T*>(Alloc(sizeof(T) * host.size()));
    Memcpy(ptr, host.data(), sizeof(T) * host.size(), CopyDir::kHostToDevice);
  }
  explicit DevBuf(int n) : count(n) {
    ptr = static_cast<T*>(Alloc(sizeof(T) * n));
  }
  ~DevBuf() { Free(ptr); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  std::vector<T> Download() const {
    std::vector<T> host(count);
    Memcpy(host.data(), ptr, sizeof(T) * count, CopyDir::kDeviceToHost);
    return host;
  }
};

std::vector<float> FromStorage(const std::vector<ComputeType>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = U(v[i]);
  return out;
}

std::vector<ComputeType> ToStorage(const std::vector<float>& v) {
  std::vector<ComputeType> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = C(v[i]);
  return out;
}

// Deterministic xorshift in [-1, 1]; fixed seed so the fixture is reproducible.
std::uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
float Next() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return static_cast<float>((g_rng >> 40) % 2001) / 1000.0f - 1.0f;
}

std::vector<float> RandomVec(int n) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = Next();
  return v;
}

// ---------------------------------------------------------------------------
// RmsNorm (norm family)
// ---------------------------------------------------------------------------

void TestRmsNorm() {
  RmsNormParams p;
  p.rows = 5;
  p.dim = 7;
  p.eps = 1e-6f;
  const int n = p.rows * p.dim;

  const std::vector<ComputeType> x = ToStorage(RandomVec(n));
  const std::vector<float> xf = FromStorage(x);

  DevBuf<ComputeType> dx(x);
  DevBuf<ComputeType> dout(n);
  DevBuf<float> drstd(p.rows);
  RmsNormForward(p, dx.ptr, dout.ptr, drstd.ptr);
  Synchronize();

  std::vector<float> ref_out(n);
  std::vector<float> ref_rstd(p.rows);
  for (int r = 0; r < p.rows; ++r) {
    double sum_sq = 0.0;
    for (int d = 0; d < p.dim; ++d) {
      const double v = xf[r * p.dim + d];
      sum_sq += v * v;
    }
    const float rstd =
        1.0f / std::sqrt(static_cast<float>(sum_sq / p.dim) + p.eps);
    ref_rstd[r] = rstd;
    for (int d = 0; d < p.dim; ++d) {
      ref_out[r * p.dim + d] = xf[r * p.dim + d] * rstd;
    }
  }

  CheckVectorClose(FromStorage(dout.Download()), ref_out, "rmsnorm out");
  CheckVectorClose(drstd.Download(), ref_rstd, "rmsnorm rstd");
}

// ---------------------------------------------------------------------------
// Pointwise (elementwise family)
// ---------------------------------------------------------------------------

void TestPointwise() {
  const int n = 33;
  const std::vector<ComputeType> a = ToStorage(RandomVec(n));
  const std::vector<ComputeType> b = ToStorage(RandomVec(n));
  const std::vector<float> af = FromStorage(a);
  const std::vector<float> bf = FromStorage(b);

  DevBuf<ComputeType> da(a);
  DevBuf<ComputeType> db(b);
  DevBuf<ComputeType> dout(n);
  PointwiseForward(PointwiseOp::kReluSquare, n, da.ptr, db.ptr, 0.0f, 0.0f,
                   dout.ptr);
  Synchronize();
  std::vector<float> ref(n);
  for (int i = 0; i < n; ++i) {
    const float relu = af[i] > 0.0f ? af[i] : 0.0f;
    ref[i] = relu * relu;
  }
  CheckVectorClose(FromStorage(dout.Download()), ref, "pointwise relu^2");

  const float alpha = 0.5f;
  const float beta = 0.25f;
  PointwiseForward(PointwiseOp::kScaleAdd, n, da.ptr, db.ptr, alpha, beta,
                   dout.ptr);
  Synchronize();
  for (int i = 0; i < n; ++i) ref[i] = alpha * af[i] + beta * bf[i];
  CheckVectorClose(FromStorage(dout.Download()), ref, "pointwise scale-add");
}

// ---------------------------------------------------------------------------
// Embedding (sequence family): duplicate ids force the fp16 scatter-add atomic
// ---------------------------------------------------------------------------

void TestEmbedding() {
  const int tokens = 4;
  const int dim = 5;
  const int vocab = 5;  // rows 3 and 4 are untouched and must keep their value
  const int ids[tokens] = {0, 2, 1, 0};  // id 0 repeats

  const std::vector<ComputeType> table = ToStorage(RandomVec(vocab * dim));
  const std::vector<ComputeType> dout = ToStorage(RandomVec(tokens * dim));
  const std::vector<float> tablef = FromStorage(table);
  const std::vector<float> doutf = FromStorage(dout);

  DevBuf<ComputeType> dtable(table);  // forward value source
  DevBuf<ComputeType> dout_dev(dout);
  DevBuf<ComputeType> out_dev(tokens * dim);
  EmbeddingForward(tokens, dim, ids, dtable.ptr, out_dev.ptr);
  Synchronize();

  std::vector<float> ref_out(tokens * dim);
  for (int t = 0; t < tokens; ++t) {
    for (int d = 0; d < dim; ++d) {
      ref_out[t * dim + d] = tablef[ids[t] * dim + d];
    }
  }
  CheckVectorClose(FromStorage(out_dev.Download()), ref_out, "embedding fwd");

  // Backward: the kernel zeroes the touched rows, then scatter-adds dout. With
  // a repeated id the two contributions must add (the atomic path), and an
  // untouched row keeps its previous value.
  const float kPrev = 7.0f;
  std::vector<ComputeType> grad_init(vocab * dim, C(kPrev));
  DevBuf<ComputeType> dtable_grad(grad_init);
  EmbeddingBackward(tokens, dim, ids, dout_dev.ptr, dtable_grad.ptr);
  Synchronize();

  std::vector<float> ref_grad(vocab * dim, kPrev);
  for (int t = 0; t < tokens; ++t) {
    for (int d = 0; d < dim; ++d) ref_grad[ids[t] * dim + d] = 0.0f;
  }
  for (int t = 0; t < tokens; ++t) {
    for (int d = 0; d < dim; ++d) {
      ref_grad[ids[t] * dim + d] += doutf[t * dim + d];
    }
  }
  CheckVectorClose(FromStorage(dtable_grad.Download()), ref_grad,
                   "embedding bwd");
}

// ---------------------------------------------------------------------------
// Attention (sequence family), including the grouped-query atomic backward
// ---------------------------------------------------------------------------

int KvHead(int h, int num_heads, int num_kv_heads) {
  if (num_kv_heads <= 0 || num_kv_heads >= num_heads) return h;
  const int group = num_heads / num_kv_heads;
  const int kv = h / (group > 0 ? group : 1);
  return kv < num_kv_heads ? kv : num_kv_heads - 1;
}

bool KeyAllowed(const AttentionParams& p, int qpos, int j) {
  if (p.causal && j > qpos) return false;
  if (p.window_left >= 0 && qpos - j > p.window_left) return false;
  if (p.window_right >= 0 && j - qpos > p.window_right) return false;
  return true;
}

void TestAttention() {
  // Tiny grouped-query shape: two query heads share one key/value head, and
  // several causal query rows write the same key/value row, so the backward
  // exercises the fp16 atomic fallback.
  AttentionParams p;
  p.batch = 1;
  p.seq = 3;
  p.num_heads = 2;
  p.num_kv_heads = 1;
  p.head_dim = 4;
  p.causal = true;
  p.window_left = -1;
  p.window_right = 0;
  p.kv_len = 3;
  p.scale = 0.0f;  // default 1 / sqrt(head_dim)
  const float scale = 1.0f / std::sqrt(static_cast<float>(p.head_dim));

  const int kv_heads = p.num_kv_heads > 0 ? p.num_kv_heads : p.num_heads;
  const int q_count = p.batch * p.seq * p.num_heads * p.head_dim;
  const int kv_count = p.batch * p.kv_len * kv_heads * p.head_dim;
  const int stats_count = nanochat::AttentionStatsCount(p);

  const std::vector<ComputeType> q = ToStorage(RandomVec(q_count));
  const std::vector<ComputeType> k = ToStorage(RandomVec(kv_count));
  const std::vector<ComputeType> v = ToStorage(RandomVec(kv_count));
  const std::vector<float> qf = FromStorage(q);
  const std::vector<float> kf = FromStorage(k);
  const std::vector<float> vf = FromStorage(v);

  DevBuf<ComputeType> dq(q);
  DevBuf<ComputeType> dk(k);
  DevBuf<ComputeType> dv(v);
  DevBuf<ComputeType> dout(q_count);
  DevBuf<float> dstats(stats_count);
  AttentionForward(p, dq.ptr, dk.ptr, dv.ptr, dout.ptr, dstats.ptr);
  Synchronize();

  // Forward reference in float over the rounded inputs.
  std::vector<float> ref_out(q_count, 0.0f);
  std::vector<float> ref_stats(stats_count, 0.0f);
  for (int b = 0; b < p.batch; ++b) {
    for (int h = 0; h < p.num_heads; ++h) {
      const int kvh = KvHead(h, p.num_heads, p.num_kv_heads);
      for (int t = 0; t < p.seq; ++t) {
        const int qpos = p.kv_len - p.seq + t;
        const long long qbase =
            ((static_cast<long long>(b) * p.seq + t) * p.num_heads + h) *
            p.head_dim;
        const long long kvbase =
            static_cast<long long>(b) * p.kv_len * kv_heads * p.head_dim;
        const long long sbase =
            ((static_cast<long long>(b) * p.num_heads + h) * p.seq + t) * 2;

        float row_max = -INFINITY;
        for (int j = 0; j < p.kv_len; ++j) {
          if (!KeyAllowed(p, qpos, j)) continue;
          const float* krow =
              kf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* qrow = qf.data() + qbase;
          float dot = 0.0f;
          for (int d = 0; d < p.head_dim; ++d) dot += qrow[d] * krow[d];
          row_max = std::fmax(row_max, scale * dot);
        }
        if (row_max == -INFINITY) continue;
        float sum_exp = 0.0f;
        for (int j = 0; j < p.kv_len; ++j) {
          if (!KeyAllowed(p, qpos, j)) continue;
          const float* krow =
              kf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* qrow = qf.data() + qbase;
          float dot = 0.0f;
          for (int d = 0; d < p.head_dim; ++d) dot += qrow[d] * krow[d];
          sum_exp += std::exp(scale * dot - row_max);
        }
        for (int d = 0; d < p.head_dim; ++d) {
          float acc = 0.0f;
          for (int j = 0; j < p.kv_len; ++j) {
            if (!KeyAllowed(p, qpos, j)) continue;
            const float* krow =
                kf.data() + kvbase +
                (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
            const float* vrow =
                vf.data() + kvbase +
                (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
            const float* qrow = qf.data() + qbase;
            float dot = 0.0f;
            for (int dd = 0; dd < p.head_dim; ++dd) dot += qrow[dd] * krow[dd];
            acc += std::exp(scale * dot - row_max) * vrow[d];
          }
          ref_out[qbase + d] = acc / sum_exp;
        }
        ref_stats[sbase] = row_max;
        ref_stats[sbase + 1] = sum_exp;
      }
    }
  }
  CheckVectorClose(FromStorage(dout.Download()), ref_out, "attention out");
  CheckVectorClose(dstats.Download(), ref_stats, "attention stats");

  // Backward against the reference statistics.
  const std::vector<ComputeType> dout_h = ToStorage(RandomVec(q_count));
  const std::vector<float> doutf = FromStorage(dout_h);
  DevBuf<ComputeType> ddout(dout_h);
  DevBuf<ComputeType> ddq(q_count);
  DevBuf<ComputeType> ddk(kv_count);
  DevBuf<ComputeType> ddv(kv_count);
  DevBuf<float> dstats_ref(ref_stats);
  AttentionBackward(p, dq.ptr, dk.ptr, dv.ptr, dstats_ref.ptr, ddout.ptr,
                    ddq.ptr, ddk.ptr, ddv.ptr);
  Synchronize();

  std::vector<float> ref_dq(q_count, 0.0f);
  std::vector<float> ref_dk(kv_count, 0.0f);
  std::vector<float> ref_dv(kv_count, 0.0f);
  for (int b = 0; b < p.batch; ++b) {
    for (int h = 0; h < p.num_heads; ++h) {
      const int kvh = KvHead(h, p.num_heads, p.num_kv_heads);
      for (int t = 0; t < p.seq; ++t) {
        const int qpos = p.kv_len - p.seq + t;
        const long long qbase =
            ((static_cast<long long>(b) * p.seq + t) * p.num_heads + h) *
            p.head_dim;
        const long long kvbase =
            static_cast<long long>(b) * p.kv_len * kv_heads * p.head_dim;
        const long long sbase =
            ((static_cast<long long>(b) * p.num_heads + h) * p.seq + t) * 2;
        const float row_max = ref_stats[sbase];
        const float sum_exp = ref_stats[sbase + 1];
        if (sum_exp <= 0.0f) continue;
        const float inv = 1.0f / sum_exp;

        float weighted_dp = 0.0f;
        for (int j = 0; j < p.kv_len; ++j) {
          if (!KeyAllowed(p, qpos, j)) continue;
          const float* krow =
              kf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* vrow =
              vf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* qrow = qf.data() + qbase;
          const float* drow = doutf.data() + qbase;
          float dot = 0.0f;
          float dp = 0.0f;
          for (int d = 0; d < p.head_dim; ++d) {
            dot += qrow[d] * krow[d];
            dp += drow[d] * vrow[d];
          }
          const float pj = std::exp(scale * dot - row_max) * inv;
          weighted_dp += pj * dp;
        }
        for (int j = 0; j < p.kv_len; ++j) {
          if (!KeyAllowed(p, qpos, j)) continue;
          const float* krow =
              kf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* vrow =
              vf.data() + kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          const float* qrow = qf.data() + qbase;
          const float* drow = doutf.data() + qbase;
          float dot = 0.0f;
          float dp = 0.0f;
          for (int d = 0; d < p.head_dim; ++d) {
            dot += qrow[d] * krow[d];
            dp += drow[d] * vrow[d];
          }
          const float pj = std::exp(scale * dot - row_max) * inv;
          const float dscale = pj * (dp - weighted_dp) * scale;
          const long long kvoff =
              kvbase +
              (static_cast<long long>(j) * kv_heads + kvh) * p.head_dim;
          for (int d = 0; d < p.head_dim; ++d) {
            ref_dq[qbase + d] += dscale * krow[d];
            ref_dk[kvoff + d] += dscale * qrow[d];
            ref_dv[kvoff + d] += pj * drow[d];
          }
        }
      }
    }
  }
  CheckVectorClose(FromStorage(ddq.Download()), ref_dq, "attention dq");
  CheckVectorClose(FromStorage(ddk.Download()), ref_dk, "attention dk");
  CheckVectorClose(FromStorage(ddv.Download()), ref_dv, "attention dv");
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("test");
  std::printf("precision test (tolerance %.3g)\n", kTol);
  TestRmsNorm();
  TestPointwise();
  TestEmbedding();
  TestAttention();
  if (g_failures != 0) {
    std::printf("precision: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("precision: all %d checks passed\n", g_checks);
  return 0;
}
