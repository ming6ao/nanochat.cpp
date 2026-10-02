// Batch-1 decode baseline: a small nanochat-style transformer decode step built
// from the seam GEMM (cuBLAS) and kernel families, measured twice:
//
//   1. eager seam path  — `kernels::Gemm` and the seam kernels on the legacy
//      default stream, which is the current model behaviour; and
//   2. cuBLAS + CUDA Graphs — the same step captured on a private stream, using
//      direct cuBLAS on that stream and the capture-safe local kernel mirrors
//      in decode_kernels.cu.
//
// This is the baseline a fused persistent decode megakernel would have to beat
// (DESIGN.md section 4.3). The MMA/persistent variant is explicitly a Turing+
// (sm_75) experiment: this host is a GTX 1080 Ti at sm_61, which has no
// `mma`/`wmma`, so only the baseline is reported here.
//
// Because backends/cuda hardcodes the legacy default stream, the seam kernels
// are *not* CUDA-graph capturable; the graph path therefore uses the local
// stream-taking mirrors. The two paths are cross-checked for numerical
// agreement. Run it through the broker:
//
//   tools/nanochat bench -- <decode_fused_bench binary>
//
// See dev/kernels/README.md and DESIGN.md section 4.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "dev/kernels/decode_kernels.h"
#include "gpu_test_utils.h"
#include "nanochat/kernels.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::ComputeType;
using nanochat::GemmMode;
using nanochat::GemmParams;
using nanochat::dev::CheckVectorClose;
using nanochat::dev::DevBuf;
using nanochat::dev::Failures;
using nanochat::dev::FromStorage;
using nanochat::dev::RandomVec;
using nanochat::dev::Rng;
using nanochat::dev::ToStorage;

struct Config {
  int layers = 6;
  int d_model = 384;
  int heads = 6;
  int head_dim = 64;
  int kv_heads = 6;
  int d_ff = 1536;
  int vocab = 4096;
  int seq_len = 128;  // KV-cache length the decode step attends over
};

#if defined(NANOCHAT_PRECISION_FP16)
constexpr cudaDataType_t kGemmType = CUDA_R_16F;
#else
constexpr cudaDataType_t kGemmType = CUDA_R_32F;
#endif

// Row-major C = op(A) * op(B), mirroring backends/cuda/gemm.cu but on the
// caller's handle/stream so the operation is capturable.
void DevGemm(cublasHandle_t handle, const GemmParams& p, const ComputeType* a,
             const ComputeType* b, ComputeType* c) {
  if (p.batch_count <= 0 || p.m <= 0 || p.n <= 0 || p.k < 0) return;
  const bool ta = p.transpose_a;
  const bool tb = p.transpose_b;
  const int lda = p.lda > 0 ? p.lda : (ta ? p.m : p.k);
  const int ldb = p.ldb > 0 ? p.ldb : (tb ? p.k : p.n);
  const int ldc = p.ldc > 0 ? p.ldc : p.n;
  const std::int64_t stride_a =
      p.stride_a != 0 ? p.stride_a
                      : static_cast<std::int64_t>(ta ? p.k : p.m) * lda;
  const std::int64_t stride_b =
      p.stride_b != 0 ? p.stride_b
                      : static_cast<std::int64_t>(tb ? p.n : p.k) * ldb;
  const std::int64_t stride_c =
      p.stride_c != 0 ? p.stride_c : static_cast<std::int64_t>(p.m) * ldc;
  const cublasOperation_t transa = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  const float alpha = p.alpha;
  const float beta = p.beta;
  cublasGemmStridedBatchedEx(handle, transa, transb, /*m=*/p.n, /*n=*/p.m,
                             /*k=*/p.k, &alpha, b, kGemmType, ldb, stride_b, a,
                             kGemmType, lda, stride_a, &beta, c, kGemmType, ldc,
                             stride_c, p.batch_count, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
}

GemmParams RowGemm(int n, int k) {
  GemmParams p;
  p.m = 1;
  p.n = n;
  p.k = k;
  p.transpose_b = true;
  return p;
}

// Deterministic pseudo-random weights for one flat buffer.
DevBuf<ComputeType> MakeWeights(int n) {
  Rng rng;
  rng.s = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(n);
  return DevBuf<ComputeType>(ToStorage(RandomVec(n, &rng)));
}

// All weights and activations for the decode step. One flat buffer per weight
// family; the per-layer slice is `layer_stride * l`.
struct DecodeModel {
  Config cfg;
  int qdim;
  int kvdim;
  int qkv_dim;
  int mlp_dim;
  int qkv_stride;
  int wo_stride;
  int fc_stride;
  int proj_stride;
  int cache_stride;

  // Weights.
  DevBuf<ComputeType> emb;
  DevBuf<ComputeType> wqkv;
  DevBuf<ComputeType> wo;
  DevBuf<ComputeType> wfc;
  DevBuf<ComputeType> wproj;
  DevBuf<ComputeType> wlm;

  // KV cache: [layers, seq_len, kv_heads, head_dim].
  DevBuf<ComputeType> k_cache;
  DevBuf<ComputeType> v_cache;

  // Rope tables: [seq_len, head_dim/2].
  DevBuf<float> cos;
  DevBuf<float> sin;

  // Activations.
  DevBuf<ComputeType> x;
  DevBuf<ComputeType> xn;
  DevBuf<ComputeType> qkv;
  DevBuf<ComputeType> attn;
  DevBuf<ComputeType> attnout;
  DevBuf<ComputeType> pre;
  DevBuf<ComputeType> act;
  DevBuf<ComputeType> mlpout;
  DevBuf<ComputeType> logits;
  DevBuf<float> rstd;
  DevBuf<float> stats;
  DevBuf<int> token_dev;

  explicit DecodeModel(const Config& c)
      : cfg(c),
        qdim(c.heads * c.head_dim),
        kvdim(c.kv_heads * c.head_dim),
        qkv_dim(c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim),
        mlp_dim(4 * c.d_model),
        qkv_stride(0),
        wo_stride(0),
        fc_stride(0),
        proj_stride(0),
        cache_stride(c.seq_len * c.kv_heads * c.head_dim),
        emb(MakeWeights(c.vocab * c.d_model)),
        wqkv(MakeWeights(c.layers * qkv_dim * c.d_model)),
        wo(MakeWeights(c.layers * c.d_model * qdim)),
        wfc(MakeWeights(c.layers * mlp_dim * c.d_model)),
        wproj(MakeWeights(c.layers * c.d_model * mlp_dim)),
        wlm(MakeWeights(c.vocab * c.d_model)),
        k_cache(c.layers * cache_stride),
        v_cache(c.layers * cache_stride),
        cos(static_cast<int>(static_cast<std::size_t>(c.seq_len) *
                             (c.head_dim / 2))),
        sin(static_cast<int>(static_cast<std::size_t>(c.seq_len) *
                             (c.head_dim / 2))),
        x(c.d_model),
        xn(c.d_model),
        qkv(qkv_dim),
        attn(qdim),
        attnout(c.d_model),
        pre(mlp_dim),
        act(mlp_dim),
        mlpout(c.d_model),
        logits(c.vocab),
        rstd(1),
        stats(2 * c.heads),
        token_dev(1) {
    qkv_stride = qkv_dim * c.d_model;
    wo_stride = c.d_model * qdim;
    fc_stride = mlp_dim * c.d_model;
    proj_stride = c.d_model * mlp_dim;
  }

  const ComputeType* LayerWqkv(int l) const {
    return wqkv.ptr + static_cast<std::size_t>(l) * qkv_stride;
  }
  const ComputeType* LayerWo(int l) const {
    return wo.ptr + static_cast<std::size_t>(l) * wo_stride;
  }
  const ComputeType* LayerWfc(int l) const {
    return wfc.ptr + static_cast<std::size_t>(l) * fc_stride;
  }
  const ComputeType* LayerWproj(int l) const {
    return wproj.ptr + static_cast<std::size_t>(l) * proj_stride;
  }
  ComputeType* LayerKCache(int l) {
    return k_cache.ptr + static_cast<std::size_t>(l) * cache_stride;
  }
  ComputeType* LayerVCache(int l) {
    return v_cache.ptr + static_cast<std::size_t>(l) * cache_stride;
  }
};

void FillRope(DecodeModel* m) {
  const int half = m->cfg.head_dim / 2;
  std::vector<float> cos(static_cast<std::size_t>(m->cfg.seq_len) * half);
  std::vector<float> sin(cos.size());
  for (int t = 0; t < m->cfg.seq_len; ++t) {
    for (int d = 0; d < half; ++d) {
      const float theta =
          0.01f * static_cast<float>(t + 1) / static_cast<float>(d + 1);
      cos[static_cast<std::size_t>(t) * half + d] = std::cos(theta);
      sin[static_cast<std::size_t>(t) * half + d] = std::sin(theta);
    }
  }
  m->cos.Upload(cos);
  m->sin.Upload(sin);
}

nanochat::QkPrepParams QkPrepParamsFor(const Config& c) {
  nanochat::QkPrepParams p;
  p.batch = 1;
  p.seq = 1;  // decode: one new token per step
  p.num_heads = c.heads;
  p.num_kv_heads = c.kv_heads;
  p.head_dim = c.head_dim;
  p.eps = 1e-6f;
  p.scale = 1.2f;
  return p;
}

nanochat::AttentionParams AttentionParamsFor(const Config& c, int kv_len) {
  nanochat::AttentionParams p;
  p.batch = 1;
  p.seq = 1;
  p.num_heads = c.heads;
  p.num_kv_heads = c.kv_heads;
  p.head_dim = c.head_dim;
  p.causal = true;
  p.window_left = -1;
  p.window_right = 0;
  p.kv_len = kv_len;
  p.scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  return p;
}

// Shared step body, parameterised on how a GEMM is issued and whether the
// non-GEMM ops are the seam kernels or the local capture-safe mirrors.
template <typename GemmFn, typename NormFn, typename QkPrepFn,
          typename AttnFn, typename PointFn, typename EmbedFn,
          typename CopyFn>
void RunDecodeStep(DecodeModel* m, const GemmFn& gemm, const NormFn& norm,
                   const QkPrepFn& qkprep, const AttnFn& attn,
                   const PointFn& point, const EmbedFn& embed,
                   const CopyFn& copy, int token_host, int pos) {
  const Config& c = m->cfg;

  embed(token_host, m->x.ptr);

  for (int l = 0; l < c.layers; ++l) {
    // Attention block.
    norm(m->x.ptr, m->xn.ptr);
    gemm(RowGemm(m->qkv_dim, c.d_model), m->xn.ptr, m->LayerWqkv(l),
         m->qkv.ptr);
    ComputeType* q = m->qkv.ptr;
    ComputeType* k = m->qkv.ptr + m->qdim;
    ComputeType* v = m->qkv.ptr + m->qdim + m->kvdim;
    qkprep(q, k);
    ComputeType* ck = m->LayerKCache(l);
    ComputeType* cv = m->LayerVCache(l);
    copy(ck + static_cast<std::size_t>(pos) * m->kvdim, k, m->kvdim);
    copy(cv + static_cast<std::size_t>(pos) * m->kvdim, v, m->kvdim);
    attn(pos + 1, q, ck, cv, m->attn.ptr);
    gemm(RowGemm(c.d_model, m->qdim), m->attn.ptr, m->LayerWo(l),
         m->attnout.ptr);
    point(nanochat::PointwiseOp::kScaleAdd, c.d_model, m->x.ptr,
          m->attnout.ptr, 1.0f, 1.0f, m->x.ptr);

    // MLP block.
    norm(m->x.ptr, m->xn.ptr);
    gemm(RowGemm(m->mlp_dim, c.d_model), m->xn.ptr, m->LayerWfc(l),
         m->pre.ptr);
    point(nanochat::PointwiseOp::kReluSquare, m->mlp_dim, m->pre.ptr, nullptr,
          0.0f, 0.0f, m->act.ptr);
    gemm(RowGemm(c.d_model, m->mlp_dim), m->act.ptr, m->LayerWproj(l),
         m->mlpout.ptr);
    point(nanochat::PointwiseOp::kScaleAdd, c.d_model, m->x.ptr, m->mlpout.ptr,
          1.0f, 1.0f, m->x.ptr);
  }

  // Final norm and language-model head (stop at logits; sampling is row-local).
  norm(m->x.ptr, m->xn.ptr);
  gemm(RowGemm(c.vocab, c.d_model), m->xn.ptr, m->wlm.ptr, m->logits.ptr);
}

int RunEagerSeam(DecodeModel* m, int token_host, int pos) {
  const Config& c = m->cfg;
  nanochat::RmsNormParams rms;
  rms.rows = 1;
  rms.dim = c.d_model;
  rms.eps = 1e-6f;
  const nanochat::QkPrepParams qp = QkPrepParamsFor(c);

  auto gemm = [](const GemmParams& p, const ComputeType* a,
                 const ComputeType* b, ComputeType* out) {
    nanochat::kernels::Gemm(GemmMode::kForward, p, a, b, out);
  };
  auto norm = [&](const ComputeType* x, ComputeType* out) {
    nanochat::kernels::RmsNormForward(rms, x, out, m->rstd.ptr);
  };
  auto qkprep = [&](ComputeType* q, ComputeType* k) {
    nanochat::kernels::QkPrepForward(qp, m->cos.ptr, m->sin.ptr, q, k);
  };
  auto attn = [&](int kv_len, const ComputeType* q, const ComputeType* k,
                  const ComputeType* v, ComputeType* out) {
    const nanochat::AttentionParams ap = AttentionParamsFor(c, kv_len);
    nanochat::kernels::AttentionForward(ap, q, k, v, out, m->stats.ptr);
  };
  auto point = [](nanochat::PointwiseOp op, int n, const ComputeType* a,
                  const ComputeType* b, float alpha, float beta,
                  ComputeType* out) {
    nanochat::kernels::PointwiseForward(op, n, a, b, alpha, beta, out);
  };
  auto embed = [&](int token, ComputeType* out) {
    nanochat::kernels::EmbeddingForward(1, c.d_model, &token, m->emb.ptr, out);
  };
  auto copy = [](ComputeType* dst, const ComputeType* src, int count) {
    const std::size_t bytes =
        sizeof(ComputeType) * static_cast<std::size_t>(count);
    nanochat::kernels::Memcpy(dst, src, bytes,
                              nanochat::CopyDir::kDeviceToDevice);
  };

  RunDecodeStep(m, gemm, norm, qkprep, attn, point, embed, copy, token_host,
                pos);
  return 0;
}

// The graph path issues every op on `stream`; cuBLAS uses `handle`, whose
// stream is set to `stream` by the caller.
// `m->token_dev` must already hold the token id (staged outside capture).
void RunGraphPath(DecodeModel* m, cublasHandle_t handle, cudaStream_t stream,
                  int token_host, int pos) {
  const Config& c = m->cfg;

  auto gemm = [&](const GemmParams& p, const ComputeType* a,
                  const ComputeType* b, ComputeType* out) {
    DevGemm(handle, p, a, b, out);
  };
  auto norm = [&](const ComputeType* x, ComputeType* out) {
    nanochat::dev::DecodeRmsNormFwd(stream, 1, c.d_model, 1e-6f, x, out,
                                    m->rstd.ptr);
  };
  const nanochat::QkPrepParams qp = QkPrepParamsFor(c);
  auto qkprep = [&](ComputeType* q, ComputeType* k) {
    nanochat::dev::DecodeQkPrepFwd(stream, qp, m->cos.ptr, m->sin.ptr, q, k);
  };
  auto attn = [&](int kv_len, const ComputeType* q, const ComputeType* k,
                  const ComputeType* v, ComputeType* out) {
    const nanochat::AttentionParams ap = AttentionParamsFor(c, kv_len);
    nanochat::dev::DecodeAttentionFwd(stream, ap, q, k, v, out, m->stats.ptr);
  };
  auto point = [&](nanochat::PointwiseOp op, int n, const ComputeType* a,
                   const ComputeType* b, float alpha, float beta,
                   ComputeType* out) {
    nanochat::dev::DecodePointwiseFwd(stream, op, n, a, b, alpha, beta, out);
  };
  auto embed = [&](int /*token*/, ComputeType* out) {
    nanochat::dev::DecodeEmbeddingFwd(stream, 1, c.d_model, m->token_dev.ptr,
                                      m->emb.ptr, out);
  };
  auto copy = [&](ComputeType* dst, const ComputeType* src, int count) {
    cudaMemcpyAsync(dst, src,
                    sizeof(ComputeType) * static_cast<std::size_t>(count),
                    cudaMemcpyDeviceToDevice, stream);
  };

  RunDecodeStep(m, gemm, norm, qkprep, attn, point, embed, copy, token_host,
                pos);
}

std::vector<float> EagerLogits(DecodeModel* m, int token, int pos, int iters) {
  for (int i = 0; i < iters; ++i) RunEagerSeam(m, token, pos);
  nanochat::kernels::Synchronize();
  return FromStorage(m->logits.Download());
}

double TimeEager(DecodeModel* m, int token, int pos, int iters) {
  RunEagerSeam(m, token, pos);
  nanochat::kernels::Synchronize();
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  // Best of a few rounds: the seam path stages host ids and does synchronous
  // copies, so a single round is noisier than the graph replay.
  double best = 1e30;
  for (int round = 0; round < 3; ++round) {
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < iters; ++i) RunEagerSeam(m, token, pos);
    cudaEventRecord(end, nullptr);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    best = std::min(best, static_cast<double>(ms) / iters);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return best;
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf("decode fusion baseline (batch-1, cuBLAS + CUDA Graphs)\n");

  Config cfg;
  DecodeModel m(cfg);
  FillRope(&m);

  // Deterministic, position-independent inputs.
  Rng rng;
  const int token = 17;
  const int pos = cfg.seq_len - 1;
  m.k_cache.Upload(ToStorage(RandomVec(m.k_cache.count, &rng)));
  m.v_cache.Upload(ToStorage(RandomVec(m.v_cache.count, &rng)));

  std::printf(
      "  cfg layers=%d d_model=%d heads=%d head_dim=%d kv_heads=%d d_ff=%d "
      "vocab=%d seq_len=%d\n",
      cfg.layers, cfg.d_model, cfg.heads, cfg.head_dim, cfg.kv_heads, cfg.d_ff,
      cfg.vocab, cfg.seq_len);

  const int iters = 200;

  // --- Eager seam baseline ---------------------------------------------
  const std::vector<float> eager_logits = EagerLogits(&m, token, pos, 5);
  const double eager_ms = TimeEager(&m, token, pos, iters);
  const double eager_tok_s = 1000.0 / eager_ms;

  // --- cuBLAS + CUDA Graphs baseline -----------------------------------
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  // Stage the token id once; it is referenced by the captured embedding node.
  nanochat::kernels::Memcpy(m.token_dev.ptr, &token, sizeof(int),
                            nanochat::CopyDir::kHostToDevice);

  // Warm up on the stream (allocates any cuBLAS workspace and primes the
  // caches) before capturing.
  for (int i = 0; i < 3; ++i) RunGraphPath(&m, handle, stream, token, pos);
  cudaStreamSynchronize(stream);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t cap = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (cap != cudaSuccess) {
    std::printf("  CUDA graph capture failed: %s\n", cudaGetErrorString(cap));
    return 1;
  }
  RunGraphPath(&m, handle, stream, token, pos);
  cap = cudaStreamEndCapture(stream, &graph);
  if (cap != cudaSuccess) {
    std::printf("  CUDA graph end-capture failed: %s\n",
                cudaGetErrorString(cap));
    return 1;
  }
  cudaGraphInstantiate(&graph_exec, graph, 0);

  for (int i = 0; i < 5; ++i) cudaGraphLaunch(graph_exec, stream);
  cudaStreamSynchronize(stream);

  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  double graph_ms_per = 1e30;
  for (int round = 0; round < 3; ++round) {
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) cudaGraphLaunch(graph_exec, stream);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float graph_ms = 0.0f;
    cudaEventElapsedTime(&graph_ms, start, end);
    const double round_ms = static_cast<double>(graph_ms) / iters;
    graph_ms_per = std::min(graph_ms_per, round_ms);
  }
  const double graph_tok_s = 1000.0 / graph_ms_per;

  // Cross-check the graph path against the eager seam path.
  const std::vector<float> graph_logits = FromStorage(m.logits.Download());
  CheckVectorClose(graph_logits, eager_logits, 1e-3, "graph vs eager logits");

  std::printf("  eager seam      : %.4f ms/token  %.1f token/s\n", eager_ms,
              eager_tok_s);
  std::printf("  cuBLAS+graphs   : %.4f ms/token  %.1f token/s\n",
              graph_ms_per, graph_tok_s);
  std::printf("  graph speedup   : %.2fx\n", eager_ms / graph_ms_per);
  std::printf(
      "  note: persistent/MMA decode megakernel is sm_75+ only; this host is "
      "Pascal sm_61 (no mma/wmma), so only the baseline is reported.\n");

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);

  if (Failures() != 0) {
    std::printf("decode_fused_bench: %d check(s) failed\n", Failures());
    return 1;
  }
  std::printf("decode_fused_bench: graph matches eager\n");
  return 0;
}
