// The acceptance check for the emulator's `<mma.h>`, `<cuda_bf16.h>`, and
// `<cuda_pipeline.h>` (docs/tensor-core-gemm-plan.md sections A.1 and A.2).
//
// It runs the four `wmma` calls over the four operand layouts through the block
// runtime, then covers the properties a single round trip would not show: a
// ragged leading dimension with a poisoned pad, an accumulator read back, an
// in-place accumulation, the bf16 conversions, and the asynchronous copy.
//
// The operands are multiples of a power of two, so every product and every
// 16-term sum is exact in fp32 and a mismatch is a mapping defect.
//
// CPU only. The kernel reads host pointers, because the emulator runs every
// fiber on the launching host thread.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "backends/cuda/kernels/mma_ops.cuh"
#include "nanochat/tensor.h"

namespace {

namespace wmma = nvcuda::wmma;

using nanochat::Bf16;

int g_failures = 0;

void Fail(const char* what, const char* case_name) {
  std::fprintf(stderr, "FAIL: %s (%s)\n", what, case_name);
  ++g_failures;
}

void Expect(bool condition, const char* what, const char* case_name) {
  if (!condition) Fail(what, case_name);
}

constexpr int kTile = 16;
constexpr int kTileElements = kTile * kTile;
// A leading dimension wider than the tile, for the ragged case.
constexpr int kRaggedLd = 24;
// Far outside the operand range, so a read that lands in the pad shows.
constexpr float kPadPoison = 12345.0f;
// The operands are exact in fp32, so the comparison needs no fuzz.
constexpr float kTolerance = 1e-6f;

// A[i][j] and B[k][j]. Both are multiples of a power of two, so they are exact
// in bf16 and in fp32.
float AValue(int i, int j) {
  return 0.25f * static_cast<float>((i * 5 + j * 3) % 7 - 3);
}

float BValue(int k, int j) {
  return 0.5f * static_cast<float>((k * 2 + j * 3) % 5 - 2);
}

// C = A * B, in double, from the operands as they stand.
std::vector<float> Reference() {
  std::vector<float> out(kTileElements, 0.0f);
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kTile; ++j) {
      double sum = 0.0;
      for (int k = 0; k < kTile; ++k) {
        sum += static_cast<double>(AValue(i, k)) * BValue(k, j);
      }
      out[i * kTile + j] = static_cast<float>(sum);
    }
  }
  return out;
}

void ExpectTile(const std::vector<float>& got, const std::vector<float>& want,
                const char* case_name) {
  for (int i = 0; i < kTileElements; ++i) {
    if (std::fabs(got[i] - want[i]) > kTolerance) {
      std::fprintf(stderr,
                   "FAIL: tile mismatch (%s): element %d got %f want %f\n",
                   case_name, i, static_cast<double>(got[i]),
                   static_cast<double>(want[i]));
      ++g_failures;
      return;
    }
  }
}

// One micro-tile through the block runtime: stage both operands in shared
// memory, load the two fragments, accumulate, and store.
template <typename ALayout, typename BLayout>
__global__ void StagedMmaKernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                float* c) {
  __shared__ __nv_bfloat16 sa[kTileElements];
  __shared__ __nv_bfloat16 sb[kTileElements];
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);
  for (int i = tid; i < kTileElements; i += stride) {
    sa[i] = a[i];
    sb[i] = b[i];
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, ALayout>
      fa;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, BLayout>
      fb;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  wmma::load_matrix_sync(fa, sa, kTile);
  wmma::load_matrix_sync(fb, sb, kTile);
  wmma::mma_sync(acc, fa, fb, acc);
  __syncthreads();
  wmma::store_matrix_sync(c, acc, kTile, wmma::mem_row_major);
}

// Stages one operand in the layout its fragment declares.
template <typename Layout>
void FillOperand(std::vector<Bf16>* tile, float (*value)(int, int)) {
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kTile; ++j) {
      const std::size_t index = std::is_same_v<Layout, wmma::row_major>
                                    ? static_cast<std::size_t>(i) * kTile + j
                                    : static_cast<std::size_t>(j) * kTile + i;
      (*tile)[index] = Bf16(value(i, j));
    }
  }
}

template <typename Layout>
const char* LayoutName() {
  return std::is_same_v<Layout, wmma::row_major> ? "row" : "col";
}

template <typename ALayout, typename BLayout>
void CheckStagedMma(int threads) {
  std::vector<Bf16> a(kTileElements);
  std::vector<Bf16> b(kTileElements);
  FillOperand<ALayout>(&a, AValue);
  FillOperand<BLayout>(&b, BValue);

  std::vector<float> got(kTileElements, 0.0f);
  nanochat::cuda_backend::Launch(StagedMmaKernel<ALayout, BLayout>, dim3(1u),
                                 dim3(static_cast<unsigned int>(threads)), 0,
                                 a.data(), b.data(), got.data());

  char name[64];
  std::snprintf(name, sizeof(name), "staged %s x %s, %d thread(s)",
                LayoutName<ALayout>(), LayoutName<BLayout>(), threads);
  ExpectTile(got, Reference(), name);
}

// A ragged leading dimension with a poisoned pad, called directly.
void CheckRaggedLeadingDimension() {
  const char* kCase = "ragged leading dimension";
  std::vector<Bf16> a(static_cast<std::size_t>(kTile) * kRaggedLd);
  std::vector<Bf16> b(static_cast<std::size_t>(kTile) * kRaggedLd);
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kRaggedLd; ++j) {
      const bool inside = j < kTile;
      a[static_cast<std::size_t>(i) * kRaggedLd + j] =
          Bf16(inside ? AValue(i, j) : kPadPoison);
      b[static_cast<std::size_t>(i) * kRaggedLd + j] =
          Bf16(inside ? BValue(i, j) : kPadPoison);
    }
  }

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fa;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fb;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  wmma::load_matrix_sync(fa, a.data(), kRaggedLd);
  wmma::load_matrix_sync(fb, b.data(), kRaggedLd);
  wmma::mma_sync(acc, fa, fb, acc);

  std::vector<float> got(static_cast<std::size_t>(kTile) * kRaggedLd, 0.0f);
  wmma::store_matrix_sync(got.data(), acc, kRaggedLd, wmma::mem_row_major);

  std::vector<float> tile(kTileElements, 0.0f);
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kTile; ++j)
      tile[i * kTile + j] = got[i * kRaggedLd + j];
    for (int j = kTile; j < kRaggedLd; ++j) {
      Expect(got[i * kRaggedLd + j] == 0.0f,
             "the store must not write past the tile row", kCase);
    }
  }
  ExpectTile(tile, Reference(), kCase);
}

// `mma_sync` with `d` and `c` the same object, twice: the attention tile's
// accumulation.
void CheckAccumulateInPlace() {
  const char* kCase = "accumulate in place";
  std::vector<Bf16> a(kTileElements);
  std::vector<Bf16> b(kTileElements);
  FillOperand<wmma::row_major>(&a, AValue);
  FillOperand<wmma::row_major>(&b, BValue);

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fa;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fb;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  wmma::load_matrix_sync(fa, a.data(), kTile);
  wmma::load_matrix_sync(fb, b.data(), kTile);
  wmma::mma_sync(acc, fa, fb, acc);
  wmma::mma_sync(acc, fa, fb, acc);

  std::vector<float> got(kTileElements, 0.0f);
  wmma::store_matrix_sync(got.data(), acc, kTile, wmma::mem_row_major);

  std::vector<float> want = Reference();
  for (float& value : want) value *= 2.0f;
  ExpectTile(got, want, kCase);
}

// An accumulator read back column-major, then used as the `c` of `mma_sync`:
// the `beta` term, and the rescale between two tiles.
void CheckAccumulatorReload() {
  const char* kCase = "accumulator reload";
  std::vector<Bf16> a(kTileElements);
  std::vector<Bf16> b(kTileElements);
  FillOperand<wmma::row_major>(&a, AValue);
  FillOperand<wmma::row_major>(&b, BValue);

  // A column-major seed that differs per row and column, so a transposed read
  // is a failure.
  std::vector<float> seed(kTileElements, 0.0f);
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kTile; ++j) {
      seed[j * kTile + i] =
          static_cast<float>(i) - 0.5f * static_cast<float>(j);
    }
  }

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fa;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16,
                 wmma::row_major>
      fb;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  wmma::load_matrix_sync(fa, a.data(), kTile);
  wmma::load_matrix_sync(fb, b.data(), kTile);
  wmma::load_matrix_sync(acc, seed.data(), kTile, wmma::mem_col_major);
  wmma::mma_sync(acc, fa, fb, acc);

  std::vector<float> got(kTileElements, 0.0f);
  wmma::store_matrix_sync(got.data(), acc, kTile, wmma::mem_row_major);

  std::vector<float> want = Reference();
  for (int i = 0; i < kTile; ++i) {
    for (int j = 0; j < kTile; ++j) {
      want[i * kTile + j] +=
          static_cast<float>(i) - 0.5f * static_cast<float>(j);
    }
  }
  ExpectTile(got, want, kCase);
}

void CheckBf16Conversions() {
  const char* kCase = "bf16 conversions";
  const float values[] = {0.0f, 1.0f, -2.5f, 0.1f, 1e-3f, 1234.5f, 1e30f};
  for (float value : values) {
    const std::uint16_t expected = nanochat::FloatToBf16Bits(value);
    const __nv_bfloat16 round_trip = __float2bfloat16(value);
    Expect(round_trip.bits == expected,
           "__float2bfloat16 uses the shared rounding", kCase);
    Expect(__float2bfloat16_rn(value).bits == expected,
           "__float2bfloat16_rn is the same rounding", kCase);
    Expect(__bfloat162float(round_trip) == nanochat::Bf16BitsToFloat(expected),
           "__bfloat162float is exact", kCase);
  }

  const __nv_bfloat162 pair = __floats2bfloat162_rn(0.5f, -3.0f);
  Expect(__low2float(pair) == 0.5f, "__low2float reads the low lane", kCase);
  Expect(__high2float(pair) == -3.0f, "__high2float reads the high lane",
         kCase);
}

// The fp8 micro-tile, over the four format pairs and with `d` and `c` the same
// object. The reference decodes with its own mapping, so a wrong arm of the
// header cannot move the expectation.
void CheckFp8Mma() {
  using nanochat::Fp8Format;
  using nanochat::cuda_kernels::Fp8MicroA;
  using nanochat::cuda_kernels::Fp8MicroB;
  using nanochat::cuda_kernels::Fp8MicroC;
  using nanochat::cuda_kernels::Fp8MmaTile;
  using nanochat::cuda_kernels::kFp8MmaK;
  using nanochat::cuda_kernels::kFp8MmaM;
  using nanochat::cuda_kernels::kFp8MmaN;

  const char* kCase = "fp8 micro-tile";
  // The values both formats hold exactly.
  static constexpr float kValues[] = {-2.0f, -1.0f, -0.5f, 0.0f,
                                      0.25f, 0.5f,  1.0f,  2.0f};
  constexpr int kValueCount =
      static_cast<int>(sizeof(kValues) / sizeof(kValues[0]));
  for (float value : kValues) {
    Expect(nanochat::Fp8E4M3Decode(nanochat::Fp8E4M3Encode(value)) == value,
           "E4M3 holds this value exactly", kCase);
    Expect(nanochat::Fp8E5M2Decode(nanochat::Fp8E5M2Encode(value)) == value,
           "E5M2 holds this value exactly", kCase);
  }

  Fp8MicroA a;
  Fp8MicroB b;
  for (int i = 0; i < kFp8MmaM; ++i) {
    for (int k = 0; k < kFp8MmaK; ++k) {
      a.x[i * kFp8MmaK + k] =
          nanochat::Fp8E4M3Encode(kValues[(i * 3 + k * 5) % kValueCount]);
    }
  }
  for (int k = 0; k < kFp8MmaK; ++k) {
    for (int j = 0; j < kFp8MmaN; ++j) {
      b.x[k * kFp8MmaN + j] =
          nanochat::Fp8E5M2Encode(kValues[(k * 7 + j * 2) % kValueCount]);
    }
  }

  struct Pair {
    Fp8Format a;
    Fp8Format b;
    const char* name;
  };
  const Pair pairs[] = {
      {Fp8Format::kE4M3, Fp8Format::kE4M3, "E4M3 by E4M3"},
      {Fp8Format::kE4M3, Fp8Format::kE5M2, "E4M3 by E5M2"},
      {Fp8Format::kE5M2, Fp8Format::kE4M3, "E5M2 by E4M3"},
      {Fp8Format::kE5M2, Fp8Format::kE5M2, "E5M2 by E5M2"},
  };

  for (const Pair& pair : pairs) {
    // Twice for the first pair, which proves the in-place accumulation.
    const int repeats =
        pair.a == Fp8Format::kE4M3 && pair.b == Fp8Format::kE5M2 ? 2 : 1;
    Fp8MicroC acc;
    for (float& value : acc.x) value = 0.0f;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      Fp8MmaTile(&acc, a, b, acc, pair.a, pair.b);
    }

    float worst = 0.0f;
    for (int i = 0; i < kFp8MmaM; ++i) {
      for (int j = 0; j < kFp8MmaN; ++j) {
        double sum = 0.0;
        for (int k = 0; k < kFp8MmaK; ++k) {
          const std::uint8_t a_byte = a.x[i * kFp8MmaK + k];
          const std::uint8_t b_byte = b.x[k * kFp8MmaN + j];
          const float av = pair.a == Fp8Format::kE4M3
                               ? nanochat::Fp8E4M3Decode(a_byte)
                               : nanochat::Fp8E5M2Decode(a_byte);
          const float bv = pair.b == Fp8Format::kE4M3
                               ? nanochat::Fp8E4M3Decode(b_byte)
                               : nanochat::Fp8E5M2Decode(b_byte);
          sum += static_cast<double>(av) * static_cast<double>(bv);
        }
        const float want = static_cast<float>(sum * repeats);
        worst = std::fmax(worst, std::fabs(acc.x[i * kFp8MmaN + j] - want));
      }
    }
    if (worst > 1e-5f) {
      std::fprintf(stderr, "FAIL: %s mismatch (%s): worst %f\n", pair.name,
                   kCase, static_cast<double>(worst));
      ++g_failures;
    }
  }
}

// One full copy, one copy with a zero-filled tail, and the two empty calls.
void CheckPipelineCopy() {
  const char* kCase = "pipeline copy";
  std::uint8_t source[64];
  std::uint8_t destination[80];
  for (int i = 0; i < 64; ++i) source[i] = static_cast<std::uint8_t>(i + 1);

  std::memset(destination, 0xAB, sizeof(destination));
  __pipeline_memcpy_async(destination, source, sizeof(source));
  __pipeline_commit();
  __pipeline_wait_prior(0);
  Expect(std::memcmp(destination, source, sizeof(source)) == 0,
         "the copy completes at the call", kCase);
  Expect(destination[64] == 0xAB, "a copy without zfill leaves the tail",
         kCase);

  std::memset(destination, 0xAB, sizeof(destination));
  __pipeline_memcpy_async(destination, source, 64, 16);
  __pipeline_commit();
  __pipeline_wait_prior(1);
  Expect(std::memcmp(destination, source, 48) == 0,
         "the copied part leads the zero fill", kCase);
  for (int i = 48; i < 64; ++i) {
    Expect(destination[i] == 0, "the zfill tail is zero", kCase);
  }
  Expect(destination[64] == 0xAB, "a zfill copy leaves the rest", kCase);
}

}  // namespace

int main() {
  // The four operand layouts, at a full warp and at one thread.
  CheckStagedMma<wmma::row_major, wmma::row_major>(32);
  CheckStagedMma<wmma::row_major, wmma::row_major>(1);
  CheckStagedMma<wmma::row_major, wmma::col_major>(32);
  CheckStagedMma<wmma::col_major, wmma::row_major>(32);
  CheckStagedMma<wmma::col_major, wmma::col_major>(32);

  CheckRaggedLeadingDimension();
  CheckAccumulateInPlace();
  CheckAccumulatorReload();
  CheckBf16Conversions();
  CheckFp8Mma();
  CheckPipelineCopy();

  if (g_failures != 0) {
    std::fprintf(stderr, "mma_emu: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("mma_emu: ok\n");
  return 0;
}
