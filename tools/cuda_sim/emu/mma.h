#ifndef NANOCHAT_TOOLS_CUDA_SIM_EMU_MMA_H_
#define NANOCHAT_TOOLS_CUDA_SIM_EMU_MMA_H_

// The emulator's `<mma.h>`: the warp-level matrix multiply-accumulate interface
// (docs/simulator.md section 9.2).
//
// A fragment holds the whole micro-tile in every thread, so every thread
// computes and stores the same tile. The arithmetic is right and the cost is
// one factor of the warp width. The engine therefore checks the staging layout,
// the leading dimensions, the accumulation, and the store. It cannot check the
// per-lane fragment layout, a bank conflict, or the register count.
//
// One rule follows: pass a fragment only to `fill`, `load`, `mma`, and `store`.
// A kernel that reads one element of `fragment::x` depends on the vendor
// layout.
//
// Only the 16 by 16 by 16 form is covered, which is what the tree's tiles use.

#include <cstddef>
#include <type_traits>

#include <cuda_bf16.h>

namespace nvcuda {
namespace wmma {

// The use of a fragment, as an unscoped enumeration so the vendor spelling
// `fragment<matrix_a, ...>` works.
enum fragment_type { matrix_a = 0, matrix_b = 1, accumulator = 2 };

// The memory layout of a tile, as a runtime argument.
enum layout_t { mem_row_major = 0, mem_col_major = 1 };

// The memory layout of a matrix fragment, as a template argument.
struct row_major {};
struct col_major {};

namespace detail {

// One element of a tile at (row, column) under one layout and leading
// dimension. The load and the store share it, so they cannot disagree.
template <typename T>
inline std::size_t TileIndex(int row, int column, int ld, layout_t layout) {
  const std::size_t r = static_cast<std::size_t>(row);
  const std::size_t c = static_cast<std::size_t>(column);
  const std::size_t d = static_cast<std::size_t>(ld);
  return layout == mem_row_major ? r * d + c : c * d + r;
}

template <typename Layout>
constexpr layout_t TagToLayout() {
  return std::is_same_v<Layout, col_major> ? mem_col_major : mem_row_major;
}

}  // namespace detail

// One fragment. Every micro-tile edge is 16, so every kind holds 256 elements.
template <fragment_type Use, int M, int N, int K, typename T,
          typename Layout = void>
struct fragment {
  static_assert(M == 16 && N == 16 && K == 16,
                "the emulator covers the 16 by 16 by 16 fragment");
  T x[M * N];
};

template <int M, int N, int K, typename T, typename Layout>
inline void fill_fragment(fragment<accumulator, M, N, K, T, Layout>& d, T v) {
  for (int i = 0; i < M * N; ++i) d.x[i] = v;
}

// Reads a matrix operand. The layout is the fragment's template argument.
template <fragment_type Use, int M, int N, int K, typename T, typename Layout>
inline void load_matrix_sync(fragment<Use, M, N, K, T, Layout>& d,
                             const T* mptr, unsigned ldm) {
  static_assert(Use == matrix_a || Use == matrix_b,
                "a matrix operand is matrix_a or matrix_b");
  constexpr layout_t layout = detail::TagToLayout<Layout>();
  constexpr int rows = (Use == matrix_a) ? M : K;
  constexpr int columns = (Use == matrix_a) ? K : N;
  const int ld = static_cast<int>(ldm);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      d.x[i * columns + j] = mptr[detail::TileIndex<T>(i, j, ld, layout)];
    }
  }
}

// Reads an accumulator, for a `beta` term or a rescale between two tiles.
template <int M, int N, int K, typename T, typename Layout>
inline void load_matrix_sync(fragment<accumulator, M, N, K, T, Layout>& d,
                             const T* mptr, unsigned ldm,
                             layout_t layout = mem_row_major) {
  const int ld = static_cast<int>(ldm);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      d.x[i * N + j] = mptr[detail::TileIndex<T>(i, j, ld, layout)];
    }
  }
}

// `d = a * b + c`, accumulated in fp32 in every precision.
template <int M, int N, int K, typename Ta, typename La, typename Tb,
          typename Lb, typename Tc, typename Lc>
inline void mma_sync(fragment<accumulator, M, N, K, Tc, Lc>& d,
                     const fragment<matrix_a, M, N, K, Ta, La>& a,
                     const fragment<matrix_b, M, N, K, Tb, Lb>& b,
                     const fragment<accumulator, M, N, K, Tc, Lc>& c) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<float>(a.x[i * K + k]) *
               static_cast<float>(b.x[k * N + j]);
      }
      d.x[i * N + j] = c.x[i * N + j] + static_cast<Tc>(sum);
    }
  }
}

// Writes an accumulator.
template <int M, int N, int K, typename T, typename Layout>
inline void store_matrix_sync(
    T* mptr, const fragment<accumulator, M, N, K, T, Layout>& d, unsigned ldm,
    layout_t layout = mem_row_major) {
  const int ld = static_cast<int>(ldm);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      mptr[detail::TileIndex<T>(i, j, ld, layout)] = d.x[i * N + j];
    }
  }
}

}  // namespace wmma
}  // namespace nvcuda

#endif  // NANOCHAT_TOOLS_CUDA_SIM_EMU_MMA_H_
