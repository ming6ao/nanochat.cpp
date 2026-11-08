#ifndef NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_H_
#define NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_H_

// The host side of the CUTLASS bring-up probe (docs/kernels.md Part 3 section
// 5). This header names no CUDA type and includes no vendor header, so a plain
// host target reads it. The device side is `cutlass_probe.cu`, which builds at
// `sm_90a` and reports what it instantiated.
//
// **The probe answers three questions and measures no rate.**
//
// 1. Does a vendored CUTLASS compile in this tree, at this toolkit, with the
//    tree's C++20 setting? The tree's toolkit is CUDA 12.0 and CUTLASS needs
//    12.0 or newer.
// 2. Does a `cuda_library` that states `archs = ["sm_90a"]` build beside the
//    tree's default `sm_75`, with no command-line architecture flag? That is the
//    question `docs/kernels.md` Part 3 section 9 risk 2 leaves open, and the
//    answer decides how the adapter target is declared.
// 3. Which of the six tile candidates fit the Hopper window, and what stage
//    count does the collective builder choose for each? The builder measures the
//    window; a hand-written constant did not fit.
//
// A row is compile-time data. The binary launches no kernel and needs no card,
// so the probe is a host gate.

#include <cstddef>

namespace nanochat {
namespace cutlass_probe {

// One tile the probe instantiated, and what the builder chose for it.
struct TileReport {
  // The tile name, as the report spells it.
  const char* name;
  int tile_m;
  int tile_n;
  int tile_k;
  int cluster_m;
  // The stage count the builder selected for the mainloop.
  int stages;
  // The static shared storage the kernel asks for, mainloop and epilogue.
  std::size_t shared_bytes;
  // The block size the kernel launches with.
  int threads;
  // Whether `shared_bytes` fits the 227 KiB a Hopper block may ask for.
  bool fits_window;
};

// One (operand orientation, store type) case of the seam contract. The eight
// cases are the whole matrix `GemmParams` needs: two transposes on, two store
// types, and the forward, dgrad, and wgrad calls read every one of them.
struct CaseReport {
  const char* name;
  bool transpose_a;
  bool transpose_b;
  bool fp32_store;
  // The shared storage of the kernel this case instantiates.
  std::size_t shared_bytes;
  int threads;
};

int TileCount();
const TileReport& Tile(int index);

int LiveCases();
const CaseReport& Case(int index);

int TileCount();
const TileReport& Tile(int index);

// The address of the kernel entry of one row, or null when the translation unit
// emitted none. **A null is the finding, not a nuisance.** A `__global__`
// function template is emitted only when something odr-uses it, so a probe that
// merely names the type compiles the class and skips the mainloop body. The two
// accessors below exist so the body is compiled: a template that does not build
// fails here, and one that builds leaves SASS in the object for `cuobjdump`.
const void* CaseEntry(int index);
const void* TileEntry(int index);

}  // namespace cutlass_probe
}  // namespace nanochat

#endif  // NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_H_
