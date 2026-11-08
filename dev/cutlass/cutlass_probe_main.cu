// The host entry point of the CUTLASS bring-up probe (docs/kernels.md Part 3
// section 5).
//
// **This translation unit builds at the tree's default architecture, not at
// `sm_90a`.** That is deliberate: it is the second half of the question the
// probe exists to answer. `:cutlass_probe_lib` states `archs = ["sm_90a"]` on
// its own `cuda_library`, and this binary states none, so a build that succeeds
// proves a per-target architecture coexists with the command-line one. If it
// did not, every `tools/nanochat build //...` on the Turing host would need a
// Hopper flag.
//
// The binary launches no kernel, so it runs on a host with no card.

#include <cstdio>

#include "dev/cutlass/cutlass_probe.h"
#include "nanochat/sandbox.h"

namespace {

using nanochat::cutlass_probe::Case;
using nanochat::cutlass_probe::CaseEntry;
using nanochat::cutlass_probe::LiveCases;
using nanochat::cutlass_probe::Tile;
using nanochat::cutlass_probe::TileCount;
using nanochat::cutlass_probe::TileEntry;

void PrintTiles() {
  std::printf("\ntiles, as the collective builder sizes them\n");
  std::printf("%-14s %-8s %6s %10s %8s %-5s %s\n", "tile", "cluster", "stages",
              "shared", "threads", "window", "kernel");
  for (int i = 0; i < TileCount(); ++i) {
    const auto& row = Tile(i);
    std::printf("%-14s %dx%-6d %6d %9zu B %8d %-5s %s\n", row.name,
                row.cluster_m, 1, row.stages, row.shared_bytes, row.threads,
                row.fits_window ? "fits" : "OVER",
                TileEntry(i) != nullptr ? "emitted" : "ABSENT");
  }
}

void PrintCases() {
  std::printf("\nthe eight seam cases, all at the primary tile\n");
  std::printf("%-42s %8s %10s %8s %s\n", "case", "fp32", "shared", "threads",
              "kernel");
  for (int i = 0; i < LiveCases(); ++i) {
    const auto& row = Case(i);
    std::printf("%-42s %8s %9zu B %8d %s\n", row.name,
                row.fp32_store ? "store" : "no", row.shared_bytes, row.threads,
                CaseEntry(i) != nullptr ? "emitted" : "ABSENT");
  }
}

}  // namespace

int main() {
  nanochat::RequireSandboxOrDie("bench");
  std::printf("cutlass probe: what the vendored headers instantiate\n");
  std::printf("arch=sm_90a tiles=%d cases=%d\n", TileCount(), LiveCases());
  PrintTiles();
  PrintCases();
  return 0;
}
