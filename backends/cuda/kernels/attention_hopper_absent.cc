// The stand-in for the Hopper forward tile (docs/attention.md Part 8,
// docs/simulator.md section 9.2).
//
// The tile states `archs = ["sm_90a"]` on its own target, and it reads the
// warpgroup product and the tiled copy, which the emulation engine maps for
// addressing and not for a rate. So every build that is not a device build of
// that one architecture links this file instead of `attention_hopper.cu`, and
// `attention.cu` asks the same two questions in every build.
//
// `local_defines = precision_defines()` keeps `ComputeType` the same on both
// sides of the call: this file and its caller compile at the workspace
// precision.

#include "backends/cuda/kernels/attention_hopper.h"

namespace nanochat {
namespace kernels {

bool UseHopperAttention(const AttentionParams&) { return false; }

void LaunchHopperAttention(const AttentionParams&, const ComputeType*,
                           const ComputeType*, const ComputeType*, ComputeType*,
                           float*) {}

}  // namespace kernels
}  // namespace nanochat
