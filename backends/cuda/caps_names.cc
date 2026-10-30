// The capability name table's lookups (caps_names.h). Host-only: no CUDA header
// and no device code, so a host test, the runtime, and the caps declaration
// check all link the same implementation.
//
// See backends/cuda/caps_names.h and spec/capability_bits.md section 2.

#include "backends/cuda/caps_names.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nanochat {
namespace cuda_backend {

bool ParseCapabilityName(const char* name, std::uint64_t* out) {
  if (name == nullptr || out == nullptr) return false;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (std::strcmp(name, kCapabilityNames[i].name) == 0) {
      *out = kCapabilityNames[i].bit;
      return true;
    }
  }
  return false;
}

const char* CapabilityNameForBit(std::uint64_t bit) {
  if (!IsSingleBit(bit)) return nullptr;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (kCapabilityNames[i].bit == bit) return kCapabilityNames[i].name;
  }
  return nullptr;
}

const char* CapabilityConstantForBit(std::uint64_t bit) {
  if (!IsSingleBit(bit)) return nullptr;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (kCapabilityNames[i].bit == bit) return kCapabilityNames[i].constant;
  }
  return nullptr;
}

void RequireNamedBits(std::uint64_t mask, const char* what) {
  const std::uint64_t unnamed = mask & ~AllNamedCapabilityBits();
  if (unnamed == 0) return;
  std::fprintf(stderr,
               "fatal: the capability mask from %s carries bit(s) that "
               "spec/capability_bits.md version 1 does not name:",
               what != nullptr ? what : "(unspecified)");
  for (int i = 0; i < 64; ++i) {
    const std::uint64_t bit = std::uint64_t{1} << i;
    if ((unnamed & bit) != 0) std::fprintf(stderr, " bit %d", i);
  }
  std::fprintf(stderr,
               "\n"
               "       A mask bit nobody can name is a promise the planner "
               "cannot read and a user cannot look up. Either the declaration "
               "names a bit this build does not define (fix caps.json), or "
               "include/nanochat/tensor.h grew a constant without a "
               "vocabulary row (fix spec/capability_bits.md and "
               "backends/cuda/caps_names.h together).\n");
  std::exit(1);
}

}  // namespace cuda_backend
}  // namespace nanochat
