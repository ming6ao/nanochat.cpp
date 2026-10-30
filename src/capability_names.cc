// The lookups over the capability vocabulary table (src/capability_names.h).
// See that header for why the workflow keeps its own copy and for the two
// consumers that read a string instead of a constant.

#include "src/capability_names.h"

#include <cstring>

namespace nanochat {
namespace {

// One bit, and not zero: the shape every lookup below requires before it
// searches, so a multi-bit query cannot answer with a single row.
bool IsSingleBit(std::uint64_t bit) {
  return bit != 0 && (bit & (bit - 1)) == 0;
}

const CapabilityNameEntry* RowForBit(std::uint64_t bit) {
  if (!IsSingleBit(bit)) return nullptr;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (kCapabilityNames[i].bit == bit) return &kCapabilityNames[i];
  }
  return nullptr;
}

}  // namespace

std::uint64_t AllNamedCapabilityBits() {
  std::uint64_t all = 0;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    all |= kCapabilityNames[i].bit;
  }
  return all;
}

bool ParseCapabilityName(const std::string& name, std::uint64_t* out) {
  if (out == nullptr || name.empty()) return false;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    if (name == kCapabilityNames[i].name) {
      *out = kCapabilityNames[i].bit;
      return true;
    }
  }
  return false;
}

const char* CapabilityNameForBit(std::uint64_t bit) {
  const CapabilityNameEntry* row = RowForBit(bit);
  return row != nullptr ? row->name : nullptr;
}

const char* CapabilityConstantForBit(std::uint64_t bit) {
  const CapabilityNameEntry* row = RowForBit(bit);
  return row != nullptr ? row->constant : nullptr;
}

const char* CapabilityConsequence(std::uint64_t bit) {
  const CapabilityNameEntry* row = RowForBit(bit);
  return row != nullptr ? row->consequence : nullptr;
}

bool IsOverrideCapability(std::uint64_t bit) {
  return bit == kFeatureFusedResidualRmsNorm || bit == kFeatureFusedQkPrep ||
         bit == kFeatureFusedClassifier || bit == kFeatureFusedAttention;
}

bool IsStatefulCapability(std::uint64_t bit) {
  return bit == kFeatureAdamW || bit == kFeatureMuon;
}

std::string FormatCapabilityList(std::uint64_t mask) {
  std::string out;
  for (int i = 0; i < kCapabilityNameCount; ++i) {
    const std::uint64_t bit = kCapabilityNames[i].bit;
    if ((mask & bit) == 0) continue;
    if (!out.empty()) out += ", ";
    out += kCapabilityNames[i].name;
  }
  if (out.empty()) out = "(none)";
  return out;
}

}  // namespace nanochat
