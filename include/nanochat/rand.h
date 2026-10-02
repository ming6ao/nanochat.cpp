#ifndef NANOCHAT_RAND_H_
#define NANOCHAT_RAND_H_

#include <cmath>
#include <cstdint>

// Deterministic host random number generator (xoshiro256** seeded with
// splitmix64). Every stochastic path in the stack draws from an explicit Rand
// so runs are reproducible under a fixed seed (docs/testing.md). Vendor-free
// and header-only.

namespace nanochat {

class Rand {
 public:
  explicit Rand(std::uint64_t seed = 0) { Seed(seed); }

  void Seed(std::uint64_t seed) {
    for (int i = 0; i < 4; ++i) {
      seed += 0x9e3779b97f4a7c15ULL;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      state_[i] = z ^ (z >> 31);
    }
    if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0) state_[0] = 1;
  }

  std::uint64_t NextU64() {
    const std::uint64_t result = Rotl(state_[1] * 5, 7) * 9;
    const std::uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = Rotl(state_[3], 45);
    return result;
  }

  std::uint32_t NextU32() {
    return static_cast<std::uint32_t>(NextU64() >> 32);
  }

  // Uniform in [0, 1).
  float NextFloat() { return static_cast<float>(NextU64() >> 40) * 0x1p-24f; }

  // Uniform in [low, high).
  float NextFloat(float low, float high) {
    return low + (high - low) * NextFloat();
  }

  // Uniform integer in [low, high].
  int NextInt(int low, int high) {
    if (high <= low) return low;
    const std::uint32_t span = static_cast<std::uint32_t>(high - low + 1);
    return low + static_cast<int>(NextU32() % span);
  }

  // Standard normal via Box-Muller.
  float NextNormal() {
    const float u1 = 1.0f - NextFloat();
    const float u2 = NextFloat();
    constexpr float kTwoPi = 6.28318530717958647692f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(kTwoPi * u2);
  }

  // Normal with the given mean and standard deviation.
  float NextNormal(float mean, float stddev) {
    return mean + stddev * NextNormal();
  }

 private:
  static std::uint64_t Rotl(std::uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
  }

  std::uint64_t state_[4] = {1, 2, 3, 4};
};

}  // namespace nanochat

#endif  // NANOCHAT_RAND_H_
