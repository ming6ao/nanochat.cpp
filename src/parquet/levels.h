#ifndef NANOCHAT_SRC_PARQUET_LEVELS_H_
#define NANOCHAT_SRC_PARQUET_LEVELS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The parquet RLE and bit-packed hybrid decoder (docs/parquet-native.md). The
// definition and repetition levels of a data page use this encoding. The
// decoder produces exactly `count` values.

namespace nanochat {
namespace parquet {

bool DecodeRleBitPacked(const std::uint8_t* data, std::size_t size,
                        int bit_width, std::size_t count,
                        std::vector<std::uint32_t>* out, std::string* error);

}  // namespace parquet
}  // namespace nanochat

#endif  // NANOCHAT_SRC_PARQUET_LEVELS_H_
