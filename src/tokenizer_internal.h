#ifndef NANOCHAT_SRC_TOKENIZER_INTERNAL_H_
#define NANOCHAT_SRC_TOKENIZER_INTERNAL_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Internal writer for the `NCTOKEN1` artifact. The public interface in
// `nanochat/tokenizer.h` exposes the loader only. The trainer command line and
// the tests use this writer (docs/tokenizer.md section 5).

namespace nanochat {

// Writes one little-endian `NCTOKEN1` artifact. The rank of a merge pair is
// 256 plus its index. The special records are written in ascending id order.
// The method returns false when the file cannot be opened or written.
bool SaveTokenizer(
    const std::string& path, std::string_view pattern,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& merge_pairs,
    const std::vector<std::pair<std::string, std::uint32_t>>& special_tokens);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_TOKENIZER_INTERNAL_H_
