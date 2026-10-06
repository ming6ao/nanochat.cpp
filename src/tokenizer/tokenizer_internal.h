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

// Recovers one `(left, right)` merge pair for every merge in a trained list.
// `merges` holds the token bytes and the rank of each merge, in rank order.
// The method picks the smallest left rank and then the smallest right rank,
// the same rule as the native trainer writer. A pair is `(0, 0)` when no
// split reconstructs the token.
std::vector<std::pair<std::uint32_t, std::uint32_t>> RecoverMergePairs(
    const std::vector<std::pair<std::vector<std::uint8_t>, int>>& merges);

}  // namespace nanochat

#endif  // NANOCHAT_SRC_TOKENIZER_INTERNAL_H_
