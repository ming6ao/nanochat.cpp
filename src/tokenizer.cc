#include "nanochat/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/split_pattern.h"
#include "src/tokenizer_internal.h"
#include "src/utf8.h"

namespace nanochat {
namespace {

constexpr std::size_t kByteTokenCount = 256;
constexpr char kMagic[] = "NCTOKEN1";
constexpr std::uint32_t kVersion = 1;

// The loader accepts a token byte string up to this size. A malformed file
// can build a very long token through nested merges, so the loader bounds the
// work. The limit also keeps the derived length inside the `uint8_t` result of
// `TokenBytes`. The total guard bounds the memory for every token.
constexpr std::size_t kMaxTokenBytes = 255;
constexpr std::size_t kMaxTotalBytes = 64u << 20;

bool ReadU32(const std::vector<std::uint8_t>& data, std::size_t* offset,
             std::uint32_t* value) {
  if (*offset + 4 > data.size()) return false;
  const std::uint8_t* bytes = data.data() + *offset;
  *value = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
  *offset += 4;
  return true;
}

bool ReadBytes(const std::vector<std::uint8_t>& data, std::size_t* offset,
               std::size_t count, std::string* value) {
  if (*offset + count > data.size()) return false;
  value->assign(reinterpret_cast<const char*>(data.data() + *offset), count);
  *offset += count;
  return true;
}

void WriteU32(std::ostream& out, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8) & 0xffu),
      static_cast<unsigned char>((value >> 16) & 0xffu),
      static_cast<unsigned char>((value >> 24) & 0xffu)};
  out.write(reinterpret_cast<const char*>(bytes), 4);
}

class BpeTokenizer : public Tokenizer {
 public:
  BpeTokenizer(std::vector<std::string> token_bytes,
               std::vector<std::pair<std::string, int>> specials)
      : token_bytes_(std::move(token_bytes)) {
    base_vocab_size_ = static_cast<int>(token_bytes_.size());
    vocab_size_ = base_vocab_size_ + static_cast<int>(specials.size());

    // Map each base token byte string to its id. The encoder looks up the
    // concatenated bytes of an adjacent pair, the same way tiktoken does.
    for (int id = 0; id < base_vocab_size_; ++id) {
      byte_rank_.emplace(token_bytes_[id], id);
    }

    id_bytes_.resize(vocab_size_);
    token_lengths_.assign(vocab_size_, 0);
    is_special_.assign(vocab_size_, 0);
    for (int id = 0; id < base_vocab_size_; ++id) {
      id_bytes_[id] = token_bytes_[id];
      token_lengths_[id] = static_cast<std::uint8_t>(token_bytes_[id].size());
    }
    for (const auto& [name, id] : specials) {
      if (id < base_vocab_size_ || id >= vocab_size_) continue;
      id_bytes_[id] = name;
      token_lengths_[id] = 0;
      is_special_[id] = 1;
      special_names_.push_back(name);
      special_ids_.push_back(id);
      if (name == "<|bos|>") bos_id_ = id;
    }
  }

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }

  int SpecialId(std::string_view name) const override {
    for (std::size_t i = 0; i < special_names_.size(); ++i) {
      if (special_names_[i] == name) return special_ids_[i];
    }
    return -1;
  }

  std::vector<int> Encode(std::string_view text, int prepend,
                          int append) const override {
    std::vector<int> ids;
    if (prepend >= 0) ids.push_back(prepend);
    for (std::string_view piece : SplitPattern(text)) {
      AppendPieceIds(piece, &ids);
    }
    if (append >= 0) ids.push_back(append);
    return ids;
  }

  std::string Decode(const int* ids, int count) const override {
    std::string raw;
    for (int index = 0; index < count; ++index) {
      const int id = ids[index];
      if (id < 0 || id >= vocab_size_) continue;
      raw.append(id_bytes_[id]);
    }
    return Utf8Replaced(raw);
  }

  bool AppendTokenBytes(int id, std::string* out) const override {
    if (id < 0 || id >= vocab_size_) return false;
    out->append(id_bytes_[id]);
    return true;
  }

  bool IsSpecial(int id) const override {
    return id >= 0 && id < vocab_size_ && is_special_[id] != 0;
  }

  const std::uint8_t* TokenBytes() const override {
    return token_lengths_.data();
  }

 private:
  int PairRank(int left, int right, std::string* scratch) const {
    scratch->clear();
    scratch->append(id_bytes_[left]);
    scratch->append(id_bytes_[right]);
    const auto it = byte_rank_.find(*scratch);
    return it == byte_rank_.end() ? -1 : it->second;
  }

  void AppendPieceIds(std::string_view piece, std::vector<int>* out) const {
    if (piece.empty()) return;
    std::vector<int> ids(piece.size());
    for (std::size_t index = 0; index < piece.size(); ++index) {
      ids[index] = static_cast<unsigned char>(piece[index]);
    }
    std::string scratch;
    while (ids.size() >= 2) {
      bool found = false;
      std::size_t best_index = 0;
      int best_rank = 0;
      for (std::size_t index = 0; index + 1 < ids.size(); ++index) {
        const int rank = PairRank(ids[index], ids[index + 1], &scratch);
        if (rank >= 0 && (!found || rank < best_rank)) {
          found = true;
          best_rank = rank;
          best_index = index;
        }
      }
      if (!found) break;
      ids[best_index] = best_rank;
      ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    out->insert(out->end(), ids.begin(), ids.end());
  }

  int base_vocab_size_ = 0;
  int vocab_size_ = 0;
  int bos_id_ = -1;
  std::vector<std::string> token_bytes_;
  std::vector<std::string> id_bytes_;
  std::vector<std::uint8_t> token_lengths_;
  std::vector<std::uint8_t> is_special_;
  std::vector<std::string> special_names_;
  std::vector<int> special_ids_;
  std::unordered_map<std::string, int> byte_rank_;
};

}  // namespace

std::unique_ptr<Tokenizer> LoadTokenizer(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return nullptr;
  std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (data.size() < 16) return nullptr;
  if (std::memcmp(data.data(), kMagic, 8) != 0) return nullptr;

  std::size_t offset = 8;
  std::uint32_t version = 0;
  if (!ReadU32(data, &offset, &version) || version != kVersion) return nullptr;

  std::uint32_t pattern_length = 0;
  if (!ReadU32(data, &offset, &pattern_length)) return nullptr;
  std::string pattern;
  if (!ReadBytes(data, &offset, pattern_length, &pattern)) return nullptr;
  if (pattern != NanochatSplitPattern()) return nullptr;

  std::uint32_t merge_count = 0;
  if (!ReadU32(data, &offset, &merge_count)) return nullptr;
  if (merge_count > (data.size() - offset) / 8) return nullptr;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> merge_pairs(merge_count);
  for (std::uint32_t index = 0; index < merge_count; ++index) {
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    if (!ReadU32(data, &offset, &left) || !ReadU32(data, &offset, &right)) {
      return nullptr;
    }
    merge_pairs[index] = {left, right};
  }

  // Derive the token bytes from the merge pairs (docs/tokenizer.md section 5).
  std::vector<std::string> token_bytes(kByteTokenCount);
  for (std::size_t id = 0; id < kByteTokenCount; ++id) {
    token_bytes[id] = std::string(1, static_cast<char>(id));
  }
  std::size_t total_bytes = kByteTokenCount;
  for (std::uint32_t index = 0; index < merge_count; ++index) {
    const std::uint32_t rank =
        static_cast<std::uint32_t>(kByteTokenCount) + index;
    const std::uint32_t left = merge_pairs[index].first;
    const std::uint32_t right = merge_pairs[index].second;
    if (left >= rank || right >= rank) return nullptr;
    const std::size_t length =
        token_bytes[left].size() + token_bytes[right].size();
    if (length > kMaxTokenBytes) return nullptr;
    total_bytes += length;
    if (total_bytes > kMaxTotalBytes) return nullptr;
    token_bytes.push_back(token_bytes[left] + token_bytes[right]);
  }

  std::uint32_t special_count = 0;
  if (!ReadU32(data, &offset, &special_count)) return nullptr;
  std::vector<std::pair<std::string, int>> specials;
  for (std::uint32_t index = 0; index < special_count; ++index) {
    std::uint32_t name_length = 0;
    if (!ReadU32(data, &offset, &name_length)) return nullptr;
    std::string name;
    if (!ReadBytes(data, &offset, name_length, &name)) return nullptr;
    std::uint32_t id = 0;
    if (!ReadU32(data, &offset, &id)) return nullptr;
    specials.emplace_back(std::move(name), static_cast<int>(id));
  }
  if (offset != data.size()) return nullptr;

  const int base_vocab_size = static_cast<int>(token_bytes.size());
  const int vocab_size = base_vocab_size + static_cast<int>(special_count);
  std::vector<char> seen(vocab_size, 0);
  for (const auto& [name, id] : specials) {
    if (id < base_vocab_size || id >= vocab_size) return nullptr;
    if (seen[id] != 0) return nullptr;
    seen[id] = 1;
  }

  return std::make_unique<BpeTokenizer>(std::move(token_bytes),
                                        std::move(specials));
}

bool SaveTokenizer(
    const std::string& path, std::string_view pattern,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& merge_pairs,
    const std::vector<std::pair<std::string, std::uint32_t>>& special_tokens) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 8);
  WriteU32(out, kVersion);
  WriteU32(out, static_cast<std::uint32_t>(pattern.size()));
  out.write(pattern.data(), static_cast<std::streamsize>(pattern.size()));
  WriteU32(out, static_cast<std::uint32_t>(merge_pairs.size()));
  for (const auto& [left, right] : merge_pairs) {
    WriteU32(out, left);
    WriteU32(out, right);
  }
  std::vector<std::pair<std::string, std::uint32_t>> ordered = special_tokens;
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& left, const auto& right) {
              return left.second < right.second;
            });
  WriteU32(out, static_cast<std::uint32_t>(ordered.size()));
  for (const auto& [name, id] : ordered) {
    WriteU32(out, static_cast<std::uint32_t>(name.size()));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
    WriteU32(out, id);
  }
  return out.good();
}

TokenStreamDecoder::TokenStreamDecoder(const Tokenizer& tokenizer)
    : tokenizer_(tokenizer) {}

void TokenStreamDecoder::Push(int id, std::string* out) {
  if (!tokenizer_.AppendTokenBytes(id, &pending_)) return;
  std::size_t position = 0;
  while (position < pending_.size()) {
    const Utf8Sequence sequence = ReadUtf8(pending_, position);
    if (sequence.status == Utf8Status::kIncomplete) break;
    position += sequence.length;
  }
  if (position == 0) return;
  const std::string complete = pending_.substr(0, position);
  pending_.erase(0, position);
  AppendUtf8Replaced(complete, out);
}

void TokenStreamDecoder::Flush(std::string* out) {
  AppendUtf8Replaced(pending_, out);
  pending_.clear();
}

}  // namespace nanochat
