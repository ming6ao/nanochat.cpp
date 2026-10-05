// The native byte pair encoding trainer (docs/tokenizer.md section 9). It
// ports `rustbpe`'s `train_core_incremental`: the chunk counts and the merge
// loop. The split and count step uses OpenMP, because the reduction is
// commutative. The merge loop stays single-threaded, so the merge list does
// not depend on thread order. The pair counts use `int64` (decision D5).

#include "nanochat/bpe_trainer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"

namespace nanochat {
namespace {

// One token pair: the left id first and the right id second.
using Pair = std::pair<int, int>;

// A token pair that maps to a token id in the hash map.
struct PairHash {
  std::size_t operator()(const Pair& pair) const {
    return (static_cast<std::size_t>(static_cast<std::uint32_t>(pair.first))
            << 32) ^
           static_cast<std::size_t>(static_cast<std::uint32_t>(pair.second));
  }
};

// One heap entry: a pair count and the chunk indices where the pair occurs.
// The count is stale on purpose; the merge loop refreshes it with a lazy
// update, the same way `rustbpe` does.
struct MergeJob {
  Pair pair;
  std::int64_t count = 0;
  std::vector<std::size_t> positions;
};

// Max-heap by count; a tie breaks on the smallest pair. The left id compares
// first and the right id second, so the merge order is deterministic.
struct MergeJobLess {
  bool operator()(const MergeJob& left, const MergeJob& right) const {
    if (left.count != right.count) return left.count < right.count;
    return left.pair > right.pair;
  }
};

// A counted change of one pair in one chunk: +1 for a created pair and -1 for
// a removed pair.
struct PairDelta {
  Pair pair;
  int delta = 0;
};

void SortUnique(std::vector<std::size_t>* values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

// Replaces every non-overlapping occurrence of `pair` with `new_id` in `ids`.
// The method returns the local pair-count deltas of this chunk. It mirrors
// `Word::merge_pair` in `rustbpe`, including the use of the output tail.
std::vector<PairDelta> MergePairInWord(std::vector<int>* ids, Pair pair,
                                       int new_id) {
  const std::size_t size = ids->size();
  std::vector<PairDelta> deltas;
  if (size < 2) return deltas;
  std::vector<int> out;
  out.reserve(size);
  const int left_id = pair.first;
  const int right_id = pair.second;
  std::size_t index = 0;
  while (index < size) {
    if (index + 1 < size && (*ids)[index] == left_id &&
        (*ids)[index + 1] == right_id) {
      const bool has_left = !out.empty();
      const int before = has_left ? out.back() : 0;
      const bool has_right = index + 2 < size;
      const int after = has_right ? (*ids)[index + 2] : 0;
      if (has_left) {
        deltas.push_back({{before, left_id}, -1});
        deltas.push_back({{before, new_id}, 1});
      }
      deltas.push_back({{left_id, right_id}, -1});
      if (has_right) {
        deltas.push_back({{right_id, after}, -1});
        deltas.push_back({{new_id, after}, 1});
      }
      out.push_back(new_id);
      index += 2;
    } else {
      out.push_back((*ids)[index]);
      index += 1;
    }
  }
  *ids = std::move(out);
  return deltas;
}

}  // namespace

BpeTrainer::BpeTrainer(std::string pattern) : pattern_(std::move(pattern)) {}

void BpeTrainer::AddDocument(std::string_view text) {
  const std::vector<std::string_view> pieces = SplitPattern(text);
  if (pieces.empty()) return;

  // The count reduction is commutative, so the split pieces can be counted in
  // parallel. Per-thread maps keep the map updates off the critical path.
  const int thread_count = omp_get_max_threads();
  if (thread_count <= 1 || pieces.size() < 32) {
    for (std::string_view piece : pieces) {
      chunk_counts_[std::string(piece)] += 1;
    }
    return;
  }

  std::vector<std::unordered_map<std::string, std::int64_t>> local_counts(
      static_cast<std::size_t>(thread_count));
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    std::unordered_map<std::string, std::int64_t>& local =
        local_counts[static_cast<std::size_t>(thread)];
#pragma omp for schedule(static)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(pieces.size()); ++index) {
      local[std::string(pieces[static_cast<std::size_t>(index)])] += 1;
    }
  }
  for (const auto& local : local_counts) {
    for (const auto& [chunk, count] : local) {
      chunk_counts_[chunk] += count;
    }
  }
}

void BpeTrainer::AddChunk(std::string_view chunk, std::int64_t count) {
  if (chunk.empty() || count == 0) return;
  chunk_counts_[std::string(chunk)] += count;
}

void BpeTrainer::Train(int base_vocab_size) {
  merges_.clear();
  if (base_vocab_size <= 256) return;
  const std::int64_t num_merges =
      static_cast<std::int64_t>(base_vocab_size) - 256;

  // Materialize the unique chunks. The `std::map` order is deterministic, so
  // the chunk index does not depend on the ingestion order.
  std::vector<std::vector<int>> words;
  std::vector<std::int64_t> counts;
  words.reserve(chunk_counts_.size());
  counts.reserve(chunk_counts_.size());
  for (const auto& [chunk, count] : chunk_counts_) {
    if (count == 0) continue;
    std::vector<int> ids(chunk.size());
    for (std::size_t index = 0; index < chunk.size(); ++index) {
      ids[index] = static_cast<unsigned char>(chunk[index]);
    }
    words.push_back(std::move(ids));
    counts.push_back(count);
  }

  // The initial pair counts and the positions where each pair occurs. The
  // parallel `rustbpe` reduction is commutative, so a sequential loop gives
  // the same counts and a deterministic position order.
  std::unordered_map<Pair, std::int64_t, PairHash> pair_counts;
  std::unordered_map<Pair, std::vector<std::size_t>, PairHash> where_to_update;
  for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
    const std::vector<int>& ids = words[word_index];
    if (ids.size() < 2 || counts[word_index] == 0) continue;
    for (std::size_t index = 0; index + 1 < ids.size(); ++index) {
      const Pair pair = {ids[index], ids[index + 1]};
      pair_counts[pair] += counts[word_index];
      where_to_update[pair].push_back(word_index);
    }
  }

  std::priority_queue<MergeJob, std::vector<MergeJob>, MergeJobLess> heap;
  for (auto& [pair, positions] : where_to_update) {
    const auto it = pair_counts.find(pair);
    if (it == pair_counts.end() || it->second <= 0) continue;
    SortUnique(&positions);
    heap.push(MergeJob{pair, it->second, std::move(positions)});
  }

  std::int64_t merges_done = 0;
  while (merges_done < num_merges) {
    if (heap.empty()) break;
    MergeJob top = heap.top();
    heap.pop();

    const auto current_it = pair_counts.find(top.pair);
    const std::int64_t current =
        current_it == pair_counts.end() ? 0 : current_it->second;
    if (current <= 0) continue;
    if (top.count != current) {
      top.count = current;
      heap.push(std::move(top));
      continue;
    }

    const int new_id = 256 + static_cast<int>(merges_done);
    merges_.push_back(top.pair);

    std::unordered_map<Pair, std::vector<std::size_t>, PairHash> local_updates;
    for (std::size_t word_index : top.positions) {
      const std::vector<PairDelta> changes =
          MergePairInWord(&words[word_index], top.pair, new_id);
      for (const PairDelta& change : changes) {
        const std::int64_t delta_total =
            static_cast<std::int64_t>(change.delta) * counts[word_index];
        if (delta_total == 0) continue;
        pair_counts[change.pair] += delta_total;
        if (change.delta > 0) {
          local_updates[change.pair].push_back(word_index);
        }
      }
    }
    for (auto& [pair, positions] : local_updates) {
      const auto it = pair_counts.find(pair);
      if (it == pair_counts.end() || it->second <= 0) continue;
      SortUnique(&positions);
      heap.push(MergeJob{pair, it->second, std::move(positions)});
    }
    ++merges_done;
  }
}

std::vector<std::pair<std::vector<std::uint8_t>, int>>
BpeTrainer::MergeableRanks() const {
  // The token bytes of the 256 byte tokens, then one entry per merge. The rank
  // of merge `index` is `256 + index`, so the merge index gives the token id.
  std::vector<std::vector<std::uint8_t>> token_bytes(256);
  for (int id = 0; id < 256; ++id) {
    token_bytes[static_cast<std::size_t>(id)] = {static_cast<std::uint8_t>(id)};
  }
  std::vector<std::pair<std::vector<std::uint8_t>, int>> merges;
  merges.reserve(merges_.size());
  for (std::size_t index = 0; index < merges_.size(); ++index) {
    const int left = merges_[index].first;
    const int right = merges_[index].second;
    if (left < 0 || right < 0 ||
        static_cast<std::size_t>(left) >= token_bytes.size() ||
        static_cast<std::size_t>(right) >= token_bytes.size()) {
      continue;
    }
    std::vector<std::uint8_t> bytes =
        token_bytes[static_cast<std::size_t>(left)];
    const std::vector<std::uint8_t>& tail =
        token_bytes[static_cast<std::size_t>(right)];
    bytes.insert(bytes.end(), tail.begin(), tail.end());
    token_bytes.push_back(bytes);
    merges.emplace_back(std::move(bytes), 256 + static_cast<int>(index));
  }
  return merges;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> RecoverMergePairs(
    const std::vector<std::pair<std::vector<std::uint8_t>, int>>& merges) {
  std::map<std::string, int> rank_by_bytes;
  for (int id = 0; id < 256; ++id) {
    rank_by_bytes[std::string(1, static_cast<char>(id))] = id;
  }
  for (const auto& [bytes, rank] : merges) {
    rank_by_bytes[std::string(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size())] = rank;
  }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
  pairs.reserve(merges.size());
  for (const auto& [bytes, rank] : merges) {
    const std::string token(reinterpret_cast<const char*>(bytes.data()),
                            bytes.size());
    std::pair<int, int> best = {-1, -1};
    for (std::size_t split = 1; split < token.size(); ++split) {
      const auto left = rank_by_bytes.find(token.substr(0, split));
      const auto right = rank_by_bytes.find(token.substr(split));
      if (left == rank_by_bytes.end() || right == rank_by_bytes.end()) {
        continue;
      }
      if (left->second >= rank || right->second >= rank) continue;
      const std::pair<int, int> candidate = {left->second, right->second};
      if (best.first < 0 || candidate < best) best = candidate;
    }
    if (best.first < 0) {
      pairs.emplace_back(0, 0);
    } else {
      pairs.emplace_back(static_cast<std::uint32_t>(best.first),
                         static_cast<std::uint32_t>(best.second));
    }
  }
  return pairs;
}

}  // namespace nanochat
