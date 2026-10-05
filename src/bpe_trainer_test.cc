// src/bpe_trainer_test.cc -- the reference merge list.
//
// Trains the native trainer on the fixed reference corpus and compares the
// ordered merge list against `tests/data/tokenizer_fixture.bin` rank by rank.
// The fixture is data, so this test never imports Python. The corpus literal
// below is the `CORPUS` list of `tools/dump_tokenizer_fixture.py`; the merge
// comparison fails if the two corpora differ.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/bpe_trainer.h"
#include "src/tokenizer_internal.h"
#include "tests/oracle_fixture.h"

namespace {

using nanochat::oracle::Fixture;
using nanochat::oracle::Tensor;

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

// The fixed reference corpus, in order. It is the `CORPUS` list of
// `tools/dump_tokenizer_fixture.py`.
const char* const kCorpus[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
    "The five boxing wizards jump quickly.",
    "Bright vixens jump; dozy fowl quack.",
    "Jinxed wizards pluck ivy from the big quilt.",
    "Waltz, bad nymph, for quick jigs vex.",
    "Quick zephyrs blow, vexing daft Jim.",
    "Two driven jocks help fax my big quiz.",
    "It was a bright cold day in April, and the clocks were striking "
    "thirteen.",
    "All happy families are alike; each unhappy family is unhappy in its own "
    "way.",
    "Call me Ishmael. Some years ago, never mind how long precisely.",
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife.",
    "The sky above the port was the color of television, tuned to a dead "
    "channel.",
    "I've seen things you people wouldn't believe.",
    "Don't panic. The answer is 42, not 1234.",
    "We're going to need a bigger boat, aren't we?",
    "She said, \"Hello, World!\" and waved. Numbers: 1 22 333 4444.",
    "cafe naive resume jalapeno cooperate",
    "The value is 3.14 and 2,718, plus 100% of it.",
    "Line one\nline two\r\nline three\n",
    "   leading spaces and trailing spaces   ",
    "Tabs\tand\tspaces\t mixed together.",
    "Symbols: @#$%^&*()_+-=[]{}|;:',.<>/?`~",
    "def train_model(x, y): return x + y  # comment",
    "for i in range(10): print(i * 2)",
    "The rain in Spain stays mainly in the plain.",
    "Peter Piper picked a peck of pickled peppers.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "A man, a plan, a canal: Panama!",
    "Was it a car or a cat I saw?",
    "Never odd or even. Madam, I'm Adam.",
    "The llama is a quadruped, related to the camel.",
    "Pack my box with five dozen liquor jugs, again.",
};

std::string Join(const std::vector<std::string>& parts) {
  std::string joined;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) joined += ", ";
    joined += parts[i];
  }
  return joined;
}

std::string LocateFixture(int argc, char** argv) {
  std::vector<std::string> candidates;
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    candidates.emplace_back(argv[1]);
  }
  if (const char* src_dir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      candidates.emplace_back(std::string(src_dir) + "/" + workspace +
                              "/tests/data/tokenizer_fixture.bin");
    }
    candidates.emplace_back(std::string(src_dir) +
                            "/_main/tests/data/tokenizer_fixture.bin");
    candidates.emplace_back(std::string(src_dir) +
                            "/nanochat_cpp/tests/data/tokenizer_fixture.bin");
  }
  candidates.emplace_back("tests/data/tokenizer_fixture.bin");
  candidates.emplace_back("nanochat.cpp/tests/data/tokenizer_fixture.bin");
  candidates.emplace_back("../tests/data/tokenizer_fixture.bin");
  for (const std::string& candidate : candidates) {
    std::ifstream in(candidate, std::ios::binary);
    if (in.good()) return candidate;
  }
  throw std::runtime_error("tokenizer fixture not found; tried: " +
                           Join(candidates));
}

std::vector<int> Ints(const Fixture& fixture, const std::string& name) {
  const Tensor& tensor = fixture.Get(name);
  const std::int32_t* data = tensor.i32();
  return std::vector<int>(data, data + tensor.numel());
}

std::string Bytes(const Fixture& fixture, const std::string& name) {
  const Tensor& tensor = fixture.Get(name);
  return std::string(reinterpret_cast<const char*>(tensor.data), tensor.bytes);
}

std::string Slice(const std::string& bytes, const std::vector<int>& offsets,
                  std::size_t index) {
  const int begin = offsets[index];
  const int end = offsets[index + 1];
  return bytes.substr(static_cast<std::size_t>(begin),
                      static_cast<std::size_t>(end - begin));
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  try {
    path = LocateFixture(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  Fixture fixture;
  try {
    fixture = Fixture::Load(path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }

  const std::vector<int> config_base = Ints(fixture, "config/base_vocab_size");
  const std::vector<int> config_merges = Ints(fixture, "config/merge_count");
  if (config_merges[0] != config_base[0] - 256) {
    Fail("fixture merge count is inconsistent");
  }

  nanochat::BpeTrainer trainer;
  for (const char* document : kCorpus) {
    trainer.AddDocument(document);
  }
  trainer.Train(config_base[0]);

  const std::vector<std::pair<std::vector<std::uint8_t>, int>> merges =
      trainer.MergeableRanks();
  if (static_cast<int>(merges.size()) != config_merges[0]) {
    Fail("merge count mismatch: port " + std::to_string(merges.size()) +
         " fixture " + std::to_string(config_merges[0]));
  }

  const std::string merge_bytes = Bytes(fixture, "merges/token_bytes");
  const std::vector<int> merge_offsets =
      Ints(fixture, "merges/token_bytes_offsets");
  const std::vector<int> merge_ranks = Ints(fixture, "merges/ranks");
  const int common =
      std::min(static_cast<int>(merges.size()), config_merges[0]);
  for (int index = 0; index < common; ++index) {
    const std::string expected =
        Slice(merge_bytes, merge_offsets, static_cast<std::size_t>(index));
    const auto& [bytes, rank] = merges[static_cast<std::size_t>(index)];
    const std::string actual(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
    if (actual != expected) {
      Fail("merge " + std::to_string(index) + " bytes mismatch");
    }
    if (rank != merge_ranks[static_cast<std::size_t>(index)]) {
      Fail("merge " + std::to_string(index) + " rank mismatch");
    }
  }

  const std::vector<std::pair<std::uint32_t, std::uint32_t>> recovered =
      nanochat::RecoverMergePairs(merges);
  const std::vector<int> merge_left = Ints(fixture, "merges/left");
  const std::vector<int> merge_right = Ints(fixture, "merges/right");
  for (int index = 0; index < common; ++index) {
    const std::pair<std::uint32_t, std::uint32_t> expected = {
        static_cast<std::uint32_t>(merge_left[static_cast<std::size_t>(index)]),
        static_cast<std::uint32_t>(
            merge_right[static_cast<std::size_t>(index)])};
    if (recovered[static_cast<std::size_t>(index)] != expected) {
      Fail("merge " + std::to_string(index) + " pair mismatch");
    }
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "bpe_trainer_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("bpe_trainer_test: ok\n");
  return 0;
}
