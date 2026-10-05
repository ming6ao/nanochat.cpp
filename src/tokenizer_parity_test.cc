// src/tokenizer_parity_test.cc -- the reference tokenizer fixture.
//
// Builds an `NCTOKEN1` artifact from `tests/data/tokenizer_fixture.bin`, loads
// it, and compares the merges, the token bytes, the split cases, the encode
// cases, the decode cases, and the stream decoder against the recorded
// reference. The fixture is data, so this test never imports Python.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/tokenizer.h"
#include "src/split_pattern.h"
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

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

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

void CheckSplitCases(const Fixture& fixture) {
  const std::string strings = Bytes(fixture, "split/strings");
  const std::vector<int> string_offsets =
      Ints(fixture, "split/strings_offsets");
  const std::vector<int> string_piece_offsets =
      Ints(fixture, "split/string_piece_offsets");
  const std::vector<int> piece_offsets = Ints(fixture, "split/piece_offsets");
  const std::string pieces = Bytes(fixture, "split/pieces");
  for (std::size_t string_index = 0; string_index + 1 < string_offsets.size();
       ++string_index) {
    const std::string text = Slice(strings, string_offsets, string_index);
    std::vector<std::string> expected;
    for (int piece_index = string_piece_offsets[string_index];
         piece_index < string_piece_offsets[string_index + 1]; ++piece_index) {
      expected.push_back(
          Slice(pieces, piece_offsets, static_cast<std::size_t>(piece_index)));
    }
    const std::vector<std::string_view> actual = nanochat::SplitPattern(text);
    if (actual.size() != expected.size()) {
      Fail("split size mismatch for '" + text + "'");
      continue;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (actual[i] != expected[i]) {
        Fail("split mismatch for '" + text + "' at piece " + std::to_string(i));
      }
    }
  }
}

void CheckEncodeCases(const nanochat::Tokenizer& tokenizer,
                      const Fixture& fixture) {
  const std::string inputs = Bytes(fixture, "encode/inputs");
  const std::vector<int> input_offsets = Ints(fixture, "encode/input_offsets");
  const std::vector<int> prepend = Ints(fixture, "encode/prepend");
  const std::vector<int> append = Ints(fixture, "encode/append");
  const std::vector<int> id_offsets = Ints(fixture, "encode/id_offsets");
  const std::vector<int> ids = Ints(fixture, "encode/ids");
  for (std::size_t case_index = 0; case_index + 1 < input_offsets.size();
       ++case_index) {
    const std::string text = Slice(inputs, input_offsets, case_index);
    std::vector<int> expected;
    for (int i = id_offsets[case_index]; i < id_offsets[case_index + 1]; ++i) {
      expected.push_back(ids[static_cast<std::size_t>(i)]);
    }
    const std::vector<int> actual =
        tokenizer.Encode(text, prepend[case_index], append[case_index]);
    if (actual != expected) {
      Fail("encode mismatch for '" + text + "'");
    }
  }
}

void CheckDecodeCases(const nanochat::Tokenizer& tokenizer,
                      const Fixture& fixture) {
  const std::vector<int> id_offsets = Ints(fixture, "decode/id_offsets");
  const std::vector<int> ids = Ints(fixture, "decode/ids");
  const std::string outputs = Bytes(fixture, "decode/outputs");
  const std::vector<int> output_offsets =
      Ints(fixture, "decode/output_offsets");
  for (std::size_t case_index = 0; case_index + 1 < id_offsets.size();
       ++case_index) {
    std::vector<int> row;
    for (int i = id_offsets[case_index]; i < id_offsets[case_index + 1]; ++i) {
      row.push_back(ids[static_cast<std::size_t>(i)]);
    }
    const std::string expected = Slice(outputs, output_offsets, case_index);
    const std::string actual =
        tokenizer.Decode(row.data(), static_cast<int>(row.size()));
    if (actual != expected) {
      Fail("decode mismatch for case " + std::to_string(case_index));
    }
    nanochat::TokenStreamDecoder decoder(tokenizer);
    std::string streamed;
    for (int id : row) decoder.Push(id, &streamed);
    decoder.Flush(&streamed);
    if (streamed != expected) {
      Fail("stream decode mismatch for case " + std::to_string(case_index));
    }
  }
}

void CheckArtifact(const nanochat::Tokenizer& tokenizer,
                   const Fixture& fixture) {
  const std::vector<int> config_vocab = Ints(fixture, "config/vocab_size");
  const std::vector<int> config_base = Ints(fixture, "config/base_vocab_size");
  const std::vector<int> config_merges = Ints(fixture, "config/merge_count");
  if (tokenizer.vocab_size() != config_vocab[0]) {
    Fail("vocab_size does not match the fixture");
  }
  if (config_merges[0] != config_base[0] - 256) {
    Fail("fixture merge count is inconsistent");
  }

  const std::string merge_bytes = Bytes(fixture, "merges/token_bytes");
  const std::vector<int> merge_offsets =
      Ints(fixture, "merges/token_bytes_offsets");
  for (int index = 0; index < config_merges[0]; ++index) {
    std::string actual;
    if (!tokenizer.AppendTokenBytes(256 + index, &actual)) {
      Fail("AppendTokenBytes rejected a merge id");
      continue;
    }
    const std::string expected =
        Slice(merge_bytes, merge_offsets, static_cast<std::size_t>(index));
    if (actual != expected) {
      Fail("merge " + std::to_string(index) + " bytes mismatch");
    }
  }

  const std::string token_bytes = Bytes(fixture, "tokens/bytes");
  const std::vector<int> token_offsets = Ints(fixture, "tokens/bytes_offsets");
  for (int id = 0; id < config_base[0]; ++id) {
    std::string actual;
    if (!tokenizer.AppendTokenBytes(id, &actual)) {
      Fail("AppendTokenBytes rejected a base id");
      continue;
    }
    const std::string expected =
        Slice(token_bytes, token_offsets, static_cast<std::size_t>(id));
    if (actual != expected) {
      Fail("base token " + std::to_string(id) + " bytes mismatch");
    }
  }

  const std::string lengths = Bytes(fixture, "token_bytes/lengths");
  const std::uint8_t* token_lengths = tokenizer.TokenBytes();
  for (int id = 0; id < config_vocab[0]; ++id) {
    if (token_lengths[id] !=
        static_cast<std::uint8_t>(lengths[static_cast<std::size_t>(id)])) {
      Fail("token length mismatch for id " + std::to_string(id));
    }
  }

  const std::string names = Bytes(fixture, "special/names");
  const std::vector<int> name_offsets = Ints(fixture, "special/names_offsets");
  const std::vector<int> special_ids = Ints(fixture, "special/ids");
  for (std::size_t index = 0; index < special_ids.size(); ++index) {
    const std::string name = Slice(names, name_offsets, index);
    const int id = special_ids[index];
    if (tokenizer.SpecialId(name) != id) {
      Fail("SpecialId mismatch for '" + name + "'");
    }
    if (!tokenizer.IsSpecial(id)) {
      Fail("IsSpecial is false for '" + name + "'");
    }
    std::string bytes;
    if (!tokenizer.AppendTokenBytes(id, &bytes) || bytes != name) {
      Fail("special bytes mismatch for '" + name + "'");
    }
  }
  if (tokenizer.bos_id() != special_ids[0]) {
    Fail("bos_id does not match the fixture");
  }
  if (tokenizer.IsSpecial(0) || tokenizer.IsSpecial(config_base[0] - 1)) {
    Fail("IsSpecial is true for a base token");
  }
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

  const std::string pattern = Bytes(fixture, "config/pattern");
  const std::vector<int> merge_left = Ints(fixture, "merges/left");
  const std::vector<int> merge_right = Ints(fixture, "merges/right");
  std::vector<std::pair<std::uint32_t, std::uint32_t>> merge_pairs;
  for (std::size_t index = 0; index < merge_left.size(); ++index) {
    merge_pairs.emplace_back(static_cast<std::uint32_t>(merge_left[index]),
                             static_cast<std::uint32_t>(merge_right[index]));
  }
  const std::string names = Bytes(fixture, "special/names");
  const std::vector<int> name_offsets = Ints(fixture, "special/names_offsets");
  const std::vector<int> special_ids = Ints(fixture, "special/ids");
  std::vector<std::pair<std::string, std::uint32_t>> specials;
  for (std::size_t index = 0; index < special_ids.size(); ++index) {
    specials.emplace_back(Slice(names, name_offsets, index),
                          static_cast<std::uint32_t>(special_ids[index]));
  }

  const std::string artifact = TempPath("parity.nctoken");
  if (!nanochat::SaveTokenizer(artifact, pattern, merge_pairs, specials)) {
    std::fprintf(stderr, "FAIL: could not write the parity artifact\n");
    return 1;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(artifact);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "FAIL: could not load the parity artifact\n");
    return 1;
  }

  CheckArtifact(*tokenizer, fixture);
  CheckSplitCases(fixture);
  CheckEncodeCases(*tokenizer, fixture);
  CheckDecodeCases(*tokenizer, fixture);

  if (g_failures != 0) {
    std::fprintf(stderr, "tokenizer_parity_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("tokenizer_parity_test: ok\n");
  return 0;
}
