// src/tokenizer/tokenizer_test.cc -- the native BPE tokenizer.
//
// Round-trips a tiny `NCTOKEN1` artifact, checks the special ids, the token
// bytes, the encoder, the lossy decoder, the stream decoder, and the
// bad-file contract of `LoadTokenizer`.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanochat/tokenizer.h"
#include "src/tokenizer/split_pattern.h"
#include "src/tokenizer/tokenizer_internal.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

std::string TempPath(const std::string& name) {
  const char* dir = std::getenv("TEST_TMPDIR");
  return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

void WriteRaw(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> TinyMerges() {
  return {
      {'h', 'e'},    // rank 256: "he"
      {'l', 'l'},    // rank 257: "ll"
      {256, 257},    // rank 258: "hell"
      {258, 'o'},    // rank 259: "hello"
      {0xE2, 0x82},  // rank 260: the first two bytes of the euro sign
  };
}

std::vector<std::pair<std::string, std::uint32_t>> TinySpecials() {
  return {{"<|bos|>", 261}, {"<|user_start|>", 262}};
}

std::unique_ptr<nanochat::Tokenizer> LoadTiny(const std::string& name) {
  const std::string path = TempPath(name);
  if (!nanochat::SaveTokenizer(path, nanochat::NanochatSplitPattern(),
                               TinyMerges(), TinySpecials())) {
    Fail("SaveTokenizer could not write the tiny artifact");
    return nullptr;
  }
  std::unique_ptr<nanochat::Tokenizer> tokenizer =
      nanochat::LoadTokenizer(path);
  if (tokenizer == nullptr) {
    Fail("LoadTokenizer returned null for the tiny artifact");
  }
  return tokenizer;
}

void TestMetadata(const nanochat::Tokenizer& tokenizer) {
  if (tokenizer.vocab_size() != 263) {
    Fail("vocab_size is not 263");
  }
  if (tokenizer.bos_id() != 261) {
    Fail("bos_id is not the <|bos|> id");
  }
  if (tokenizer.SpecialId("<|bos|>") != 261 ||
      tokenizer.SpecialId("<|user_start|>") != 262 ||
      tokenizer.SpecialId("<|nope|>") != -1) {
    Fail("SpecialId disagrees with the artifact");
  }
  if (!tokenizer.IsSpecial(261) || !tokenizer.IsSpecial(262) ||
      tokenizer.IsSpecial(260) || tokenizer.IsSpecial(-1) ||
      tokenizer.IsSpecial(263)) {
    Fail("IsSpecial disagrees with the artifact");
  }

  std::string bytes;
  if (!tokenizer.AppendTokenBytes(256, &bytes) || bytes != "he") {
    Fail("AppendTokenBytes(256) is not \"he\"");
  }
  bytes.clear();
  if (!tokenizer.AppendTokenBytes(259, &bytes) || bytes != "hello") {
    Fail("AppendTokenBytes(259) is not \"hello\"");
  }
  bytes.clear();
  if (!tokenizer.AppendTokenBytes(261, &bytes) || bytes != "<|bos|>") {
    Fail("AppendTokenBytes(261) is not the special name");
  }
  if (tokenizer.AppendTokenBytes(-1, &bytes) ||
      tokenizer.AppendTokenBytes(263, &bytes)) {
    Fail("AppendTokenBytes accepted a bad id");
  }

  const std::uint8_t* lengths = tokenizer.TokenBytes();
  if (lengths == nullptr || lengths[0] != 1 || lengths[256] != 2 ||
      lengths[258] != 4 || lengths[259] != 5 || lengths[260] != 2 ||
      lengths[261] != 0 || lengths[262] != 0) {
    Fail("TokenBytes disagrees with the artifact");
  }
}

void TestEncodeDecode(const nanochat::Tokenizer& tokenizer) {
  const std::vector<int> hello = tokenizer.Encode("hello");
  if (hello != std::vector<int>({259})) {
    Fail("Encode(\"hello\") is not the merge id");
  }
  const std::vector<int> prepended = tokenizer.Encode("hello", 261, -1);
  if (prepended != std::vector<int>({261, 259})) {
    Fail("Encode with a prepended special is wrong");
  }
  const std::vector<int> appended = tokenizer.Encode("hello", -1, 261);
  if (appended != std::vector<int>({259, 261})) {
    Fail("Encode with an appended special is wrong");
  }

  const std::vector<int> special_row = {261, 259, 262};
  if (tokenizer.Decode(special_row.data(),
                       static_cast<int>(special_row.size())) !=
      "<|bos|>hello<|user_start|>") {
    Fail("Decode with specials is wrong");
  }
  if (tokenizer.Decode(nullptr, 0) != "") {
    Fail("Decode of an empty row is not empty");
  }

  for (const std::string& text :
       {std::string("hello"), std::string("hello world"), std::string("abc"),
        std::string("")}) {
    const std::vector<int> ids = tokenizer.Encode(text);
    if (tokenizer.Decode(ids.data(), static_cast<int>(ids.size())) != text) {
      Fail("round trip failed for '" + text + "'");
    }
  }
}

void TestStreamDecoder(const nanochat::Tokenizer& tokenizer) {
  // A token that ends in the middle of a UTF-8 sequence must not emit a
  // replacement character when the next token completes the sequence.
  nanochat::TokenStreamDecoder decoder(tokenizer);
  std::string out;
  decoder.Push(260, &out);
  if (!out.empty()) {
    Fail("stream decoder emitted a partial sequence");
  }
  decoder.Push(0xAC, &out);
  if (out != "\xE2\x82\xAC") {
    Fail("stream decoder did not repair the partial sequence");
  }
  std::string flushed;
  decoder.Flush(&flushed);
  if (!flushed.empty()) {
    Fail("stream decoder flush after a repair is not empty");
  }

  nanochat::TokenStreamDecoder flushing(tokenizer);
  std::string partial;
  flushing.Push(260, &partial);
  if (!partial.empty()) {
    Fail("stream decoder emitted a partial sequence before flush");
  }
  flushing.Flush(&partial);
  if (partial != "\xEF\xBF\xBD") {
    Fail("stream decoder flush did not replace the partial sequence");
  }

  const std::vector<int> row = {261, 259, 262};
  nanochat::TokenStreamDecoder streamed(tokenizer);
  std::string streamed_text;
  for (int id : row) streamed.Push(id, &streamed_text);
  streamed.Flush(&streamed_text);
  if (streamed_text !=
      tokenizer.Decode(row.data(), static_cast<int>(row.size()))) {
    Fail("streamed row differs from Decode");
  }
}

void TestLossyDecode(const nanochat::Tokenizer& tokenizer) {
  const std::string replacement = "\xEF\xBF\xBD";
  const std::vector<int> invalid_lead = {0x80};
  if (tokenizer.Decode(invalid_lead.data(), 1) != replacement) {
    Fail("Decode did not replace an invalid lead byte");
  }
  const std::vector<int> out_of_range = {0xF5};
  if (tokenizer.Decode(out_of_range.data(), 1) != replacement) {
    Fail("Decode did not replace an out-of-range lead byte");
  }
  const std::vector<int> bad_second = {0xE2, 0x41};
  if (tokenizer.Decode(bad_second.data(), 2) != replacement + "A") {
    Fail("Decode did not replace an invalid continuation byte");
  }
  const std::vector<int> overlong = {0xC0, 0x80};
  if (tokenizer.Decode(overlong.data(), 2) != replacement + replacement) {
    Fail("Decode did not replace an overlong sequence");
  }
  const std::vector<int> incomplete = {0xE2, 0x82};
  if (tokenizer.Decode(incomplete.data(), 2) != replacement) {
    Fail("Decode did not replace an incomplete sequence");
  }
}

void TestBadFiles() {
  if (nanochat::LoadTokenizer(TempPath("missing.nctoken")) != nullptr) {
    Fail("LoadTokenizer accepted a missing file");
  }

  const std::string magic_only = TempPath("magic_only.nctoken");
  WriteRaw(magic_only, "NCTOKEN1");
  if (nanochat::LoadTokenizer(magic_only) != nullptr) {
    Fail("LoadTokenizer accepted a truncated header");
  }

  const std::string bad_magic = TempPath("bad_magic.nctoken");
  WriteRaw(bad_magic, std::string(16, '\0'));
  if (nanochat::LoadTokenizer(bad_magic) != nullptr) {
    Fail("LoadTokenizer accepted a bad magic");
  }

  std::string bad_version = "NCTOKEN1";
  bad_version.push_back('\x02');
  bad_version.append(3, '\0');
  bad_version.append(4, '\0');
  const std::string bad_version_path = TempPath("bad_version.nctoken");
  WriteRaw(bad_version_path, bad_version);
  if (nanochat::LoadTokenizer(bad_version_path) != nullptr) {
    Fail("LoadTokenizer accepted a bad version");
  }

  const std::string wrong_pattern = TempPath("wrong_pattern.nctoken");
  if (!nanochat::SaveTokenizer(wrong_pattern, "a different pattern",
                               TinyMerges(), TinySpecials())) {
    Fail("SaveTokenizer could not write the wrong-pattern artifact");
  }
  if (nanochat::LoadTokenizer(wrong_pattern) != nullptr) {
    Fail("LoadTokenizer accepted a different pattern");
  }

  const std::string bad_merge = TempPath("bad_merge.nctoken");
  if (!nanochat::SaveTokenizer(bad_merge, nanochat::NanochatSplitPattern(),
                               {{300, 0}}, TinySpecials())) {
    Fail("SaveTokenizer could not write the bad-merge artifact");
  }
  if (nanochat::LoadTokenizer(bad_merge) != nullptr) {
    Fail("LoadTokenizer accepted a forward merge reference");
  }
}

}  // namespace

int main() {
  std::unique_ptr<nanochat::Tokenizer> tokenizer = LoadTiny("tiny.nctoken");
  if (tokenizer != nullptr) {
    TestMetadata(*tokenizer);
    TestEncodeDecode(*tokenizer);
    TestLossyDecode(*tokenizer);
    TestStreamDecoder(*tokenizer);
  }
  TestBadFiles();
  if (g_failures != 0) {
    std::fprintf(stderr, "tokenizer_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("tokenizer_test: ok\n");
  return 0;
}
