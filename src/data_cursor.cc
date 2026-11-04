// src/data_cursor.cc -- the data-cursor sidecar
// (docs/distributed-design.md section 11, spec/api-surface.md R13).
//
// The layout moved here unchanged from `src/train.cc`, which wrote it before
// this unit existed, so a checkpoint written by an older run still resumes. A
// test in `//src:data_test` pins the bytes.

#include "src/data_cursor.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <string>
#include <vector>

namespace nanochat {
namespace {

void AppendLE64(std::vector<std::byte>* out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<std::byte>((value >> (8 * i)) & 0xffu));
  }
}

bool ReadLE64(std::istream& in, std::uint64_t* value) {
  std::byte bytes[8];
  in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(bytes))) {
    return false;
  }
  std::uint64_t result = 0;
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  *value = result;
  return true;
}

}  // namespace

std::string DataCursorPath(const std::string& checkpoint_path) {
  return checkpoint_path + kDataCursorSuffix;
}

bool WriteDataCursor(const std::string& checkpoint_path,
                     const DataLoaderState& state) {
  if (checkpoint_path.empty()) return false;
  const std::string path = DataCursorPath(checkpoint_path);
  std::vector<std::byte> bytes;
  bytes.reserve(24 + state.documents.size() * 8);
  AppendLE64(&bytes, static_cast<std::uint64_t>(state.epoch));
  AppendLE64(&bytes, static_cast<std::uint64_t>(state.source_position));
  AppendLE64(&bytes, static_cast<std::uint64_t>(state.documents.size()));
  for (const std::vector<int>& document : state.documents) {
    AppendLE64(&bytes, static_cast<std::uint64_t>(document.size()));
    for (int token : document) {
      AppendLE64(&bytes,
                 static_cast<std::uint64_t>(static_cast<std::int64_t>(token)));
    }
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

bool ReadDataCursor(const std::string& checkpoint_path,
                    DataLoaderState* state) {
  if (checkpoint_path.empty() || state == nullptr) return false;
  std::ifstream in(DataCursorPath(checkpoint_path), std::ios::binary);
  if (!in) return false;
  std::uint64_t epoch = 0;
  std::uint64_t position = 0;
  std::uint64_t count = 0;
  if (!ReadLE64(in, &epoch) || !ReadLE64(in, &position) ||
      !ReadLE64(in, &count)) {
    return false;
  }
  // A badly large count must not allocate without bound.
  if (count > (1ull << 30)) return false;
  std::vector<std::vector<int>> documents(static_cast<std::size_t>(count));
  for (std::vector<int>& document : documents) {
    std::uint64_t tokens = 0;
    if (!ReadLE64(in, &tokens)) return false;
    if (tokens > (1ull << 30)) return false;
    document.resize(static_cast<std::size_t>(tokens));
    for (int& token : document) {
      std::uint64_t raw = 0;
      if (!ReadLE64(in, &raw)) return false;
      token = static_cast<int>(static_cast<std::int64_t>(raw));
    }
  }
  state->epoch = static_cast<std::int64_t>(epoch);
  state->source_position = static_cast<std::int64_t>(position);
  state->documents = std::move(documents);
  return true;
}

}  // namespace nanochat
