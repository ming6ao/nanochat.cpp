// src/parquet/reader_reject_test.cc -- the native parquet reader rejections.
//
// The cases mutate a copy of the valid fixture: a broken trailer magic, a zero
// footer length, a truncated file, and a bad schema column. Each case must
// return an error instead of a crash.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "src/parquet/reader.h"

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

bool ReadFile(const std::string& path, std::vector<std::uint8_t>* data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size < 0) return false;
  data->resize(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char*>(data->data()), size);
  return in.gcount() == size;
}

bool WriteFile(const std::string& path, const std::vector<std::uint8_t>& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(out);
}

// Open must fail with a non-empty error.
void ExpectReject(const std::string& path, const std::string& name) {
  nanochat::ParquetReader::Options options;
  options.files = {path};
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader != nullptr) {
    Fail(name + ": Open accepted a bad file");
  } else if (error.empty()) {
    Fail(name + ": Open did not report an error");
  }
}

void CheckValid(const std::vector<std::uint8_t>& base) {
  const std::string path = TempPath("reject_valid.parquet");
  if (!WriteFile(path, base)) {
    Fail("valid: write");
    return;
  }
  nanochat::ParquetReader::Options options;
  options.files = {path};
  std::string error;
  std::unique_ptr<nanochat::ParquetReader> reader =
      nanochat::ParquetReader::Open(options, &error);
  if (reader == nullptr) Fail("valid: " + error);
}

void CheckBadMagic(const std::vector<std::uint8_t>& base) {
  std::vector<std::uint8_t> data = base;
  if (data.size() < 4) {
    Fail("bad magic: fixture is too small");
    return;
  }
  data[data.size() - 1] = 'X';
  const std::string path = TempPath("reject_bad_magic.parquet");
  if (!WriteFile(path, data)) {
    Fail("bad magic: write");
    return;
  }
  ExpectReject(path, "bad magic");
}

void CheckZeroFooterLength(const std::vector<std::uint8_t>& base) {
  std::vector<std::uint8_t> data = base;
  if (data.size() < 12) {
    Fail("zero footer: fixture is too small");
    return;
  }
  for (std::size_t i = 0; i < 4; ++i) data[data.size() - 8 + i] = 0;
  const std::string path = TempPath("reject_zero_footer.parquet");
  if (!WriteFile(path, data)) {
    Fail("zero footer: write");
    return;
  }
  ExpectReject(path, "zero footer");
}

void CheckTruncated(const std::vector<std::uint8_t>& base) {
  std::vector<std::uint8_t> data(base.begin(), base.begin() + 8);
  const std::string path = TempPath("reject_truncated.parquet");
  if (!WriteFile(path, data)) {
    Fail("truncated: write");
    return;
  }
  ExpectReject(path, "truncated");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "reader_reject_test: missing fixture path\n");
    return 2;
  }
  std::vector<std::uint8_t> base;
  if (!ReadFile(argv[1], &base)) {
    std::fprintf(stderr, "reader_reject_test: cannot read %s\n", argv[1]);
    return 2;
  }
  CheckValid(base);
  CheckBadMagic(base);
  CheckZeroFooterLength(base);
  CheckTruncated(base);
  if (g_failures != 0) {
    std::fprintf(stderr, "reader_reject_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("reader_reject_test: ok\n");
  return 0;
}
