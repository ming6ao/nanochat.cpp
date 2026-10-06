// src/parquet/levels_test.cc -- the RLE and bit-packed hybrid decoder.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/parquet/levels.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

bool Expect(const std::vector<std::uint32_t>& got,
            const std::vector<std::uint32_t>& want, const std::string& name) {
  if (got == want) return true;
  Fail(name + ": value mismatch");
  return false;
}

void CheckRepeatedRun() {
  // One RLE run: five copies of 1. Header (5 << 1) = 0x0A, value 0x01.
  const std::uint8_t bytes[] = {0x0a, 0x01};
  std::vector<std::uint32_t> out;
  std::string error;
  if (!nanochat::parquet::DecodeRleBitPacked(bytes, sizeof(bytes), 1, 5, &out,
                                             &error)) {
    Fail("repeated run: " + error);
    return;
  }
  Expect(out, {1, 1, 1, 1, 1}, "repeated run");
}

void CheckBitPackedRun() {
  // One bit-packed group: values 1,0,1,0,1,0,1,0 -> 0x55.
  const std::uint8_t bytes[] = {0x03, 0x55};
  std::vector<std::uint32_t> out;
  std::string error;
  if (!nanochat::parquet::DecodeRleBitPacked(bytes, sizeof(bytes), 1, 8, &out,
                                             &error)) {
    Fail("bit-packed run: " + error);
    return;
  }
  Expect(out, {1, 0, 1, 0, 1, 0, 1, 0}, "bit-packed run");
}

void CheckMixedRuns() {
  // Two RLE copies of 0, then one bit-packed group of 0x55.
  const std::uint8_t bytes[] = {0x04, 0x00, 0x03, 0x55};
  std::vector<std::uint32_t> out;
  std::string error;
  if (!nanochat::parquet::DecodeRleBitPacked(bytes, sizeof(bytes), 1, 10, &out,
                                             &error)) {
    Fail("mixed runs: " + error);
    return;
  }
  Expect(out, {0, 0, 1, 0, 1, 0, 1, 0, 1, 0}, "mixed runs");
}

void CheckTwoBitWidth() {
  // One bit-packed group of values 0,1,2,3,0,1,2,3 at two bits each.
  const std::uint8_t bytes[] = {0x03, 0xe4, 0xe4};
  std::vector<std::uint32_t> out;
  std::string error;
  if (!nanochat::parquet::DecodeRleBitPacked(bytes, sizeof(bytes), 2, 8, &out,
                                             &error)) {
    Fail("two-bit width: " + error);
    return;
  }
  Expect(out, {0, 1, 2, 3, 0, 1, 2, 3}, "two-bit width");
}

void CheckZeroBitWidth() {
  std::vector<std::uint32_t> out;
  std::string error;
  if (!nanochat::parquet::DecodeRleBitPacked(nullptr, 0, 0, 4, &out, &error)) {
    Fail("zero-bit width: " + error);
    return;
  }
  Expect(out, {0, 0, 0, 0}, "zero-bit width");
}

void CheckTruncation() {
  // One bit-packed group needs one payload byte; the buffer stops early.
  const std::uint8_t bytes[] = {0x03};
  std::vector<std::uint32_t> out;
  std::string error;
  if (nanochat::parquet::DecodeRleBitPacked(bytes, sizeof(bytes), 1, 8, &out,
                                            &error)) {
    Fail("truncation: decoder accepted a short buffer");
  } else if (error.empty()) {
    Fail("truncation: no error message");
  }
}

}  // namespace

int main() {
  CheckRepeatedRun();
  CheckBitPackedRun();
  CheckMixedRuns();
  CheckTwoBitWidth();
  CheckZeroBitWidth();
  CheckTruncation();
  if (g_failures != 0) {
    std::fprintf(stderr, "levels_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("levels_test: ok\n");
  return 0;
}
