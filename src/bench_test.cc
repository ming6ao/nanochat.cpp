// The benchmark report's optional row keys (include/nanochat/bench.h).
//
// Two facts travel with a measured row. They are the iteration count the
// per-call time came from, and the occupancy the arm answers. An older row
// states neither, and the schema had no place for either. A reader of a
// `nanochat.bench.v3` document must be able to tell a row that states them from
// a row that does not. The writer therefore omits both keys rather than writing
// a zero (docs/kernels.md section 11.7).
//
// The test drives the writer and reads its document back as text. It asserts no
// formatting beyond the two keys, because the schema's field order belongs to
// `bench.h` and not to this test.

#include <cstdio>
#include <string>

#include "nanochat/bench.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

// True when `text` holds `key` as a JSON object member.
bool HasKey(const std::string& text, const std::string& key) {
  return text.find("\"" + key + "\":") != std::string::npos;
}

void TestKeysAreOptional() {
  nanochat::BenchReport report;
  report.Add("gemm:muon:1280x5120:prod:wide-n:cutlass", "M=1280 N=5120 K=1280",
             0.5, 2000.0);

  const std::string text = report.Json();
  Check(!HasKey(text, "iterations"),
        "an unstated iteration count writes no key");
  Check(!HasKey(text, "occupancy"), "an unstated occupancy writes no key");
}

void TestKeysAreWritten() {
  nanochat::BenchReport report;
  report.Add("gemm:muon:1280x5120:prod:wide-n:cutlass", "M=1280 N=5120 K=1280",
             0.5, 2000.0, /*work_ratio=*/0.0, /*iterations=*/7,
             /*occupancy=*/2);

  const std::string text = report.Json();
  Check(HasKey(text, "iterations"), "a stated iteration count writes its key");
  Check(HasKey(text, "occupancy"), "a stated occupancy writes its key");
  Check(text.find("\"iterations\":7") != std::string::npos,
        "the iteration count is written as an integer");
  Check(text.find("\"occupancy\":2") != std::string::npos,
        "the occupancy is written as an integer");
  Check(text.find("\"schema\":\"nanochat.bench.v3\"") != std::string::npos,
        "the schema name is unchanged");
}

// The median row of one measurement carries the same two facts. A reader joins
// the pair, and it would otherwise compare a shape against nothing.
void TestMedianRowCarriesThem() {
  nanochat::BenchReport report;
  report.Add("seam", "M=1280 N=5120 K=1280", 0.5, 2000.0, 0.0, 3, 2);
  const std::string text = report.Json();
  const std::size_t rows = text.find("\"rows\":[");
  Check(rows != std::string::npos, "the document has a row array");
  Check(text.find("\"iterations\":3", rows) != std::string::npos,
        "the row states its iteration count");
}

}  // namespace

int main() {
  TestKeysAreOptional();
  TestKeysAreWritten();
  TestMedianRowCarriesThem();
  if (g_failures != 0) {
    std::printf("bench: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("bench: all checks passed\n");
  return 0;
}
