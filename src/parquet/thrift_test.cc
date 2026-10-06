// src/parquet/thrift_test.cc -- the Thrift compact decoder.
//
// The cases are hand-encoded structs: scalar fields, the field-id delta, an
// explicit field id, a list, a map, a nested struct, and a truncated buffer.

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "src/parquet/thrift.h"

namespace {

int g_failures = 0;

void Fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++g_failures;
}

using nanochat::parquet::CompactReader;
using nanochat::parquet::CompactType;

void CheckScalarFields() {
  // Field 1 i32 = 7, field 2 binary = "abc", STOP.
  const std::uint8_t bytes[] = {0x15, 0x0e, 0x18, 0x03, 'a', 'b', 'c', 0x00};
  CompactReader reader(bytes, sizeof(bytes));
  int id = 0;
  CompactType type = CompactType::kStop;
  if (!reader.NextField(&id, &type) || id != 1 || type != CompactType::kI32) {
    Fail("scalar: bad field 1 header");
  } else if (reader.ReadInt() != 7) {
    Fail("scalar: field 1 value");
  }
  if (!reader.NextField(&id, &type) || id != 2 ||
      type != CompactType::kBinary) {
    Fail("scalar: bad field 2 header");
  } else if (reader.ReadBinary() != "abc") {
    Fail("scalar: field 2 value");
  }
  if (reader.NextField(&id, &type)) Fail("scalar: expected STOP");
  if (!reader.ok()) Fail("scalar: reader error: " + reader.error());
}

void CheckExplicitFieldId() {
  // Field id 20 with delta 0, i32 = -3, STOP.
  const std::uint8_t bytes[] = {0x05, 0x28, 0x05, 0x00};
  CompactReader reader(bytes, sizeof(bytes));
  int id = 0;
  CompactType type = CompactType::kStop;
  if (!reader.NextField(&id, &type) || id != 20 || type != CompactType::kI32) {
    Fail("explicit id: bad header");
  } else if (reader.ReadInt() != -3) {
    Fail("explicit id: value");
  }
  if (reader.NextField(&id, &type)) {
    Fail("explicit id: expected STOP");
  }
  if (!reader.ok()) Fail("explicit id: reader error: " + reader.error());
}

void CheckList() {
  // Field 1 list<i32> = [1, 2, 3], STOP.
  const std::uint8_t bytes[] = {0x19, 0x35, 0x02, 0x04, 0x06, 0x00};
  CompactReader reader(bytes, sizeof(bytes));
  int id = 0;
  CompactType type = CompactType::kStop;
  if (!reader.NextField(&id, &type) || id != 1 || type != CompactType::kList) {
    Fail("list: bad field header");
  }
  CompactReader::ListHeader header;
  if (!reader.ReadListHeader(&header) || header.type != CompactType::kI32 ||
      header.size != 3) {
    Fail("list: bad list header");
  }
  for (int i = 0; i < 3 && reader.ok(); ++i) {
    if (reader.ReadInt() != i + 1) Fail("list: element value");
  }
  if (reader.NextField(&id, &type)) Fail("list: expected STOP");
  if (!reader.ok()) Fail("list: reader error: " + reader.error());
}

void CheckSkipNestedStructAndMap() {
  // Field 1 struct { field 1 i32 = 7 }, field 2 binary = "hi", STOP.
  const std::uint8_t struct_bytes[] = {0x1c, 0x15, 0x0e, 0x00, 0x18,
                                       0x02, 'h',  'i',  0x00};
  CompactReader reader(struct_bytes, sizeof(struct_bytes));
  int id = 0;
  CompactType type = CompactType::kStop;
  if (!reader.NextField(&id, &type) || id != 1 ||
      type != CompactType::kStruct) {
    Fail("struct: bad field header");
  }
  reader.Skip(type);
  if (!reader.NextField(&id, &type) || id != 2 ||
      type != CompactType::kBinary) {
    Fail("struct: skipped too far");
  } else if (reader.ReadBinary() != "hi") {
    Fail("struct: field after skip");
  }
  if (reader.NextField(&id, &type)) Fail("struct: expected STOP");
  if (!reader.ok()) Fail("struct: reader error: " + reader.error());

  // Field 1 map<i32, i32> with two pairs, STOP.
  const std::uint8_t map_bytes[] = {0x1b, 0x02, 0x55, 0x02,
                                    0x04, 0x06, 0x08, 0x00};
  CompactReader map_reader(map_bytes, sizeof(map_bytes));
  if (!map_reader.NextField(&id, &type) || id != 1 ||
      type != CompactType::kMap) {
    Fail("map: bad field header");
  }
  map_reader.Skip(type);
  if (map_reader.NextField(&id, &type)) Fail("map: expected STOP");
  if (!map_reader.ok()) Fail("map: reader error: " + map_reader.error());
}

void CheckTruncation() {
  const std::uint8_t bytes[] = {0x15};
  CompactReader reader(bytes, sizeof(bytes));
  int id = 0;
  CompactType type = CompactType::kStop;
  if (!reader.NextField(&id, &type)) Fail("truncation: bad field header");
  reader.ReadInt();
  if (reader.ok()) Fail("truncation: reader accepted a missing value");
  if (reader.error().empty()) Fail("truncation: no error message");
}

}  // namespace

int main() {
  CheckScalarFields();
  CheckExplicitFieldId();
  CheckList();
  CheckSkipNestedStructAndMap();
  CheckTruncation();
  if (g_failures != 0) {
    std::fprintf(stderr, "thrift_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("thrift_test: ok\n");
  return 0;
}
