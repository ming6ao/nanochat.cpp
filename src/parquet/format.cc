#include "src/parquet/format.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "src/parquet/thrift.h"

namespace nanochat {
namespace parquet {
namespace {

// Parses the fields of one struct until STOP. The callback calls a `Read*`
// method or `Skip` for every field, so the reader advances exactly once per
// field. The field-id delta state is saved and restored, so a nested struct
// does not disturb its parent.
template <typename FieldFn>
void ParseStructFields(CompactReader* reader, FieldFn field) {
  const int saved = reader->field_id_state();
  reader->set_field_id_state(0);
  while (reader->ok()) {
    int field_id = 0;
    CompactType type = CompactType::kStop;
    if (!reader->NextField(&field_id, &type)) break;
    field(field_id, type);
  }
  reader->set_field_id_state(saved);
}

void ParseSchemaElement(CompactReader* reader, SchemaElement* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->type = static_cast<int>(reader->ReadInt());
        out->has_type = true;
        break;
      case 3:
        out->repetition = static_cast<int>(reader->ReadInt());
        out->has_repetition = true;
        break;
      case 4:
        out->name = std::string(reader->ReadBinary());
        break;
      case 5:
        out->num_children = static_cast<int>(reader->ReadInt());
        break;
      case 6:
        out->converted_type = static_cast<int>(reader->ReadInt());
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

void ParseColumnMetaData(CompactReader* reader, ColumnMetaData* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->type = static_cast<int>(reader->ReadInt());
        break;
      case 2: {
        CompactReader::ListHeader header;
        if (!reader->ReadListHeader(&header)) return;
        if (header.type != CompactType::kI32) {
          reader->Fail("parquet: malformed encodings list");
          return;
        }
        for (std::size_t i = 0; i < header.size && reader->ok(); ++i) {
          out->encodings.push_back(static_cast<int>(reader->ReadInt()));
        }
        break;
      }
      case 3: {
        CompactReader::ListHeader header;
        if (!reader->ReadListHeader(&header)) return;
        if (header.type != CompactType::kBinary) {
          reader->Fail("parquet: malformed path_in_schema list");
          return;
        }
        for (std::size_t i = 0; i < header.size && reader->ok(); ++i) {
          out->path_in_schema.emplace_back(reader->ReadBinary());
        }
        break;
      }
      case 4:
        out->codec = static_cast<int>(reader->ReadInt());
        break;
      case 5:
        out->num_values = reader->ReadInt();
        break;
      case 6:
        out->total_uncompressed_size = reader->ReadInt();
        break;
      case 7:
        out->total_compressed_size = reader->ReadInt();
        break;
      case 9:
        out->data_page_offset = reader->ReadInt();
        out->has_data_page_offset = true;
        break;
      case 11:
        out->dictionary_page_offset = reader->ReadInt();
        out->has_dictionary_page_offset = true;
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

void ParseColumnChunk(CompactReader* reader, ColumnChunk* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->file_path = std::string(reader->ReadBinary());
        break;
      case 3:
        if (type != CompactType::kStruct) {
          reader->Fail("parquet: malformed column metadata");
          return;
        }
        ParseColumnMetaData(reader, &out->meta_data);
        out->has_meta_data = true;
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

void ParseRowGroup(CompactReader* reader, RowGroup* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1: {
        CompactReader::ListHeader header;
        if (!reader->ReadListHeader(&header)) return;
        if (header.type != CompactType::kStruct) {
          reader->Fail("parquet: malformed column list");
          return;
        }
        out->columns.resize(header.size);
        for (std::size_t i = 0; i < header.size && reader->ok(); ++i) {
          ParseColumnChunk(reader, &out->columns[i]);
        }
        break;
      }
      case 2:
        out->total_byte_size = reader->ReadInt();
        break;
      case 3:
        out->num_rows = reader->ReadInt();
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

void ParseDataPageHeader(CompactReader* reader, DataPageHeader* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->num_values = static_cast<int>(reader->ReadInt());
        break;
      case 2:
        out->encoding = static_cast<int>(reader->ReadInt());
        break;
      case 3:
        out->definition_level_encoding = static_cast<int>(reader->ReadInt());
        break;
      case 4:
        out->repetition_level_encoding = static_cast<int>(reader->ReadInt());
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

void ParsePageHeaderFields(CompactReader* reader, PageHeader* out) {
  ParseStructFields(reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->type = static_cast<int>(reader->ReadInt());
        break;
      case 2:
        out->uncompressed_page_size = static_cast<int>(reader->ReadInt());
        break;
      case 3:
        out->compressed_page_size = static_cast<int>(reader->ReadInt());
        break;
      case 5:
        if (type != CompactType::kStruct) {
          reader->Fail("parquet: malformed data page header");
          return;
        }
        ParseDataPageHeader(reader, &out->data_page_header);
        out->has_data_page_header = true;
        break;
      default:
        reader->Skip(type);
        break;
    }
  });
}

}  // namespace

bool ParseFileMetaData(const std::uint8_t* data, std::size_t size,
                       FileMetaData* out, std::string* error) {
  CompactReader reader(data, size);
  ParseStructFields(&reader, [&](int field_id, CompactType type) {
    switch (field_id) {
      case 1:
        out->version = static_cast<int>(reader.ReadInt());
        break;
      case 2: {
        CompactReader::ListHeader header;
        if (!reader.ReadListHeader(&header)) return;
        if (header.type != CompactType::kStruct) {
          reader.Fail("parquet: malformed schema list");
          return;
        }
        out->schema.resize(header.size);
        for (std::size_t i = 0; i < header.size && reader.ok(); ++i) {
          ParseSchemaElement(&reader, &out->schema[i]);
        }
        break;
      }
      case 3:
        out->num_rows = reader.ReadInt();
        break;
      case 4: {
        CompactReader::ListHeader header;
        if (!reader.ReadListHeader(&header)) return;
        if (header.type != CompactType::kStruct) {
          reader.Fail("parquet: malformed row group list");
          return;
        }
        out->row_groups.resize(header.size);
        for (std::size_t i = 0; i < header.size && reader.ok(); ++i) {
          ParseRowGroup(&reader, &out->row_groups[i]);
        }
        break;
      }
      case 6:
        out->created_by = std::string(reader.ReadBinary());
        break;
      default:
        reader.Skip(type);
        break;
    }
  });
  if (!reader.ok()) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  return true;
}

bool ParsePageHeader(const std::uint8_t* data, std::size_t size,
                     std::size_t* consumed, PageHeader* out,
                     std::string* error) {
  CompactReader reader(data, size);
  ParsePageHeaderFields(&reader, out);
  if (!reader.ok()) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  if (consumed != nullptr) *consumed = reader.position();
  return true;
}

}  // namespace parquet
}  // namespace nanochat
