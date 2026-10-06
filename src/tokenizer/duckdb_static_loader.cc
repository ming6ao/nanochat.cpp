// The DuckDB static-extension loader shim.
//
// The prebuilt `libduckdb_static.a` release archive omits the generated
// `ExtensionHelper::LoadAllExtensions` object, so the DuckDB constructor leaves
// that one symbol undefined. DuckDB ships the same no-op for a static link
// without linked extensions
// (`extension/loader/dummy_static_extension_loader.cpp`). The built-in parquet
// reader and writer still register through the core function list, so
// `read_parquet` and `COPY ... (FORMAT PARQUET)` work.

#include <duckdb.hpp>

namespace duckdb {

class ExtensionHelper {
 public:
  static void LoadAllExtensions(DuckDB& db);
};

void ExtensionHelper::LoadAllExtensions(DuckDB&) {}

}  // namespace duckdb
