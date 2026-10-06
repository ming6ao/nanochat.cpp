load("@rules_cc//cc:cc_import.bzl", "cc_import")

# Build file for the prebuilt DuckDB release archive. See docs/tokenizer.md
# section 10. The archive is the pinned DuckDB 1.5.6 host build; MODULE.bazel
# unpacks it as @duckdb and exposes @duckdb//:duckdb.

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "duckdb",
    hdrs = [
        "duckdb.hpp",
        "duckdb.h",
        "duckdb_extension.h",
    ],
    static_library = "libduckdb_static.a",
)
