#!/usr/bin/env python3
"""Generate the Unicode range tables for the nanochat split pattern.

The split pattern (docs/tokenizer.md section 2.5) classifies code points with
the Unicode general categories ``\\p{L}`` (letter) and ``\\p{N}`` (number), and
with the ``\\s`` whitespace property. The C++ splitter must use the same ranges
as the Rust ``regex-syntax`` 0.8.11 tables inside ``tiktoken``. Those tables
come from Unicode 16.0.0, so this script reads the same release.

The script emits one C++ include file with three inclusive, ascending range
arrays. Run it from the repository root and redirect the output::

    python3 tools/gen_unicode_tables.py > src/unicode_tables.inc

By default the script downloads ``UnicodeData.txt`` and ``PropList.txt`` for
Unicode 16.0.0 and caches them under ``~/.cache/nanochat-unicode/16.0.0``. Use
``--unicode-data`` and ``--prop-list`` to pass local copies, for example from a
vendored ``ucd`` directory. The script never needs the network when the cache
is warm.

The letter and number ranges come from the ``General_Category`` field of
``UnicodeData.txt``. The whitespace ranges come from the ``White_Space``
property in ``PropList.txt``. The table counts are recorded in the generated
file for a quick sanity check.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

#: The pinned Unicode release. It matches ``regex-syntax`` 0.8.11.
UNICODE_VERSION = "16.0.0"
_BASE_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"
UNICODE_DATA_URL = f"{_BASE_URL}/UnicodeData.txt"
PROP_LIST_URL = f"{_BASE_URL}/PropList.txt"

#: The header line that identifies the pinned PropList release.
_PROP_LIST_MARKER = f"# PropList-{UNICODE_VERSION}.txt"

Range = tuple[int, int]


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "nanochat-unicode" / UNICODE_VERSION


def _download(url: str, destination: Path) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed URL
        payload = response.read()
    destination.write_bytes(payload)
    return payload


def _load_source(path: Path | None, url: str, cache: Path) -> bytes:
    if path is not None:
        return path.read_bytes()
    if cache.is_file() and cache.stat().st_size > 0:
        return cache.read_bytes()
    return _download(url, cache)


def parse_unicode_data(text: str) -> list[tuple[int, int, str]]:
    """Parse ``UnicodeData.txt`` into ``(first, last, category)`` ranges.

    The file lists every assigned code point on its own line. Large blocks use
    a ``<..., First>`` line and a ``<..., Last>`` line around the block. The
    return value keeps the ``General_Category`` values (for example ``Lu``).
    """
    ranges: list[tuple[int, int, str]] = []
    pending_first: int | None = None
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split(";")
        code = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        if name.endswith(", First>"):
            pending_first = code
            continue
        if name.endswith(", Last>"):
            if pending_first is None:
                raise ValueError(f"unpaired Last line for U+{code:04X}")
            ranges.append((pending_first, code, category))
            pending_first = None
            continue
        ranges.append((code, code, category))
    if pending_first is not None:
        raise ValueError("unpaired First line at end of UnicodeData.txt")
    return ranges


def parse_prop_list(text: str, property_name: str) -> list[Range]:
    """Parse one boolean property from ``PropList.txt`` into ranges."""
    ranges: list[Range] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 2 or parts[1] != property_name:
            continue
        field = parts[0]
        if ".." in field:
            first_text, last_text = field.split("..")
            first, last = int(first_text, 16), int(last_text, 16)
        else:
            first = last = int(field, 16)
        ranges.append((first, last))
    return ranges


def merge_ranges(ranges: Iterable[Range]) -> list[Range]:
    """Merge overlapping and adjacent inclusive ranges into ascending order."""
    merged: list[Range] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def general_category_ranges(
    parsed: Sequence[tuple[int, int, str]], prefixes: tuple[str, ...]
) -> list[Range]:
    """Select the ranges whose category starts with one of the prefixes."""
    selected = [
        (first, last) for first, last, category in parsed
        if category.startswith(prefixes)
    ]
    return merge_ranges(selected)


def _format_ranges(name: str, ranges: Sequence[Range]) -> list[str]:
    lines = [f"inline constexpr Range {name}[] = {{"]
    for first, last in ranges:
        lines.append(f"    {{0x{first:04X}, 0x{last:04X}}},")
    lines.append("};")
    return lines


def emit_include(
    letters: Sequence[Range],
    numbers: Sequence[Range],
    whitespace: Sequence[Range],
    unicode_data_hash: str,
    prop_list_hash: str,
) -> str:
    """Render the C++ range tables as one include file."""
    lines = [
        "// Generated by tools/gen_unicode_tables.py. DO NOT EDIT.",
        "//",
        f"// Unicode {UNICODE_VERSION} ranges for the nanochat split pattern",
        "// (docs/tokenizer.md section 2.5). The ranges are inclusive and",
        "// sorted in ascending order. A lookup uses a binary search.",
        "//",
        f"// UnicodeData.txt sha256: {unicode_data_hash}",
        f"// PropList.txt   sha256: {prop_list_hash}",
        "//",
        f"// letter ranges:     {len(letters)}",
        f"// number ranges:     {len(numbers)}",
        f"// whitespace ranges: {len(whitespace)}",
        "",
        "#ifndef NANOCHAT_SRC_UNICODE_TABLES_INC_",
        "#define NANOCHAT_SRC_UNICODE_TABLES_INC_",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace nanochat {",
        "namespace unicode {",
        "",
        "struct Range {",
        "  std::uint32_t first;",
        "  std::uint32_t last;",
        "};",
        "",
    ]
    for const_name, ranges in (
        ("kLetterRanges", letters),
        ("kNumberRanges", numbers),
        ("kWhitespaceRanges", whitespace),
    ):
        lines.extend(_format_ranges(const_name, ranges))
        lines.append("")
    lines.extend([
        "inline constexpr std::size_t kLetterRangeCount =",
        "    sizeof(kLetterRanges) / sizeof(kLetterRanges[0]);",
        "inline constexpr std::size_t kNumberRangeCount =",
        "    sizeof(kNumberRanges) / sizeof(kNumberRanges[0]);",
        "inline constexpr std::size_t kWhitespaceRangeCount =",
        "    sizeof(kWhitespaceRanges) / sizeof(kWhitespaceRanges[0]);",
        "",
        "}  // namespace unicode",
        "}  // namespace nanochat",
        "",
        "#endif  // NANOCHAT_SRC_UNICODE_TABLES_INC_",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Unicode 16.0.0 range tables.")
    parser.add_argument(
        "--unicode-data", type=Path, default=None,
        help="path to a local UnicodeData.txt")
    parser.add_argument(
        "--prop-list", type=Path, default=None,
        help="path to a local PropList.txt")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="cache directory for the downloaded source files")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="write the include file here instead of standard output")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cache = args.cache_dir if args.cache_dir is not None else _cache_dir()
    unicode_data = _load_source(
        args.unicode_data, UNICODE_DATA_URL, cache / "UnicodeData.txt")
    prop_list = _load_source(
        args.prop_list, PROP_LIST_URL, cache / "PropList.txt")
    unicode_text = unicode_data.decode("utf-8")
    prop_text = prop_list.decode("utf-8")
    if _PROP_LIST_MARKER not in prop_text:
        raise SystemExit(
            f"PropList.txt is not Unicode {UNICODE_VERSION}: missing "
            f"{_PROP_LIST_MARKER!r}")

    parsed = parse_unicode_data(unicode_text)
    letters = general_category_ranges(parsed, ("L",))
    numbers = general_category_ranges(parsed, ("N",))
    whitespace = merge_ranges(parse_prop_list(prop_text, "White_Space"))

    output = emit_include(
        letters, numbers, whitespace,
        hashlib.sha256(unicode_data).hexdigest(),
        hashlib.sha256(prop_list).hexdigest(),
    )
    if args.out is None:
        print(output, end="")
    else:
        args.out.write_text(output, encoding="utf-8")
    print(
        f"unicode {UNICODE_VERSION}: {len(letters)} letter ranges, "
        f"{len(numbers)} number ranges, "
        f"{len(whitespace)} whitespace ranges",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
