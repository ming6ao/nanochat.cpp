#!/usr/bin/env bash
# nanochat.cpp style gate: the Google C++ Style Guide checks.
#
# Runs clang-format in check mode over every tracked C++ source, then the text
# checks that clang-format cannot make:
#   * no using-directive
#   * every project include is project-relative
#   * no C-style cast
#   * no non-ASCII character
#
# usage: tools/lint.sh [file...]
#
# With no arguments it checks every tracked .h/.cc/.cu/.cuh file. See
# CONTRIBUTING.md, .clang-format, and docs/testing.md.

set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/.." && pwd)
cd "$root"

if (( $# )); then
  files=("$@")
else
  mapfile -t files < <(git ls-files '*.h' '*.cc' '*.cu' '*.cuh')
fi
(( ${#files[@]} )) || {
  echo "lint: no C++ sources to check" >&2
  exit 1
}

fail=0

# --- clang-format ----------------------------------------------------------
if ! command -v clang-format >/dev/null 2>&1; then
  echo "lint: clang-format is not on PATH" >&2
  exit 1
fi
if ! clang-format --dry-run --Werror "${files[@]}"; then
  echo "lint: clang-format reported differences" >&2
  fail=1
fi

# --- text checks -----------------------------------------------------------
report() { # message hits
  local message=$1 hits=$2
  [[ -z $hits ]] && return 0
  echo "lint: $message" >&2
  echo "$hits" >&2
  fail=1
}

report "using-directive; Google bans 'using namespace'" \
  "$(grep -nHE 'using namespace' "${files[@]}" || true)"

report "bare project include; use a path from the repository root" \
  "$(grep -nHE '^#include "' "${files[@]}" |
     grep -vE ':#include "(nanochat/|backends/|src/|include/|tools/|tests/|dev/)' || true)"

report "C-style cast; use static_cast or a brace initialization" \
  "$(grep -nHE '\((int|float|double|char|unsigned|long|size_t|std::[a-z_0-9]+)\)[A-Za-z_0-9(]' \
     "${files[@]}" || true)"

report "non-ASCII character" \
  "$(grep -nHP '[^\x00-\x7F]' "${files[@]}" || true)"

if (( fail )); then
  echo "lint: FAILED" >&2
  exit 1
fi
echo "lint: OK (${#files[@]} files)"
