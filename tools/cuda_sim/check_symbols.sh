#!/usr/bin/env bash
# The symbol gate for the S0 API interposer (docs/simulator.md sections 5.2 and
# 12.2, invariant 6).
#
# The gate reads the version script as the required set and compares it against
# two things:
#
#   * the mock's dynamic symbol table -- every name the script lists must be
#     exported, and the version nodes must be present;
#   * the undefined CUDA symbols of a binary under test -- every runtime or
#     cuBLAS symbol that binary asks for must be one the mock exports. The mock
#     may export more than the binary needs.
#
# usage: tools/cuda_sim/check_symbols.sh [options] [binary...]
#
#   --map FILE             use FILE as the version script (default cuda_sim.map)
#   --mock FILE            check FILE as the mock (default the built interposer)
#   --negative-self-test   prove the gate fails when a symbol is missing, then
#                          exit; used by `tools/nanochat simulate --suite api`
#
# With no binary arguments only the mock is checked. The mock's path is
# `bazel-bin/tools/cuda_sim/libcuda_sim_interposer.so`; build it first with
# `tools/nanochat simulate --suite api` or `tools/nanochat build`.
#
# Exit status: 0 when the gate passes, 1 when a required symbol is missing.

set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
map_file="$here/cuda_sim.map"
mock="$root/bazel-bin/tools/cuda_sim/libcuda_sim_interposer.so"

binaries=()
while (($#)); do
  case "$1" in
    --map) map_file=${2:?--map needs a path}; shift 2 ;;
    --map=*) map_file=${1#--map=}; shift ;;
    --mock) mock=${2:?--mock needs a path}; shift 2 ;;
    --mock=*) mock=${1#--mock=}; shift ;;
    --negative-self-test)
      # The gate's own proof: a required set with one symbol the mock does not
      # export must be rejected. The extra name is injected into a copy of the
      # version script, so no build artifact is touched.
      tmp_map=$(mktemp)
      trap 'rm -f "$tmp_map"' EXIT
      sed '0,/^[[:space:]]*cudaMalloc;/{s/^\([[:space:]]*cudaMalloc;\)/\1\n    cudaSimMissingSymbol;/}' \
        "$map_file" >"$tmp_map"
      if ! grep -q 'cudaSimMissingSymbol;' "$tmp_map"; then
        echo "check_symbols: negative self-test could not build its fixture" >&2
        exit 1
      fi
      if "$0" --map "$tmp_map" --mock "$mock" >/dev/null 2>&1; then
        echo "check_symbols: negative self-test FAILED (a missing symbol was accepted)" >&2
        exit 1
      fi
      echo "check_symbols: negative self-test OK (a missing symbol is rejected)"
      exit 0
      ;;
    *) binaries+=("$1"); shift ;;
  esac
done

[[ -f $map_file ]] || {
  echo "check_symbols: $map_file is missing" >&2
  exit 2
}
[[ -f $mock ]] || {
  echo "check_symbols: $mock is missing; build //tools/cuda_sim:cuda_sim_interposer" >&2
  exit 2
}

# The names in the script's `global:` clauses, one per line, with the trailing
# semicolon removed.
required=$(sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z_0-9]*\);[[:space:]]*$/\1/p' "$map_file" | sort -u)
[[ -n $required ]] || {
  echo "check_symbols: the version script lists no global symbols" >&2
  exit 2
}

defined=$(nm -D --defined-only "$mock" \
  | awk '{print $NF}' | sed 's/@.*//' | sort -u)

fail=0
missing=""
while read -r symbol; do
  [[ -z $symbol ]] && continue
  if ! grep -qx -- "$symbol" <<<"$defined"; then
    missing+="$symbol"$'\n'
  fi
done <<<"$required"

if [[ -n $missing ]]; then
  echo "check_symbols: the mock does not export:" >&2
  printf '%s' "$missing" >&2
  fail=1
fi

# The version nodes must be the ones the binary under test references.
nodes=$(readelf -V "$mock" 2>/dev/null \
  | sed -n 's/.*Name: \([^ ]*\).*/\1/p' | sort -u || true)
for node in libcudart.so.12 libcublas.so.12; do
  if ! grep -qx -- "$node" <<<"$nodes"; then
    echo "check_symbols: the mock has no version node '$node'" >&2
    fail=1
  fi
done

# Every CUDA symbol a named binary asks for must be one the mock defines.
for binary in "${binaries[@]}"; do
  [[ -f $binary ]] || {
    echo "check_symbols: $binary is missing" >&2
    exit 2
  }
  undefined=$(nm -D --undefined-only "$binary" 2>/dev/null \
    | awk '{print $NF}' | sed 's/@.*//' \
    | grep -E '^(cuda|__cuda|cublas)' | sort -u || true)
  unresolved=""
  while read -r symbol; do
    [[ -z $symbol ]] && continue
    if ! grep -qx -- "$symbol" <<<"$defined"; then
      unresolved+="$symbol"$'\n'
    fi
  done <<<"$undefined"
  if [[ -n $unresolved ]]; then
    echo "check_symbols: $binary asks for symbols the mock lacks:" >&2
    printf '%s' "$unresolved" >&2
    fail=1
  fi
done

if ((fail)); then
  echo "check_symbols: FAILED" >&2
  exit 1
fi
echo "check_symbols: OK ($(wc -l <<<"$required") required symbols)"
