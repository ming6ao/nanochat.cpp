#!/bin/sh
# The symbol gate of the S0 API interposer, end to end (docs/simulator.md,
# invariant 6).
#
# `//python:cuda_sim_symbols_test` proves the parser against fake `nm` and
# `readelf`. This wrapper proves the gate on the artefacts the build produces:
# the interposer exports every name the version script lists, the real runtime
# test binary asks only for names the interposer exports, and the gate rejects
# a mock that is missing one.
#
# It passes `--map` and `--mock` itself, because the script's defaults point at
# `bazel-bin`, which a runfiles tree does not have.

set -eu

root="${TEST_SRCDIR:-}"
if [ -n "$root" ]; then
  root="$root/${TEST_WORKSPACE:-_main}"
else
  root=$(cd "$(dirname "$0")/../.." && pwd)
fi

gate="$root/tools/cuda_sim/check_symbols.py"
map="$root/tools/cuda_sim/cuda_sim.map"
mock="$root/tools/cuda_sim/libcuda_sim_interposer.so"
real="$root/backends/cuda/cuda_runtime_gpu_test"

python3 "$gate" --map "$map" --mock "$mock"
python3 "$gate" --map "$map" --mock "$mock" "$real"
python3 "$gate" --map "$map" --mock "$mock" --negative-self-test
