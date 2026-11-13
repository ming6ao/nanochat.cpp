#!/bin/sh
# The interposition path of the S0 simulator (docs/simulator.md sections 5.1,
# 5.2, and 12.2).
#
# `//tests:cuda_sim_preload_payload` links the real CUDA backend, so the calls
# bind to the mock only when this wrapper supplies three things:
#
#   * `LD_PRELOAD`, the interposer, by absolute path, because the dynamic
#     loader rejects a relative one;
#   * `NANOCHAT_SIM_LOG`, a file under `TEST_TMPDIR`;
#   * `NANOCHAT_SIM_PROFILE`, the profile the run answers as. It defaults to
#     `h100`, and `--test_env=NANOCHAT_SIM_PROFILE=<name>` overrides it.
#
# The payload's exit status becomes the test's, because the last line is an
# `exec`.

set -eu

profile="${NANOCHAT_SIM_PROFILE:-h100}"
export NANOCHAT_SIM_PROFILE="$profile"
export NANOCHAT_SIM_LOG="${TEST_TMPDIR:-/tmp}/nanochat-sim-preload-${profile}.jsonl"

# The runfiles root under Bazel, and the package's parent for a hand run.
root="${TEST_SRCDIR:-}"
if [ -n "$root" ]; then
  root="$root/${TEST_WORKSPACE:-_main}"
else
  root=$(cd "$(dirname "$0")/.." && pwd)
fi

export LD_PRELOAD="$root/tools/cuda_sim/libcuda_sim_interposer.so"
exec "$root/tests/cuda_sim_preload_payload"
