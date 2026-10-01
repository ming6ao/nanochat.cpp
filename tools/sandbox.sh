#!/usr/bin/env bash
# Resource-limited launcher for every nanochat job.
#
# usage: tools/sandbox.sh [--profile NAME] [--no-slice] [--] <command...>
#
# Applies a cgroup v2 budget (RAM, CPU, pids, wall clock) through the
# unprivileged systemd user manager, plus CPU affinity and hard rlimits. No
# root and no container runtime are involved. See tools/sandbox/README.md and
# DESIGN.md section 15.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=sandbox/profiles.conf
. "$here/sandbox/profiles.conf"

profile=t0-cpu
use_slice=1
while (( $# )); do
  case "$1" in
    --profile)
      profile=${2:?--profile needs a name}
      shift 2
      ;;
    --profile=*)
      profile=${1#--profile=}
      shift
      ;;
    --no-slice)
      use_slice=0
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "sandbox: unknown option '$1'" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done
(( $# )) || {
  echo "sandbox: no command given" >&2
  exit 2
}

sandbox_profile "$profile"

props=(
  -p "MemoryHigh=$SB_MEM_HIGH"
  -p "MemoryMax=$SB_MEM_MAX"
  -p "MemorySwapMax=0"
  -p "CPUQuota=$SB_QUOTA"
  -p "TasksMax=$SB_PIDS"
)
if [[ -n "$SB_RUNTIME" ]]; then
  props+=( -p "RuntimeMaxSec=$SB_RUNTIME" )
fi
if (( use_slice )); then
  props+=( --slice=nanochat.slice )
fi

# Match library thread pools (OpenMP, cuBLAS) to the CPU budget; the cgroup is
# the hard ceiling, this is the polite hint that avoids 12-way oversubscription.
nthreads=0
IFS=, read -ra ranges <<<"$SB_CPUS"
for r in "${ranges[@]}"; do
  if [[ $r == *-* ]]; then
    lo=${r%-*}
    hi=${r#*-}
    nthreads=$((nthreads + hi - lo + 1))
  else
    nthreads=$((nthreads + 1))
  fi
done
(( nthreads > 0 )) || nthreads=1
export OMP_NUM_THREADS=$nthreads
export NANOCHAT_NUM_THREADS=$nthreads

# RLIMIT_NPROC is deliberately not set: it is per-UID across the whole host, so
# a small value would break unrelated agent processes. TasksMax (per-cgroup) is
# the correct fork-bomb guard. --core=0 keeps core dumps out of the worktree.
exec systemd-run --user --scope --quiet --same-dir "${props[@]}" -- \
  taskset -c "$SB_CPUS" \
  prlimit --core=0 -- \
  "$@"
