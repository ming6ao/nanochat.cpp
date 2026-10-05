#!/usr/bin/env bash
# Resource-limited launcher for every nanochat job.
#
# usage: tools/sandbox.sh [--profile NAME] [--backend systemd|none|auto]
#                         [--no-slice] [--] <command...>
#
# Applies a cgroup v2 budget (RAM, CPU, pids, wall clock) through the
# unprivileged systemd user manager, plus CPU affinity and hard rlimits. No
# root and no container runtime are involved. The limits are read back and
# verified before the command runs.
#
# With --backend=none (or NANOCHAT_SANDBOX_BACKEND=none) the script does not
# sandbox. It sets the entry-point sentinel and execs the command. A host
# without cgroup v2, such as a Kaggle Notebook, needs this mode. See
# tools/nanochat, docs/sandbox.md, and docs/host-portability.md.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=sandbox/profiles.conf
. "$here/sandbox/profiles.conf"

profile=t0-cpu
use_slice=1
backend=${NANOCHAT_SANDBOX_BACKEND:-auto}
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
    --backend)
      backend=${2:?--backend needs a value}
      shift 2
      ;;
    --backend=*)
      backend=${1#--backend=}
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

# Resolve `auto` to a concrete backend. The systemd user manager is present on
# the workstation and absent in a container. The probe is cheap and reliable.
if [[ $backend == auto ]]; then
  if systemd-run --user --scope --quiet true >/dev/null 2>&1; then
    backend=systemd
  else
    backend=none
  fi
fi
case "$backend" in
  systemd | none) ;;
  *)
    echo "sandbox: backend must be systemd, none, or auto" >&2
    exit 2
    ;;
esac

export NANOCHAT_SANDBOX_BACKEND="$backend"

# The GPU broker selects a device by exporting CUDA_VISIBLE_DEVICES, but a
# Bazel test action does not inherit the broker's shell environment.
# tools/nanochat forwards NANOCHAT_GPU_DEVICES into the action instead, so
# translate it here unless the caller already chose a device. See
# docs/host-portability.md section 5.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} && -n ${NANOCHAT_GPU_DEVICES:-} ]]; then
  export CUDA_VISIBLE_DEVICES="$NANOCHAT_GPU_DEVICES"
fi

if [[ $backend == none ]]; then
  # No cgroup, no affinity, no rlimits. The container quota and the entry-point
  # guard are the only controls. Set the sentinel so RequireSandboxOrDie()
  # passes; `off` marks a deliberate unsandboxed run.
  export NANOCHAT_SANDBOX="off"
  export NANOCHAT_SANDBOX_PROFILE="$profile"
  # Match library thread pools to the visible CPUs. taskset is unavailable in
  # this mode, so the process count is the only oversubscription control.
  nthreads=$(nproc 2>/dev/null || echo 1)
  export OMP_NUM_THREADS=$nthreads
  export NANOCHAT_NUM_THREADS=$nthreads
  printf 'sandbox: backend=none profile=%s (not sandboxed)\n' "$profile" \
    >>/tmp/nanochat-sandbox.log
  [[ ${NANOCHAT_SANDBOX_QUIET:-0} == 1 ]] || \
    printf 'sandbox: backend=none profile=%s (not sandboxed)\n' "$profile" >&2
  exec "$@"
fi

# --- systemd backend --------------------------------------------------------
#
# Make sure the aggregate ceiling exists so several concurrent jobs cannot add
# up past the host budget. Idempotent; a missing slice is a warning, not fatal.
if (( use_slice )); then
  slice_src="$here/sandbox/nanochat.slice"
  slice_dst="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/nanochat.slice"
  if [[ ! -e "$slice_dst" && -f "$slice_src" ]]; then
    mkdir -p "$(dirname "$slice_dst")"
    if cp "$slice_src" "$slice_dst" && systemctl --user daemon-reload >/dev/null 2>&1; then
      echo "sandbox: installed aggregate slice at $slice_dst" >&2
    else
      echo "sandbox: warning: could not install $slice_dst; no aggregate ceiling" >&2
    fi
  fi
fi

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
# Sentinel read by RequireSandboxOrDie(); see include/nanochat/sandbox.h.
export NANOCHAT_SANDBOX="$profile"

# RLIMIT_NPROC is deliberately not set: it is per-UID across the whole host, so
# a small value would break unrelated agent processes. TasksMax (per-cgroup) is
# the correct fork-bomb guard. --core=0 keeps core dumps out of the worktree.
exec systemd-run --user --scope --quiet --same-dir "${props[@]}" -- \
  taskset -c "$SB_CPUS" \
  prlimit --core=0 -- \
  "$here/sandbox/verify.sh" "$SB_MEM_MAX" "$SB_QUOTA" "$SB_CPUS" "$SB_PIDS" -- \
  "$@"
