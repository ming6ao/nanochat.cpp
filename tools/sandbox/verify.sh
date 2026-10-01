#!/usr/bin/env bash
# Runs inside the sandbox scope, before the real command.
#
# usage: verify.sh <mem_max> <cpu_quota> <cpus> <pids> -- <command...>
#
# Reads the limits back from this process's cgroup and exits non-zero if they
# do not match the requested profile, so a budget that was silently not applied
# cannot be mistaken for a sandboxed run. On success it execs the command.
set -euo pipefail

expected_mem=$1
expected_quota=$2
expected_cpus=$3
expected_pids=$4
shift 4
[[ ${1:-} == "--" ]] && shift

to_bytes() {
  local v=$1 n unit
  n=${v%[KkMmGg]}
  unit=${v#"$n"}
  case "$unit" in
    "" | b | B) printf '%s\n' "$n" ;;
    K | k) printf '%s\n' "$((n * 1024))" ;;
    M | m) printf '%s\n' "$((n * 1024 * 1024))" ;;
    G | g) printf '%s\n' "$((n * 1024 * 1024 * 1024))" ;;
    *)
      echo "sandbox: cannot parse size '$v'" >&2
      return 1
      ;;
  esac
}

cg=/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup)
actual_mem=$(cat "$cg/memory.max" 2>/dev/null || echo missing)
actual_cpu=$(cat "$cg/cpu.max" 2>/dev/null || echo missing)
actual_pids=$(cat "$cg/pids.max" 2>/dev/null || echo missing)
actual_cpus=$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)

want_mem=$(to_bytes "$expected_mem")
want_cpu="$(( ${expected_quota%\%} * 1000 )) 100000"

fail=""
[[ $actual_mem == "$want_mem" ]] || fail+=" memory.max=$actual_mem(want $want_mem)"
[[ $actual_cpu == "$want_cpu" ]] || fail+=" cpu.max='$actual_cpu'(want '$want_cpu')"
[[ $actual_pids == "$expected_pids" ]] || fail+=" pids.max=$actual_pids(want $expected_pids)"
[[ $actual_cpus == "$expected_cpus" ]] || fail+=" cpus=$actual_cpus(want $expected_cpus)"

if [[ -n $fail ]]; then
  echo "sandbox: limits not applied:$fail" >&2
  echo "sandbox: refusing to run the command unsandboxed." >&2
  exit 1
fi

printf 'sandbox: profile=%s mem.max=%s cpu.max=%s pids.max=%s cpus=%s\n' \
  "${NANOCHAT_SANDBOX:-?}" "$actual_mem" "$actual_cpu" "$actual_pids" "$actual_cpus" \
  >>/tmp/nanochat-sandbox.log

[[ ${NANOCHAT_SANDBOX_QUIET:-0} == 1 ]] || \
  printf 'sandbox: profile=%s mem.max=%s cpu.max=%s cpus=%s\n' \
    "${NANOCHAT_SANDBOX:-?}" "$actual_mem" "$actual_cpu" "$actual_cpus" >&2

exec "$@"
