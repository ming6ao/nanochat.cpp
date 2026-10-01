#!/usr/bin/env bash
# Exclusive GPU broker for nanochat.
#
# usage: tools/gpu.sh [--profile t1-gpu|t2-parity|t3-bench] [--lock-only] [--] <command...>
#
# Takes the single-GPU lock, refuses to run if a foreign process owns the
# device, then hands off to tools/sandbox.sh with a GPU profile. With
# --lock-only it holds the lock but does not sandbox the command, which is what
# a whole `bazel test` invocation needs (the tests themselves are sandboxed per
# action). Prefer tools/nanochat over calling this directly. See docs/sandbox.md.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

profile=t1-gpu
lock_only=0
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
    --lock-only)
      lock_only=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done
(( $# )) || {
  echo "gpu: no command given" >&2
  exit 2
}
case "$profile" in
  t1-gpu | t2-parity | t3-bench) ;;
  *)
    echo "gpu: profile must be t1-gpu, t2-parity, or t3-bench" >&2
    exit 2
    ;;
esac

# The lock fd is held for the lifetime of the process, so the broker serializes
# GPU access even though the job itself runs under systemd.
LOCK=/tmp/nanochat-gpu.lock
exec 9>"$LOCK"
flock 9 # exclusive; blocks until the previous GPU job finishes

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU busy with an unrelated process" >&2
  exit 1
fi

echo "acquired GPU ($profile): $*" | tee -a /tmp/nanochat-gpu.log
export CUDA_VISIBLE_DEVICES=0

if (( lock_only )); then
  exec "$@"
fi
exec "$here/sandbox.sh" --profile "$profile" -- "$@"
