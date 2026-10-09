#!/usr/bin/env bash
# T0 syntax test for the Kaggle host scripts. Hermetic and CPU-only.
#
# usage: scripts_test.sh <script...>
set -euo pipefail
for script in "$@"; do
  bash -n "$script"
done

# bootstrap.sh writes the host contract once. Source it with a temporary HOME
# and check the keys, so the notebook and the terminal stay in step
# (docs/host-portability.md section 8.3).
bootstrap=""
for script in "$@"; do
  case "$script" in
    */bootstrap.sh) bootstrap="$script" ;;
  esac
done
if [[ -n $bootstrap ]]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  HOME="$tmp" bash -c 'source "$1"; write_env_file' _ "$bootstrap"
  for key in NANOCHAT_SANDBOX_BACKEND NANOCHAT_CPP_BACKEND \
             NANOCHAT_CPP_PRECISION NANOCHAT_CUDA_ARCH \
             NANOCHAT_CPP_CACHE CUDA_HOME; do
    if ! grep -q "export $key=" "$tmp/.nanochat.env"; then
      echo "missing $key in .nanochat.env" >&2
      exit 1
    fi
  done
  grep -q 'NANOCHAT_CPP_BACKEND="cuda"' "$tmp/.nanochat.env"
  grep -q 'NANOCHAT_CPP_PRECISION="fp32"' "$tmp/.nanochat.env"
  grep -q 'NANOCHAT_CUDA_ARCH="sm_75"' "$tmp/.nanochat.env"
  echo "kaggle env file: OK"
fi

echo "kaggle scripts: OK ($# files)"
