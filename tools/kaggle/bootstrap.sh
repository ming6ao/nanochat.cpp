#!/usr/bin/env bash
# Prepare a Kaggle Notebook session for nanochat.cpp.
#
# usage: tools/kaggle/bootstrap.sh
#
# Installs Bazelisk, Node.js, and pi, points the shell at the image CUDA
# toolkit, writes the host .bazelrc.local for the T4, and runs
# tools/nanochat doctor. A second run is safe and fast.
#
# A Kaggle container has no systemd and no cgroup v2 user manager, so the
# resource sandbox stays off. tools/sandbox.sh resolves to backend=none by
# itself; this script also exports NANOCHAT_SANDBOX_BACKEND=none for the
# session. See docs/host-portability.md.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)

NODE_VERSION=${NANOCHAT_KAGGLE_NODE_VERSION:-22.19.0}
export PATH="$HOME/.local/bin:$PATH"

log() { printf 'kaggle: %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

install_packages() {
  local missing=() cmd
  for cmd in git curl tar xz gcc g++ rg; do
    if ! have "$cmd"; then
      missing+=("$cmd")
    fi
  done
  if (( ${#missing[@]} == 0 )); then
    return 0
  fi
  if ! have apt-get || [[ $(id -u) != 0 ]]; then
    log "warning: missing ${missing[*]}; install them by hand if a step fails"
    return 0
  fi
  log "install ${missing[*]} with apt-get"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git curl ca-certificates tar xz-utils ripgrep build-essential
}

setup_cuda() {
  if [[ -z ${CUDA_HOME:-} ]]; then
    local candidate
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
      if [[ -x $candidate/bin/nvcc ]]; then
        CUDA_HOME=$candidate
        break
      fi
    done
  fi
  if [[ -z ${CUDA_HOME:-} ]] && have nvcc; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  fi
  if [[ -n ${CUDA_HOME:-} ]]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    log "CUDA_HOME=$CUDA_HOME"
  else
    log "warning: nvcc not found; the CUDA build will fail"
  fi
}

install_bazelisk() {
  if have bazelisk; then
    return 0
  fi
  log "install bazelisk into ~/.local/bin"
  mkdir -p "$HOME/.local/bin"
  curl -fsSL \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -o "$HOME/.local/bin/bazelisk"
  chmod +x "$HOME/.local/bin/bazelisk"
}

node_ok() {
  if ! have node; then
    return 1
  fi
  node -e 'const [a, b] = process.versions.node.split(".").map(Number);
           process.exit(a > 22 || (a === 22 && b >= 19) ? 0 : 1)' 2>/dev/null
}

install_node() {
  if node_ok; then
    return 0
  fi
  local dir="$HOME/.local/node"
  local tarball="node-v${NODE_VERSION}-linux-x64"
  if [[ ! -x $dir/bin/node ]]; then
    log "install Node.js $NODE_VERSION into $dir"
    mkdir -p "$HOME/.local"
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/${tarball}.tar.xz" \
      | tar -xJ -C "$HOME/.local"
    rm -rf "$dir"
    mv "$HOME/.local/$tarball" "$dir"
  fi
  export PATH="$dir/bin:$PATH"
}

install_pi() {
  if have pi; then
    return 0
  fi
  log "install pi with npm"
  npm install -g --ignore-scripts @earendil-works/pi-coding-agent
}

write_bazelrc_local() {
  local file="$root/.bazelrc.local"
  if [[ -e $file ]]; then
    log "keep the existing $file"
    return 0
  fi
  log "write $file for the Kaggle T4 host"
  cat >"$file" <<'EOF'
# Kaggle host file, written by tools/kaggle/bootstrap.sh.
# sm_75 for the T4, and a budget for a small container. See docs/host-portability.md.
build --@rules_cuda//cuda:archs=sm_75
build --jobs=2
build --local_resources=memory=2048
startup --host_jvm_args=-Xmx1024m
EOF
}

# Writes the host contract once, so the notebook and the terminal read the same
# values. See docs/host-portability.md section 8.3.
write_env_file() {
  local file="$HOME/.nanochat.env"
  {
    echo "export NANOCHAT_SANDBOX_BACKEND=\"${NANOCHAT_SANDBOX_BACKEND:-none}\""
    echo "export NANOCHAT_CPP_BACKEND=\"${NANOCHAT_CPP_BACKEND:-cuda}\""
    echo "export NANOCHAT_CPP_PRECISION=\"${NANOCHAT_CPP_PRECISION:-fp16}\""
    echo "export NANOCHAT_CUDA_ARCH=\"${NANOCHAT_CUDA_ARCH:-sm_75}\""
    echo "export NANOCHAT_CPP_CACHE=\"${NANOCHAT_CPP_CACHE:-$HOME/.cache/nanochat_cpp}\""
    echo "export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\""
  } >"$file"
  log "wrote $file"
}

main() {
  log "prepare the Kaggle session"
  install_packages
  setup_cuda
  install_bazelisk
  install_node
  install_pi
  write_bazelrc_local
  write_env_file
  export NANOCHAT_SANDBOX_BACKEND="${NANOCHAT_SANDBOX_BACKEND:-none}"
  log "sandbox backend: $NANOCHAT_SANDBOX_BACKEND"
  log "node: $(node --version 2>/dev/null || echo missing)"
  log "pi:   $(pi --version 2>/dev/null || echo missing)"
  "$root/tools/nanochat" doctor
}

# Run only when the script is executed, not when a test sources it.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
