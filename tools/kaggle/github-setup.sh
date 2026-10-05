#!/usr/bin/env bash
# Authenticate git and gh, then clone nanochat.cpp on a Kaggle Notebook.
#
# usage: tools/kaggle/github-setup.sh
#
# Requires GH_TOKEN, from a Kaggle Notebook secret, and NANOCHAT_REPO set to
# owner/nanochat.cpp. Sets the git identity from GIT_AUTHOR_NAME and
# GIT_AUTHOR_EMAIL, runs `gh auth setup-git` so git uses the token, and clones
# the repository into /kaggle/working. A second run fetches instead of cloning.
# See docs/kaggle.md.
set -euo pipefail

GH_VERSION=${NANOCHAT_KAGGLE_GH_VERSION:-2.63.2}
: "${NANOCHAT_REPO:?set NANOCHAT_REPO to owner/nanochat.cpp}"
: "${GH_TOKEN:?set GH_TOKEN from a Kaggle secret}"
REPO=$NANOCHAT_REPO
DEST=${NANOCHAT_DEST:-/kaggle/working/nanochat.cpp}
CLONE_URL=${NANOCHAT_CLONE_URL:-https://github.com/$REPO}

export PATH="$HOME/.local/bin:$PATH"

log() { printf 'kaggle: %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

install_gh() {
  if have gh; then
    return 0
  fi
  log "install gh $GH_VERSION into ~/.local/bin"
  local tmp url
  mkdir -p "$HOME/.local/bin"
  tmp=$(mktemp -d)
  url="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz"
  curl -fsSL "$url" | tar -xz -C "$tmp"
  install -m 0755 "$tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" "$HOME/.local/bin/gh"
  rm -rf "$tmp"
}

# Git uses the token through the gh credential helper. Do not put the token in
# the remote URL: it leaks into `git remote -v` and into the model context.
setup_git() {
  gh auth setup-git
  git config --global user.name "${GIT_AUTHOR_NAME:-Kaggle Agent}"
  git config --global user.email "${GIT_AUTHOR_EMAIL:-kaggle@example.com}"
}

clone_or_fetch() {
  if [[ -d $DEST/.git ]]; then
    log "fetch $DEST"
    git -C "$DEST" fetch --all --prune
  else
    log "clone $CLONE_URL into $DEST"
    git clone "$CLONE_URL" "$DEST"
  fi
  git -C "$DEST" remote set-url origin "$CLONE_URL"
}

install_gh
setup_git
clone_or_fetch
gh auth status
log "ready: $DEST"
