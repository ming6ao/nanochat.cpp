#!/usr/bin/env bash
# T0 test for tools/prune_bazel_output_bases.
#
# Hermetic: it builds a synthetic Bazel output user root under TEST_TMPDIR and
# asserts the reaper's classification and removal semantics. No GPU, no network,
# and it never touches the real ~/.cache/bazel (every call passes
# --output-root). A live Bazel server is simulated with a process whose comm is
# `java` and whose argv carries `A-server.jar --output_base=<base>`, which is
# exactly what the helper's liveness scan looks for.
#
# usage: prune_test.sh <path to prune_bazel_output_bases>
set -euo pipefail

helper=${1:?usage: prune_test.sh <path to prune_bazel_output_bases>}
[[ -f "$helper" ]] || {
  echo "FAIL: helper not found: $helper" >&2
  exit 1
}

tmp=${TEST_TMPDIR:-$(mktemp -d)}
root=$tmp/root
mkdir -p "$root"

fake_pid=""
cleanup() {
  if [[ -n "$fake_pid" ]] && kill -0 "$fake_pid" 2>/dev/null; then
    kill "$fake_pid" 2>/dev/null || true
    wait "$fake_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

run() { bash "$helper" --output-root "$root" "$@"; }
fail() {
  echo "FAIL: $*" >&2
  exit 1
}
assert_has() { # haystack needle message
  grep -qF -- "$2" <<<"$1" || fail "$3"
}
assert_lacks() { # haystack needle message
  if grep -qF -- "$2" <<<"$1"; then fail "$3"; fi
}

# mkbase <workspace path> -> prints the md5-named base directory, as Bazel names it
mkbase() {
  local ws=$1 name
  name=$(printf '%s' "$ws" | md5sum | cut -d' ' -f1)
  mkdir -p "$root/$name/execroot"
  printf '%s' "$ws" >"$root/$name/execroot/DO_NOT_BUILD_HERE"
  printf '%s\n' "$root/$name"
}

# --- fixtures --------------------------------------------------------------
mkdir -p "$tmp/live-ws/nanochat.cpp"     # exists, kept
mkdir -p "$tmp/wtree-keep/nanochat.cpp"  # exists, reaped only via --worktree
mkdir -p "$tmp/other-ws/nanochat.cpp"    # exists, unrelated
b_live=$(mkbase "$tmp/live-ws/nanochat.cpp")
b_gone=$(mkbase "$tmp/gone-ws/nanochat.cpp") # workspace missing
b_wtree=$(mkbase "$tmp/wtree-keep/nanochat.cpp")
b_other=$(mkbase "$tmp/other-ws/nanochat.cpp")
b_fake=$(mkbase "$tmp/fake-ws/nanochat.cpp") # workspace missing; faked live below

# --- 1. global dry run targets only vanished workspaces --------------------
out=$(run)
assert_has "$out" "$b_gone" "global dry run must list the vanished-workspace base"
assert_has "$out" "$b_fake" "global dry run must list the other vanished base"
assert_lacks "$out" "$b_live" "global dry run must not list a live workspace"
assert_lacks "$out" "$b_other" "global dry run must not list an unrelated base"
[[ -d "$b_gone" ]] || fail "dry run must not remove anything"

# --- 2. --worktree scopes to a directory, existing or not ------------------
out=$(run --worktree "$tmp/wtree-keep")
assert_has "$out" "$b_wtree" "--worktree must target a base under the directory"
assert_lacks "$out" "$b_gone" "--worktree must not reach outside the directory"

# --- 3. --json reports status and workspace --------------------------------
out=$(run --json)
assert_has "$out" '"status": "target"' "--json must report the target status"
assert_has "$out" '"workspace":' "--json must report a workspace field"
assert_has "$out" "$tmp/gone-ws/nanochat.cpp" "--json must name the vanished workspace"

# --- 4. a base held by a live server is skipped, even with --apply ---------
cp /bin/bash "$tmp/java"
"$tmp/java" -c 'sleep 30; true' A-server.jar "--output_base=$b_fake" &
fake_pid=$!
sleep 0.3
out=$(run --apply)
assert_has "$out" "skip (live server) $b_fake" "a live-server base must be skipped"
assert_has "$out" "removed $b_gone" "--apply must still remove the other orphan"
[[ -d "$b_fake" ]] || fail "a live-server base must survive --apply"
[[ ! -d "$b_gone" ]] || fail "--apply must remove the non-live orphan"
kill "$fake_pid" 2>/dev/null || true
wait "$fake_pid" 2>/dev/null || true
fake_pid=""

# --- 5. once the server is gone, the last orphan is reaped ------------------
out=$(run --apply)
assert_has "$out" "removed $b_fake" "--apply must remove the now-dead orphan"
[[ ! -d "$b_fake" ]] || fail "--apply must remove the now-dead orphan base"
[[ -d "$b_live" ]] || fail "--apply must keep a live-workspace base"
[[ -d "$b_wtree" ]] || fail "--apply must keep a base outside the global scope"
[[ -d "$b_other" ]] || fail "--apply must keep an unrelated base"

# --- 6. empty sweep and error paths ---------------------------------------
out=$(run --json)
[[ "$out" == "[]" ]] || fail "an empty sweep must print [] (got: $out)"

if run --nonsense >/dev/null 2>&1; then fail "an unknown argument must exit non-zero"; fi
if bash "$helper" --output-root "$tmp/does-not-exist" >/dev/null 2>&1; then
  fail "a missing output root must exit non-zero"
fi

echo "PASS: prune_bazel_output_bases"
