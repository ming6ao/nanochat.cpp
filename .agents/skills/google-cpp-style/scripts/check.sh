#!/usr/bin/env bash
# Run the nanochat.cpp style gate.
#
# See .agents/skills/google-cpp-style/SKILL.md. Pass file paths to check a
# subset; with no arguments the gate checks every tracked C++ source.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
exec "$root/tools/nanochat" lint "$@"
