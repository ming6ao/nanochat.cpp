#!/usr/bin/env bash
# T0 syntax test for the Kaggle host scripts. Hermetic and CPU-only.
#
# usage: scripts_test.sh <script...>
set -euo pipefail
for script in "$@"; do
  bash -n "$script"
done
echo "kaggle scripts: OK ($# files)"
