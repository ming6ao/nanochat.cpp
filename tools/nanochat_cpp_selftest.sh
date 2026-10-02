#!/usr/bin/env bash
# T0 test for the Python bridge's torch-free logic: configuration math,
# token-count estimate, and the NANO shard round-trip. It runs with the system
# python3 because nothing here imports torch or the reference package. See
# docs/python-bridge.md.
set -euo pipefail

export PYTHONPATH="${TEST_SRCDIR}/${TEST_WORKSPACE}/python${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m nanochat_cpp.selftest
