#!/usr/bin/env python3
"""Run the ``analyze`` phase of the profile-guided optimization loop.

The loop is ``nanochat_cpp.pgo``; this file only finds the tree and calls it.
See ``docs/performance.md`` (Part 1) and the skill's ``SKILL.md``.
"""

from __future__ import annotations

import _cli

if __name__ == "__main__":
    raise SystemExit(_cli.main("analyze"))
