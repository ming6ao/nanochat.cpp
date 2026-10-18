"""Reap orphaned Bazel output bases.

``tools/nanochat prune`` calls this module. Bazel names each output base
``md5(workspace_path)`` under the output user root
(``~/.cache/bazel/_bazel_$USER`` by default). Every git worktree is a distinct
workspace, so it gets its own output base that survives the worktree. This
module finds the bases whose recorded workspace
(``execroot/DO_NOT_BUILD_HERE``) no longer exists, or, with ``--worktree``,
that are at or under a given directory. It removes them on request.

The default is a dry run. ``--apply`` performs the removals. A base held by a
live Bazel server is never removed. When its workspace still exists the module
shuts the server down first, and it re-checks either way. See
``docs/sandbox.md``, "Reaping Bazel output bases".
"""

from __future__ import annotations

import argparse
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "server_holds"]

#: The marker that records the workspace path inside an output base.
MARKER = Path("execroot") / "DO_NOT_BUILD_HERE"

#: The Bazel server main class. A server command line carries this token.
SERVER_JAR = "A-server.jar"

USAGE = """tools/nanochat prune [--worktree DIR] [--apply] [--json]
                        [--sizes] [--output-root DIR]

  --worktree DIR    only consider output bases whose workspace is DIR or under
                    it (the teardown case). Without it, only bases whose
                    workspace no longer exists are considered.
  --apply           remove the bases (default: dry run, removes nothing)
  --json            emit a JSON array instead of human-readable lines
  --sizes           report size per base (implies reading every file; slower)
  --output-root DIR override the Bazel output user root
                    (default $HOME/.cache/bazel/_bazel_$USER, or
                    $NANOCHAT_BAZEL_OUTPUT_ROOT)
"""


def default_output_root() -> str:
    """The Bazel output user root, unless the environment overrides it."""
    override = os.environ.get("NANOCHAT_BAZEL_OUTPUT_ROOT")
    if override:
        return override
    user = pwd.getpwuid(os.getuid()).pw_name
    return str(Path.home() / ".cache" / "bazel" / f"_bazel_{user}")


def server_holds(base: str) -> bool:
    """True when a live Bazel server currently holds this output base.

    The server command line carries ``--output_base=<abs path>``. Only a
    process whose executable name is ``java`` and whose command line names
    ``A-server.jar`` is considered, so a shell that merely mentions the path
    cannot be mistaken for a server.
    """
    wanted = f"--output_base={base}".encode()
    jar = SERVER_JAR.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_bytes().strip() != b"java":
                continue
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if jar not in raw:
            continue
        if wanted in raw.split(b"\0"):
            return True
    return False


def _canonical(path: str) -> str:
    """``realpath -m``: normalise without touching the filesystem."""
    return os.path.realpath(path)


def _size_kb(base: Path) -> int:
    """The base size in kibibytes, or 0 when the measurement fails."""
    try:
        result = subprocess.run(["du", "-sk", "--", str(base)],
                                capture_output=True, text=True, check=False)
    except OSError:
        return 0
    first = result.stdout.split("\t")[0].strip()
    try:
        return int(first)
    except ValueError:
        return 0


def _megabytes(kilobytes: int) -> str:
    return f"{kilobytes / 1024:.1f}"


def _json_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _targets(output_root: Path, worktree: str) -> list[tuple[Path, str]]:
    """The output bases in scope: a dry-run or apply target, with workspace."""
    found: list[tuple[Path, str]] = []
    for base in sorted(output_root.iterdir()):
        if not base.is_dir():
            continue
        marker = base / MARKER
        if not marker.is_file():
            continue
        workspace = marker.read_text(encoding="utf-8", errors="replace")
        workspace = workspace.rstrip("\n")
        if not workspace:
            continue
        workspace = _canonical(workspace)
        if worktree:
            if not (workspace == worktree or
                    workspace.startswith(worktree + os.sep)):
                continue
        elif os.path.isdir(workspace):
            continue
        found.append((base, workspace))
    return found


def _stop_server(workspace: str) -> None:
    """Ask the server of a still-present workspace to stop. Ignore failures."""
    try:
        subprocess.run(["bazel", "shutdown"], cwd=workspace,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    except OSError:
        return


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, classify the bases, and reap them."""
    parser = argparse.ArgumentParser(
        prog="tools/nanochat prune", usage=USAGE,
        description="Reap orphaned Bazel output bases.")
    parser.add_argument("--worktree", default="")
    parser.add_argument("--output-root", default=default_output_root())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--sizes", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        print(f"prune: no output user root at {output_root}", file=sys.stderr)
        return 2
    worktree = _canonical(args.worktree) if args.worktree else ""

    targets = _targets(output_root, worktree)
    if not targets:
        if args.as_json:
            print("[]")
        return 0

    records: list[str] = []
    removed = 0
    skipped = 0
    total_kb = 0

    for base, workspace in targets:
        status = "target"
        if server_holds(str(base)):
            # A server owns it. Stop it through its own workspace when that
            # still exists; otherwise leave it for the idle timeout. Re-check
            # either way.
            if args.apply and os.path.isdir(workspace):
                _stop_server(workspace)
            if server_holds(str(base)):
                status = "live"

        if status == "live":
            skipped += 1
            if not args.as_json:
                print(f"skip (live server) {base} <- {workspace}")
            records.append(
                f'  {{"path": "{_json_escape(str(base))}", '
                f'"workspace": "{_json_escape(workspace)}", '
                f'"status": "live", "bytes": null}}')
            continue

        kilobytes = 0
        if args.apply or args.sizes:
            kilobytes = _size_kb(base)
            total_kb += kilobytes
        megabytes = _megabytes(kilobytes)

        if args.apply:
            try:
                shutil.rmtree(base)
            except OSError as error:
                print(f"prune: cannot remove {base}: {error}", file=sys.stderr)
                return 1
            removed += 1
            if not args.as_json:
                print(f"removed {base} <- {workspace} ({megabytes} MB)")
        elif args.sizes:
            if not args.as_json:
                print(f"would remove {base} <- {workspace} ({megabytes} MB)")
        elif not args.as_json:
            print(f"would remove {base} <- {workspace}")
        records.append(
            f'  {{"path": "{_json_escape(str(base))}", '
            f'"workspace": "{_json_escape(workspace)}", '
            f'"status": "target", "bytes": {kilobytes}}}')

    if args.as_json:
        sys.stdout.write("[\n")
        sys.stdout.write(",\n".join(records))
        sys.stdout.write("\n]\n")
    elif args.apply:
        print(f"prune: removed {removed} base(s), {skipped} skipped (live), "
              f"{_megabytes(total_kb)} MB reclaimed")
    else:
        print(f"prune: {len(targets)} base(s) would be removed, {skipped} "
              f"skipped (live). Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
