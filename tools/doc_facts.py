#!/usr/bin/env python3
"""Check the fact tables and the prose rules of ``docs/build.md``.

The build document states design, and it holds two tables that a reader can
also derive from the build itself: the subcommand table of the one stage
binary, and the table of named build configurations. Both sit between a
``BEGIN GENERATED`` and an ``END GENERATED`` marker pair, so this script can
compare them without touching the prose around them.

    tools/doc_facts.py --check          # compare, and name every disagreement
    tools/doc_facts.py --write          # rewrite the two marked regions
    tools/doc_facts.py --check --root <a copy of the tree>

``--check`` also holds four prose rules. The document states design, so it
holds no file position, no calendar date outside its ``Logs`` section, and no
label that a reader must resolve in another document. Every link with a
fragment resolves to a heading.

The rules are deliberate. A file position and a date go stale, and a bare label
such as a zone name hides its meaning in a table that a reader has to find
first. The log is the one exception: it records a real run, and a run has a day.

Exit status: 0 when the document agrees with the tree, 1 with one line per
disagreement, 2 for a usage error. Standard library only. See
``docs/build.md`` and ``docs/README.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

__all__ = ["main"]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The document this script owns, relative to the repository root.
DOC = "docs/build.md"

#: The tracked Bazel configuration: the authority for the configuration table.
BAZELRC = ".bazelrc"

#: The subcommand table of the stage binary: the authority for section one.
COMMANDS = "src/commands.cc"

#: The four build values the configuration table must state, in column order.
#: The simulator column names an engine and not a value, so it stays prose.
BUILD_VALUES = ("arch", "precision", "backend", "collective")

#: The column headings of the configuration table, by build value.
COLUMN = {
    "arch": "Architecture",
    "precision": "Precision",
    "backend": "Backend",
    "collective": "Collective",
}

#: A generated region, keyed by name.
REGION = re.compile(
    r"<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n"
    r"(?P<body>.*?)\n"
    r"<!-- END GENERATED: (?P=name) -->",
    re.S,
)

#: One row of the subcommand table in the C++ source.
COMMAND_ROW = re.compile(
    r'\{\s*"(?P<name>[a-z_0-9]+)",\s*"[A-Za-z_0-9]+",\s*'
    r'"(?P<summary>(?:[^"\\]|\\.)*)"'
)

#: One ``--define`` line of the tracked Bazel configuration.
DEFINE_LINE = re.compile(
    r"^build(?::(?P<config>[A-Za-z0-9_-]+))?\s+--define=(?P<key>[A-Za-z0-9_]+)="
    r"(?P<value>.*)$"
)

#: One architecture line of the tracked Bazel configuration.
ARCH_LINE = re.compile(
    r"^build(?::(?P<config>[A-Za-z0-9_-]+))?\s+"
    r"--@rules_cuda//cuda:archs=(?P<value>.*)$"
)

#: A define name that carries a build value, mapped to its column key.
DEFINE_KEY = {
    "precision": "precision",
    "backend": "backend",
    "collective": "collective",
}

#: The retired labels: a short form whose meaning lives in another document.
RETIRED = (
    (re.compile(r"\bZ[0-3]\b"), "a retired zone label"),
    (re.compile(r"\bP[4-7]\b"), "a retired plan row label"),
    (re.compile(r"\bD\d{1,2}\b"), "a retired decision label"),
    (re.compile(r"\bR\d{1,2}\b"), "a retired review row label"),
    (re.compile(r"\bE[1-4]\b"), "a retired layer label"),
    (re.compile(r"\bG[1-5]\b"), "a retired gate label"),
    (re.compile(r"\bS\d\b"), "a retired stage label"),
    (re.compile(r"\bT[0-3]\b"), "a retired test tier label"),
)

#: A prose position in a file, which goes stale.
LINE_REFERENCE = re.compile(r"\bline[s]?\s+\d+\b", re.I)
FILE_POSITION = re.compile(
    r"[\w./-]+\.(?:py|cc|h|cu|cuh|md|json|toml|bzl|bazelrc|txt|sh):\d+"
)

#: A calendar date, which belongs in the log and nowhere else.
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")

#: A Markdown link, with its target.
LINK = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")

#: A Markdown heading, and its level.
HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*$", re.M)

#: The heading that starts the log, where a date is allowed.
LOG_HEADING = "Logs"


def _slug(text: str) -> str:
    """The fragment a Markdown renderer gives a heading."""
    lowered = text.strip().lower()
    without_marks = re.sub(r"[^\w\s-]", "", lowered)
    return re.sub(r"\s+", "-", without_marks)


def _headings(text: str) -> list[tuple[int, str, str]]:
    """Every heading of a document as ``(level, text, fragment)``."""
    found = []
    for match in HEADING.finditer(text):
        found.append((len(match.group("hashes")), match.group("text"),
                      _slug(match.group("text"))))
    return found


def _prose_scope(text: str) -> tuple[str, int]:
    """The text before the log heading, and the line the log starts on."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"## {LOG_HEADING}":
            return "\n".join(lines[:index]), index + 1
    return text, len(lines) + 1


def _configurations(lines: list[str]) -> tuple[dict[str, str], dict[str, dict],
                                               list[str]]:
    """The default build values, the per-config overlay, and the config order.

    The values come from the tracked Bazel configuration alone. A named
    configuration is additive, so its effective value is the overlay on the
    default build.
    """
    default: dict[str, str] = {}
    overlay: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line.startswith("build"):
            continue
        match = DEFINE_LINE.match(line)
        if match:
            config = match.group("config") or ""
            if config:
                _remember(order, config)
            key = DEFINE_KEY.get(match.group("key"))
            if key is None:
                continue
            value = match.group("value").strip().strip('"')
            if config:
                overlay.setdefault(config, {})[key] = value
            else:
                default[key] = value
            continue
        arch = ARCH_LINE.match(line)
        if arch:
            config = arch.group("config") or ""
            if config:
                _remember(order, config)
            value = arch.group("value").strip().strip('"')
            if config:
                overlay.setdefault(config, {})["arch"] = value
            else:
                default["arch"] = value

    return default, overlay, order


def _remember(order: list[str], name: str) -> None:
    if name not in order:
        order.append(name)


def _effective(default: dict[str, str], overlay: dict[str, dict],
               name: str) -> dict[str, str]:
    values = dict(default)
    values.update(overlay.get(name, {}))
    return values


def _commands(path: Path) -> list[tuple[str, str]]:
    """The subcommand table of the stage binary, in help order."""
    return [(match.group("name"), match.group("summary"))
            for match in COMMAND_ROW.finditer(path.read_text(encoding="utf-8"))]


def _rows(body: str) -> list[list[str]]:
    """The data rows of a Markdown table, without its header and separator."""
    cells: list[list[str]] = []
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in row):
            continue
        cells.append(row)
    return cells[1:]


def _config_name(cell: str) -> str:
    """The configuration name of the first cell of a table row."""
    name = cell.strip().strip("`").strip()
    if name.startswith("--config="):
        name = name[len("--config="):]
    if name.strip("_ ") in ("the default build", ""):
        return ""
    return name.strip().strip("_")


def _value(cell: str) -> str:
    return cell.strip().strip("`").strip().lower()


def _label(name: str) -> str:
    """The row label of a configuration, as the document writes it."""
    return f"`--config={name}`" if name else "_the default build_"


def _check_configurations(text: str, default: dict[str, str],
                          overlay: dict[str, dict], order: list[str]) -> list[str]:
    """Every configuration row states the build the configuration gives."""
    failures: list[str] = []
    region = _region(text, "build-configurations")
    if region is None:
        return [f"{DOC}: no build-configurations region"]

    seen: set[str] = set()
    for row in _rows(region):
        if not row:
            continue
        name = _config_name(row[0])
        seen.add(name)
        if len(row) < 1 + len(BUILD_VALUES):
            failures.append(f"{DOC}: row `{row[0]}` is too short with "
                            f"{len(row)} cells")
            continue
        expected = _effective(default, overlay, name)
        for offset, key in enumerate(BUILD_VALUES, start=1):
            want = expected.get(key)
            if want is None:
                continue
            got = _value(row[offset])
            if got != _value(want):
                failures.append(
                    f"{DOC}: row {_label(name)} states {COLUMN[key]} "
                    f"{row[offset]}, the configuration gives {want}")

    for name in order:
        if name not in seen:
            failures.append(
                f"{DOC}: no row for `--config={name}`, which {BAZELRC} declares")
    return failures


def _check_commands(text: str, commands: list[tuple[str, str]]) -> list[str]:
    """The subcommand table matches the source, in order and in summary."""
    failures: list[str] = []
    region = _region(text, "subcommands")
    if region is None:
        return [f"{DOC}: no subcommands region"]

    rows = [(row[0].strip().strip("`"), row[1] if len(row) > 1 else "")
            for row in _rows(region) if len(row) > 1]
    names = [name for name, _ in rows]
    expected = [name for name, _ in commands]
    if names != expected:
        for index, (got, want) in enumerate(zip(names, expected)):
            if got != want:
                failures.append(
                    f"{DOC}: subcommand row {index + 1} is `{got}`, "
                    f"{COMMANDS} has `{want}`")
                break
        else:
            extra = names[len(expected):] or expected[len(names):]
            failures.append(
                f"{DOC}: {len(names)} subcommand rows, {COMMANDS} has "
                f"{len(expected)}; the difference starts at `{extra[0]}`")
        return failures

    for (name, summary), (_, want) in zip(rows, commands):
        if summary != want:
            failures.append(
                f"{DOC}: subcommand `{name}` reads \"{summary}\", "
                f"{COMMANDS} says \"{want}\"")
    return failures


def _check_prose(text: str) -> list[str]:
    """The design text holds no stale position, date, or bare label."""
    failures: list[str] = []
    scope, _ = _prose_scope(text)
    lines = scope.splitlines()

    for number, line in enumerate(lines, start=1):
        for pattern, hint in RETIRED:
            for match in pattern.finditer(line):
                failures.append(
                    f"{DOC}:{number}: `{match.group(0)}` is {hint}; "
                    f"name the subject instead")
        for pattern, hint in ((LINE_REFERENCE, "a line number"),
                              (FILE_POSITION, "a file position"),
                              (DATE, "a calendar date before the log")):
            for match in pattern.finditer(line):
                failures.append(
                    f"{DOC}:{number}: {hint} (`{match.group(0)}`); the "
                    f"document states design, and the log holds a dated run")
    return failures


def _check_links(text: str, root: Path) -> list[str]:
    """Every link with a fragment reaches a heading of its target."""
    failures: list[str] = []
    source = root / DOC
    for match in LINK.finditer(text):
        target = match.group("target")
        if "#" not in target:
            continue
        path, fragment = target.split("#", 1)
        if not fragment:
            continue
        if path:
            destination = (source.parent / path).resolve()
            if not destination.is_file():
                failures.append(f"{DOC}: link target {target} has no file")
                continue
            body = destination.read_text(encoding="utf-8")
            where = path
        else:
            body = text
            where = DOC
        if fragment not in {fragment_of for _, _, fragment_of in _headings(body)}:
            line = text.count("\n", 0, match.start()) + 1
            failures.append(
                f"{DOC}:{line}: {where} has no heading for #{fragment}")
    return failures


def _region(text: str, name: str) -> str | None:
    for match in REGION.finditer(text):
        if match.group("name") == name:
            return match.group("body")
    return None


def _write_subcommands(commands: list[tuple[str, str]]) -> str:
    lines = ["| Subcommand | Purpose |", "|---|---|"]
    lines += [f"| `{name}` | {summary} |" for name, summary in commands]
    return "\n".join(lines)


def _write_configurations(text: str, default: dict[str, str],
                          overlay: dict[str, dict], order: list[str]) -> str:
    """The configuration table, with the prose simulator cell preserved."""
    keep: dict[str, str] = {}
    region = _region(text, "build-configurations")
    if region is not None:
        for row in _rows(region):
            if len(row) > 1 + len(BUILD_VALUES):
                keep[_config_name(row[0])] = row[1 + len(BUILD_VALUES)]

    header = "| Configuration | " + " | ".join(
        COLUMN[key] for key in BUILD_VALUES) + " | Simulator |"
    lines = [header, "|" + "---|" * (len(BUILD_VALUES) + 2)]

    rows = [("", "_the default build_")]
    rows += [(name, f"`--config={name}`") for name in order]
    for name, label in rows:
        values = _effective(default, overlay, name)
        cells = [label]
        for key in BUILD_VALUES:
            value = values.get(key, "")
            cells.append(f"`{value}`" if key == "arch" else value)
        cells.append(keep.get(name, "off"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _replace_region(text: str, name: str, body: str) -> str:
    def swap(match: re.Match) -> str:
        if match.group("name") != name:
            return match.group(0)
        return (f"<!-- BEGIN GENERATED: {name} -->\n{body}\n"
                f"<!-- END GENERATED: {name} -->")

    return REGION.sub(swap, text)


def _report(failures: list[str]) -> int:
    for failure in failures:
        print(failure)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    root = REPO_ROOT
    mode = "check"
    argument = iter(argv)
    for flag in argument:
        if flag in ("--check", "--write"):
            mode = flag[2:]
        elif flag in ("-h", "--help"):
            print(__doc__)
            return 0
        elif flag == "--root":
            root = Path(next(argument, "")).resolve()
        elif flag.startswith("--root="):
            root = Path(flag[len("--root="):]).resolve()
        else:
            print(f"doc_facts: unknown argument {flag}", file=sys.stderr)
            return 2

    doc = root / DOC
    if not doc.is_file():
        print(f"{DOC}: missing", file=sys.stderr)
        return 1
    text = doc.read_text(encoding="utf-8")
    commands = _commands(root / COMMANDS)
    default, overlay, order = _configurations(
        (root / BAZELRC).read_text(encoding="utf-8").splitlines())

    if mode == "write":
        updated = _replace_region(
            text, "subcommands", _write_subcommands(commands))
        updated = _replace_region(
            updated, "build-configurations",
            _write_configurations(text, default, overlay, order))
        if updated == text:
            print(f"{DOC}: the marked regions already agree with the tree")
            return 0
        doc.write_text(updated, encoding="utf-8")
        print(f"{DOC}: rewrote the marked regions")
        return 0

    failures = _check_commands(text, commands)
    failures += _check_configurations(text, default, overlay, order)
    failures += _check_prose(text)
    failures += _check_links(text, root)
    if not failures:
        print(f"{DOC}: the tables and the links agree with the tree")
    return _report(failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
