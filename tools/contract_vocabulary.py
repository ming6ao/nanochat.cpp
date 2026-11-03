#!/usr/bin/env python3
"""Generate the capability rows that are data, and check the rows that are
prose.

``docs/api-surface.md`` R4a splits the capability vocabulary by *content*.
``spec/capability_bits.md`` section 3 is the authority for the fifteen names and
their order, and ``include/nanochat/tensor.h`` is the authority for the values.
A table that holds nothing but a name, a bit and a position is therefore
derived, and a table that also carries a hand-written sentence is not:

* **emitted** -- ``backends/cpu/caps_names.h``, ``backends/cuda/caps_names.h``,
  the six closed lists in ``backends/cpu/caps_test.cc``, and ``kBackendNames``
  in ``src/device_profile.cc``. Each holds a name and a bit and nothing else.
* **checked** -- the rows of ``src/capability_names.h``, which keep a
  hand-written ``consequence`` sentence per row that the spec does not supply,
  and the ``features[]`` array of every ``backends/*/caps.json``. The check
  compares the name, the bit, and the order, and names the file and the row.

Both modes read the same parse, so a name that is added to the spec and not to
the header is one failure with one message, not two.

Every emitted region sits between a ``BEGIN GENERATED`` and an ``END GENERATED``
marker pair, so the prose around it survives a rewrite. ``--write`` replaces the
bytes between the markers and touches nothing else; running it twice leaves the
tree unchanged.

The same ``--check`` also runs the two seam inventory checks of
``docs/api-surface.md`` R14, because they consume the same parse:

* **kind half** -- ``spec/block_kinds.md`` section 2 names a kind that
  ``src/model_spec.cc`` does not accept, or ``src/model_spec.cc`` accepts a
  spelling that has no prose row. The reader is the field-list authority
  (``include/nanochat/model_spec.h``), so the divergences are an explicit alias
  table with a comment per row rather than a silent rename.
* **primitive half** -- a ``caps.json`` ``primitives[]`` entry has no symbol in
  ``include/nanochat/kernels.h``, or a primitive-tier seam symbol has no
  ``primitives[]`` entry. One tier table classifies all thirty seam symbols, so
  a symbol added to the seam without a tier row fails here.

Usage::

    # Rewrite every generated region from the spec and the header.
    python3 tools/contract_vocabulary.py --write

    # Check the hand-written rows, the block kinds, and the seam inventory.
    # Exits non-zero on a mismatch, naming the file and the row.
    python3 tools/contract_vocabulary.py --check

The script is standard-library only and hermetic: it reads the repository and
writes only the generated regions of the four files named above.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The prose authority for the fifteen names and their order.
SPEC_BITS = "spec/capability_bits.md"

#: The authority for the values: one `kFeature*` constant per bit.
TENSOR_H = "include/nanochat/tensor.h"

#: The authority for the seam inventory the six closed lists are read from.
KERNELS_H = "include/nanochat/kernels.h"

#: The prose authority for the block kinds (`docs/api-surface.md` R14).
SPEC_KINDS = "spec/block_kinds.md"

#: The reader whose closed block-entry vocabularies the kind half compares.
MODEL_SPEC_CC = "src/model_spec.cc"

#: The hand-written table the check mode compares against the parse.
CAPABILITY_NAMES_H = "src/capability_names.h"

#: The device declarations whose `features[]` array the check mode compares.
CAPS_JSON = ("backends/cpu/caps.json", "backends/cuda/caps.json")

#: The generated regions, one marker pair per region.
CPU_CAPS_NAMES_H = "backends/cpu/caps_names.h"
CUDA_CAPS_NAMES_H = "backends/cuda/caps_names.h"
CPU_CAPS_TEST_CC = "backends/cpu/caps_test.cc"
DEVICE_PROFILE_CC = "src/device_profile.cc"

BEGIN = "// BEGIN GENERATED (tools/contract_vocabulary.py --write)"
END = "// END GENERATED (tools/contract_vocabulary.py --write)"

#: `clang-format` is switched off inside a generated region: the script cannot
#: reproduce the formatter's column packing, and a generated region must be a
#: fixed point of the generator rather than of the formatter.
FORMAT_OFF = "// clang-format off"
FORMAT_ON = "// clang-format on"

#: The header spellings that differ from the mechanical rule
#: (`kFeature` + upper camel case of the vocabulary name).
#: `spec/capability_bits.md` section 2 states the rule;
#: `include/nanochat/tensor.h` spells these two differently, so the row records
#: what the header says. Keyed by vocabulary name, valued by the header's
#: constant.
HEADER_SPELLINGS = {
    # The header predates the `stateful_` prefix: bit 10 is `kFeatureAdamW`.
    "stateful_adamw": "kFeatureAdamW",
    # As above: bit 11 is `kFeatureMuon`.
    "stateful_muon": "kFeatureMuon",
}

#: The group whose bits a model never requires and the reference backend never
#: offers (`spec/capability_bits.md` section 3, "Declared by": an override
#: bit is device-only). The reference feature set excludes it.
OVERRIDE_GROUP = "Override"

#: The precision pair the reference backend does not claim: it has no
#: tensor-core fp8 path, and its fp8 emulation is the oracle a GPU's fp8 bits
#: are measured against rather than a mechanism it offers.
FP8_PAIR = ("fp8_storage", "fp8_gemm")

#: The seam section whose six entries are the device utilities
#: (`docs/kernels.md`: Alloc, Free, Memcpy, Memset, Synchronize, GetCaps).
UTILITY_SECTION = "Device / memory"

#: The reader's closed block-entry vocabularies, one per group of
#: `spec/block_kinds.md` section 1. `kDTypeRows` is deliberately absent: a
#: per-block dtype policy is a field of the entry, not a kind
#: (`spec/block_kinds.md` section 6).
READER_KIND_TABLES = (
    "kNormRows",
    "kAttentionRows",
    "kVariantRows",
    "kPositionRows",
    "kMlpRows",
    "kResidualRows",
)

#: `src/model_spec.cc` accepts these spellings; `spec/block_kinds.md` section 2
#: names the kind each one stands for, or `None` when it names no kind at all.
#: One row per divergence, with the reason on the row: the reader is the
#: field-list authority (`include/nanochat/model_spec.h`), so the mapping is
#: recorded here rather than papered over by a rename on either side.
KIND_ALIASES: dict[str, tuple[str | None, str]] = {
    # The one spelling the prose and the reader share.
    "rms_norm": ("rms_norm", "the same spelling on both sides"),
    # The prose carries the mask as the `attention_variant` field (`full` or
    # `sliding`); the reader carries it in `AttentionKind` instead, so its two
    # values stand for the one attention kind of section 2.
    "causal": ("gqa_self_attention",
               "the full-context mask of the one attention kind; the prose "
               "calls the mask `attention_variant = full`"),
    "sliding_window": ("gqa_self_attention",
                       "the windowed mask of the same kind; section 2 states "
                       "that the variant is a field and not a kind"),
    # The prose's one attention kind carries the head mapping; the reader's
    # `AttentionVariant` spells the two cases of that mapping.
    "mha": ("gqa_self_attention",
            "the 1:1 head mapping, the degenerate case of the one attention "
            "kind (`num_kv_heads == num_heads`)"),
    "gqa": ("gqa_self_attention",
            "the group head mapping the prose kind is named for"),
    # A field's absence value, not a kind: recorded, not resolved. The prose's
    # closure rule (section 2) says every value a block may name appears there,
    # so this row is an F17 instance for the contract owner.
    "none": (None,
             "the position field's absence value: a block that names it "
             "enters no position into q and k, so section 2 lists no kind for "
             "it. Recorded, not resolved"),
    "rope": ("rope", "the same spelling on both sides"),
    # A spelling divergence `src/model_spec.cc:213-214` does not record: the
    # prose spells the relu-squared MLP `relu2_mlp` and the reader
    # `relu_square`.
    "relu_square": ("relu2_mlp",
                    "the prose spells the relu-squared MLP `relu2_mlp`; the "
                    "reader's spelling is not in its recorded divergence "
                    "list"),
    "plain": ("plain_residual", "the prose spells the plain add "
                               "`plain_residual`"),
    # The prose names the second residual kind for the gate it adds; the
    # reader names it for the ResFormer blend, which the prose row does not
    # carry (`models/nanochat-default/README.md`, contract gap 1).
    "blend": ("value_gated_residual",
              "the header's `kBlend` is the ResFormer residual (per-layer "
              "scale plus learned embedding blend), which the prose's "
              "`value_gated_residual` does not carry"),
}

#: The tier of every symbol in `include/nanochat/kernels.h`, keyed by symbol.
#: The tier names are the tree's own (`docs/kernels.md` "Kernel tiers",
#: `spec/capability_bits.md` section 5), and they say which declaration owns the
#: symbol: `utilities` and `primitives` and `stateful` are the three `caps.json`
#: arrays, `override` is a fused single-kernel replacement of a composition,
#: gated by an `override_*` bit, and `composition` is a composition with no
#: override bit. The primitive tier is the one this script compares against
#: `primitives[]`; the other two arrays are gated by each backend's caps test.
SEAM_TIERS: dict[str, tuple[str, str]] = {
    # `utilities[]`: the six device utilities of docs/kernels.md.
    "Alloc": ("utilities", "device memory"),
    "Free": ("utilities", "device memory"),
    "Memcpy": ("utilities", "device memory"),
    "Memset": ("utilities", "device memory"),
    "Synchronize": ("utilities", "the stream barrier"),
    "GetCaps": ("utilities", "the capability report"),
    # `primitives[]`: the arithmetic tier every device owes, plus the two
    # optional fp8 entries (docs/kernels.md "Required bring-up cost").
    "Gemm": ("primitives", "the forward GEMM"),
    "GemmWgrad": ("primitives", "the weight-gradient GEMM"),
    "QuantizeFp8": ("primitives", "optional: the `fp8_storage` bit"),
    "GemmScaledFp8": ("primitives", "optional: the `fp8_gemm` bit"),
    "RowReduce": ("primitives", "the per-segment reduction"),
    "SpanReduce": ("primitives", "the ragged reduction"),
    "AttentionForward": ("primitives", "causal group-query attention"),
    "AttentionBackward": ("primitives", "its backward"),
    "PointwiseForward": ("primitives", "the nine op codes"),
    "PointwiseBackward": ("primitives", "their backward"),
    "EmbeddingForward": ("primitives", "the gather"),
    "EmbeddingBackward": ("primitives", "the scatter-add"),
    "SmearForward": ("primitives", "the bigram residual"),
    "SmearBackward": ("primitives", "its backward"),
    # `stateful[]`: the two training updates.
    "AdamWUpdate": ("stateful", "the `stateful_adamw` bit"),
    "MuonUpdate": ("stateful", "the `stateful_muon` bit"),
    # The four compositions a backend may serve with one fused kernel, one
    # `override_*` bit each (spec/capability_bits.md bits 12 to 15).
    "RmsNormForward": ("override", "the `override_rms_norm` family"),
    "RmsNormBackward": ("override", "the `override_rms_norm` family"),
    "QkPrepForward": ("override", "the `override_qk_prep` family"),
    "QkPrepBackward": ("override", "the `override_qk_prep` family"),
    "ClassifierForward": ("override", "the `override_classifier` family"),
    "ClassifierBackward": ("override", "the `override_classifier` family"),
    "ValueGateForward": ("override", "a composition with no override bit "
                                       "yet (spec/capability_bits.md 5)"),
    "ValueGateBackward": ("override", "a composition with no override bit "
                                        "yet (spec/capability_bits.md 5)"),
    # The two compositions with no fused counterpart and no bit.
    "GradNorm": ("composition", "built from SpanReduce"),
    "ScalarDot": ("composition", "built from SpanReduce"),
}

#: The `primitives[]` names that do not resolve to one symbol by spelling. The
#: pointwise family is one name for a forward/backward pair; every other name
#: is the upper camel case of the symbol (`gemm_wgrad` -> `GemmWgrad`, which is
#: the seam's own spelling and not `GemmBackward`).
PRIMITIVE_FAMILIES = {
    "pointwise": ("PointwiseForward", "PointwiseBackward"),
    "smear": ("SmearForward", "SmearBackward"),
}


@dataclass(frozen=True)
class CapabilityRow:
    """One row of `spec/capability_bits.md` section 3, joined to the header."""

    position: int  # the spec's row number, 1-based: the order column
    name: str  # the vocabulary spelling
    group: str  # the spec's group column
    constant: str | None  # the header's spelling, None when the header has none
    bit: int | None  # the header's value, `1 << index`


def _read(root: Path, relative: str) -> str:
    """The text of `relative`, or an empty string when it is absent.

    A missing file is reported by the caller that needs it: `--check` has to
    name the file it could not compare, and `--write` must not create one.
    """
    try:
        return (root / relative).read_text(encoding="utf-8")
    except OSError:
        return ""


def _line_of(text: str, offset: int) -> int:
    """The 1-based line number of `offset`, for a diagnostic."""
    return text.count("\n", 0, offset) + 1


def mechanical_constant(name: str) -> str:
    """`kFeature` + upper camel case of `name` (`spec/capability_bits.md` 2)."""
    return "kFeature" + "".join(part[:1].upper() + part[1:]
                                for part in name.split("_") if part)


def snake_case(name: str) -> str:
    """`GetCaps` -> `get_caps`: the vocabulary spelling of a seam symbol."""
    out = []
    for index, character in enumerate(name):
        if character.isupper() and index and not name[index - 1].isupper():
            out.append("_")
        out.append(character.lower())
    return "".join(out)


def parse_vocabulary(text: str) -> list[tuple[int, str, str]]:
    """The `| # | Name | Group |` rows of `spec/capability_bits.md` section 3.

    Returns `(position, name, group)` in the order the spec lists them, which is
    the order every generated table carries.
    """
    rows: list[tuple[int, str, str]] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line.startswith("## 3. The bits")
            continue
        if not in_section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        name = cells[1].strip("`")
        if not re.fullmatch(r"[a-z0-9_]+", name):
            continue
        rows.append((int(cells[0]), name, cells[2]))
    return rows


def parse_header_constants(text: str) -> dict[str, tuple[int, int]]:
    """The `kFeature*` constants of `include/nanochat/tensor.h`.

    Returns `constant -> (bit index, bit value)`, where the value is the mask
    bit `CapabilityBit(index)` computes.
    """
    constants: dict[str, tuple[int, int]] = {}
    for match in re.finditer(
            r"^inline constexpr std::uint64_t (kFeature\w+) = "
            r"CapabilityBit\((\d+)\);", text, re.MULTILINE):
        index = int(match.group(2))
        constants[match.group(1)] = (index, 1 << index)
    return constants


def capability_rows(root: Path) -> list[CapabilityRow]:
    """The vocabulary joined to the header: one row per bit the header defines.

    A vocabulary name whose constant the header does not define yields a row
    with `constant` and `bit` unset; no table here can carry it, because a name
    with no constant is a promise the planner can never see
    (`spec/capability_bits.md` section 5).
    """
    vocabulary = parse_vocabulary(_read(root, SPEC_BITS))
    constants = parse_header_constants(_read(root, TENSOR_H))
    rows: list[CapabilityRow] = []
    for position, name, group in vocabulary:
        spelling = HEADER_SPELLINGS.get(name, mechanical_constant(name))
        found = constants.get(spelling)
        rows.append(CapabilityRow(
            position=position,
            name=name,
            group=group,
            constant=spelling if found else None,
            bit=found[1] if found else None,
        ))
    return rows


def parse_seam(text: str) -> dict[str, str]:
    """The seam symbols of `include/nanochat/kernels.h`, keyed by section.

    The section is the `// --- Name ---` marker above the declaration, so a
    caller can ask for the device utilities without carrying its own copy of the
    six names.
    """
    symbols: dict[str, str] = {}
    section = ""
    in_namespace = False
    for line in text.splitlines():
        marker = re.match(r"^// --- (.+?) -+$", line)
        if marker:
            section = marker.group(1).strip()
            continue
        if line.startswith("namespace kernels {"):
            in_namespace = True
            continue
        if line.startswith("}  // namespace kernels"):
            in_namespace = False
            continue
        if not in_namespace:
            continue
        declaration = re.match(
            r"^[A-Za-z_][A-Za-z0-9_:<>]*\*?\s+\*?([A-Za-z_]\w*)\(", line)
        if declaration:
            symbols[declaration.group(1)] = section
    return symbols


def parse_enum(text: str, name: str) -> list[str]:
    """The member names of `enum class name { ... }`, in declaration order."""
    match = re.search(r"enum class " + name + r" \{(.*?)\n\};", text,
                      re.DOTALL)
    if match is None:
        return []
    return re.findall(r"^\s*(k\w+) = \d+,", match.group(1), re.MULTILINE)


def parse_prose_kinds(text: str) -> dict[str, str]:
    """The kind rows of `spec/block_kinds.md` section 2, as kind -> group.

    Each `### Group` heading introduces a table whose first column is the kind
    name. A kind is a meaning, so the group is carried beside it: it is what a
    reader needs to find the prose row again.
    """
    kinds: dict[str, str] = {}
    in_section = False
    group = ""
    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line.startswith("## 2. The kinds")
            group = ""
            continue
        if not in_section:
            continue
        if line.startswith("### "):
            group = line[4:].strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not cells[0].startswith("`"):
            continue
        name = cells[0].strip("`")
        if re.fullmatch(r"[a-z0-9_]+", name):
            kinds[name] = group
    return kinds


def parse_reader_kinds(text: str) -> dict[str, list[str]]:
    """The closed block-entry vocabularies of `src/model_spec.cc`.

    Returns `table -> spellings` in declaration order, for the six tables a
    block entry names a kind in. The dtype policy table is not one of them
    (`spec/block_kinds.md` section 6).
    """
    tables: dict[str, list[str]] = {}
    for name in READER_KIND_TABLES:
        match = re.search(r"constexpr \w+ " + name + r"\[\] = \{(.*?)\n\};",
                          text, re.DOTALL)
        tables[name] = (re.findall(r'\{"([a-z0-9_]+)",', match.group(1))
                        if match else [])
    return tables


def resolve_primitive(name: str) -> tuple[str, ...]:
    """The seam symbols a `caps.json` `primitives[]` name stands for."""
    if name in PRIMITIVE_FAMILIES:
        return PRIMITIVE_FAMILIES[name]
    return ("".join(part[:1].upper() + part[1:]
                    for part in name.split("_") if part),)


def bit_words(positions: list[int]) -> str:
    """`bit 9`, `bits 10 and 11`, `bits 1 to 4`: the spec's row numbers."""
    if len(positions) == 1:
        return f"bit {positions[0]}"
    if len(positions) == 2:
        return f"bits {positions[0]} and {positions[1]}"
    return f"bits {positions[0]} to {positions[-1]}"


def render_entry_row(row: CapabilityRow) -> list[str]:
    """One `CapabilityNameEntry` initializer, wrapped at 80 columns."""
    line = f'    {{"{row.name}", {row.constant}, "{row.constant}"}},'
    if len(line) <= 80:
        return [line]
    return [f'    {{"{row.name}", {row.constant},',
            f'     "{row.constant}"}},']


def render_capability_names(rows: list[CapabilityRow]) -> list[str]:
    """The `kCapabilityNames[]` array, grouped and commented like the spec."""
    named = [row for row in rows if row.constant is not None]
    lines = ["inline constexpr CapabilityNameEntry kCapabilityNames[] = {"]
    group = ""
    for row in named:
        if row.group != group:
            group = row.group
            positions = [r.position for r in named if r.group == group]
            lines.append(f"    // {group} ({bit_words(positions)}).")
        lines.extend(render_entry_row(row))
    lines.append("};")
    return lines


def render_string_list(declaration: str, name: str,
                       items: list[str]) -> list[str]:
    """A `const char* const name[]` list: one line if it fits, else one each."""
    single = (f'{declaration} {name}[] = {{'
              + ", ".join(f'"{item}"' for item in items) + "};")
    if len(single) <= 80:
        return [single]
    return [f"{declaration} {name}[] = {{"] + [
        f'    "{item}",' for item in items] + ["};"]


def generated_regions(root: Path) -> dict[tuple[str, str], list[str]]:
    """Every generated region, keyed by `(path, symbol)`, as body lines.

    The key names the file and the declaration the region holds, so a region
    that disappears is a named failure rather than a silent skip.
    """
    rows = capability_rows(root)
    named = [row for row in rows if row.constant is not None]
    vocabulary = [row.name for row in rows]
    reference = [row.name for row in rows
                 if row.group != OVERRIDE_GROUP and row.name not in FP8_PAIR]
    families = [row.name.removeprefix("override_") for row in rows
                if row.group == OVERRIDE_GROUP]
    seam = parse_seam(_read(root, KERNELS_H))
    utilities = [snake_case(symbol) for symbol, section in seam.items()
                 if section.startswith(UTILITY_SECTION)]
    kernels = _read(root, KERNELS_H)
    backends = [Path(path).parent.name for path in CAPS_JSON
                if (root / path).is_file()]

    regions: dict[tuple[str, str], list[str]] = {}
    for path in (CPU_CAPS_NAMES_H, CUDA_CAPS_NAMES_H):
        regions[(path, "kCapabilityNames")] = render_capability_names(named)
    regions[(CPU_CAPS_TEST_CC, "kVocabulary")] = render_string_list(
        "const char* const", "kVocabulary", vocabulary)
    regions[(CPU_CAPS_TEST_CC, "kReferenceFeatures")] = render_string_list(
        "const char* const", "kReferenceFeatures", reference)
    regions[(CPU_CAPS_TEST_CC, "kUtilities")] = render_string_list(
        "const char* const", "kUtilities", utilities)
    regions[(CPU_CAPS_TEST_CC, "kPointwiseCodes")] = render_string_list(
        "const char* const", "kPointwiseCodes",
        parse_enum(kernels, "PointwiseOp"))
    regions[(CPU_CAPS_TEST_CC, "kReduceOps")] = render_string_list(
        "const char* const", "kReduceOps", parse_enum(kernels, "ReduceOp"))
    regions[(CPU_CAPS_TEST_CC, "kCompositionFamilies")] = render_string_list(
        "const char* const", "kCompositionFamilies", families)
    regions[(DEVICE_PROFILE_CC, "kBackendNames")] = render_string_list(
        "constexpr const char*", "kBackendNames", backends)
    return regions


def region_bounds(text: str, symbol: str) -> tuple[int, int] | None:
    """The character range between a region's markers, markers excluded.

    A region is found by its marker pair and must then contain `symbol`, so two
    regions in one file cannot be confused for each other.
    """
    begin = text.find(BEGIN)
    while begin != -1:
        body = begin + len(BEGIN)
        end = text.find(END, body)
        if end == -1:
            return None
        if symbol in text[body:end]:
            return body, end
        begin = text.find(BEGIN, end + len(END))
    return None


def render_region(body: list[str]) -> str:
    """The bytes a region holds: one marker pair around the generated lines."""
    return "\n" + "\n".join([FORMAT_OFF, *body, FORMAT_ON]) + "\n"


def write_regions(root: Path) -> int:
    """Rewrite every generated region in place. Returns the exit code."""
    regions = generated_regions(root)
    failures = 0
    for path in dict.fromkeys(path for path, _ in regions):
        source = root / path
        text = _read(root, path)
        if not text:
            print(f"{path}: no such file; --write does not create one",
                  file=sys.stderr)
            failures += 1
            continue
        updated = text
        for (region_path, symbol), body in regions.items():
            if region_path != path:
                continue
            bounds = region_bounds(updated, symbol)
            if bounds is None:
                print(f"{path}: no BEGIN/END GENERATED marker pair holds "
                      f"{symbol}; add the pair by hand", file=sys.stderr)
                failures += 1
                continue
            begin, end = bounds
            updated = updated[:begin] + render_region(body) + updated[end:]
        if updated != text:
            source.write_text(updated, encoding="utf-8")
    return 1 if failures else 0


@dataclass(frozen=True)
class Mismatch:
    """One failed comparison, in the `path:line: message` shape of `lint`."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def check_capability_names(root: Path,
                           rows: list[CapabilityRow]) -> list[Mismatch]:
    """The name, bit, and order of every row of `src/capability_names.h`.

    The `consequence` column is hand-written prose and is not compared: the
    spec does not supply it, so this script cannot be its authority.
    """
    text = _read(root, CAPABILITY_NAMES_H)
    if not text:
        return [Mismatch(CAPABILITY_NAMES_H, "the file is missing")]
    expected = [row for row in rows if row.constant is not None]
    found = list(re.finditer(
        r'\{"([a-z0-9_]+)",\s*(kFeature\w+),\s*"(kFeature\w+)"', text))
    if not found:
        return [Mismatch(CAPABILITY_NAMES_H,
                         "no kCapabilityNames[] row was found")]
    mismatches: list[Mismatch] = []
    constants = parse_header_constants(_read(root, TENSOR_H))
    for index, match in enumerate(found):
        where = f"{CAPABILITY_NAMES_H}:{_line_of(text, match.start())}"
        if index >= len(expected):
            mismatches.append(Mismatch(
                where, f"row {index + 1} is `{match.group(1)}`, which "
                       f"spec/capability_bits.md section 3 has no constant "
                       f"for"))
            continue
        row = expected[index]
        if match.group(1) != row.name:
            mismatches.append(Mismatch(
                where, f"row {index + 1} is `{match.group(1)}` but "
                       f"spec/capability_bits.md section 3 puts "
                       f"`{row.name}` there"))
        if match.group(2) != row.constant:
            mismatches.append(Mismatch(
                where, f"row `{row.name}` uses {match.group(2)} but "
                       f"include/nanochat/tensor.h spells it {row.constant}"))
        if match.group(3) != match.group(2):
            mismatches.append(Mismatch(
                where, f"row `{row.name}` carries the diagnostic spelling "
                       f"{match.group(3)} beside the constant "
                       f"{match.group(2)}"))
        found_bit = constants.get(match.group(2))
        if found_bit is None:
            mismatches.append(Mismatch(
                where, f"row `{row.name}` names {match.group(2)}, which "
                       f"include/nanochat/tensor.h does not define"))
        elif row.bit is not None and found_bit[1] != row.bit:
            mismatches.append(Mismatch(
                where, f"row `{row.name}` carries bit {found_bit[0]} "
                       f"({found_bit[1]}) but the header's order gives it "
                       f"{row.bit}"))
    if len(found) < len(expected):
        mismatches.append(Mismatch(
            CAPABILITY_NAMES_H,
            f"{len(found)} rows, but spec/capability_bits.md section 3 and "
            f"include/nanochat/tensor.h define {len(expected)}"))
    return mismatches


def check_caps_json(root: Path, rows: list[CapabilityRow]) -> list[Mismatch]:
    """The `features[]` array of every declaration: name, bit, and order.

    The order is the spec's: a declaration lists a subset of the vocabulary and
    must list it in the vocabulary's order, so a reader can diff two
    declarations line by line.
    """
    order = {row.name: row.position for row in rows}
    bits = {row.name: row.bit for row in rows}
    mismatches: list[Mismatch] = []
    for path in CAPS_JSON:
        text = _read(root, path)
        if not text:
            mismatches.append(Mismatch(path, "the file is missing"))
            continue
        try:
            declaration = json.loads(text)
        except json.JSONDecodeError as error:
            mismatches.append(Mismatch(path, f"not JSON: {error}"))
            continue
        devices = declaration.get("devices", [])
        if not devices:
            mismatches.append(Mismatch(path, "no devices[] entry to check"))
        for device in devices:
            features = device.get("features", [])
            where = f"{path} ({device.get('name', '?')}.features)"
            previous = 0
            for index, feature in enumerate(features):
                row = f"row {index + 1} (`{feature}`)"
                if feature not in order:
                    mismatches.append(Mismatch(
                        where, f"{row} is not a name in "
                               f"spec/capability_bits.md section 3"))
                    continue
                if bits.get(feature) is None:
                    mismatches.append(Mismatch(
                        where, f"{row} has no kFeature constant in "
                               f"include/nanochat/tensor.h"))
                if order[feature] <= previous:
                    mismatches.append(Mismatch(
                        where, f"{row} is out of the spec's order: it follows "
                               f"a bit the spec lists later"))
                previous = order[feature]
    return mismatches


def check_block_kinds(root: Path) -> list[Mismatch]:
    """`spec/block_kinds.md` section 2 against the spellings the reader accepts.

    Both directions fail: a kind the prose names and the reader does not accept
    is a model package that cannot load, and a spelling the reader accepts with
    no prose row is a meaning a reader of the contract cannot look up. The
    second direction is the one `models/nanochat-default/README.md` reports as
    F17, so its message says which of the two a repair has to touch.
    """
    prose = parse_prose_kinds(_read(root, SPEC_KINDS))
    if not prose:
        return [Mismatch(SPEC_KINDS, "no kind row was found in section 2")]
    reader_text = _read(root, MODEL_SPEC_CC)
    if not reader_text:
        return [Mismatch(MODEL_SPEC_CC, "the file is missing")]
    tables = parse_reader_kinds(reader_text)
    accepted = {spelling: table for table, spellings in tables.items()
                for spelling in spellings}
    mismatches: list[Mismatch] = []
    # Prose -> reader: a kind with no alias row and no accepted spelling.
    claimed = {kind for kind, _ in KIND_ALIASES.values() if kind is not None}
    for kind, group in prose.items():
        if kind not in claimed:
            mismatches.append(Mismatch(
                f"{SPEC_KINDS} ({group})",
                f"section 2 names `{kind}`, which {MODEL_SPEC_CC} does not "
                f"accept and no alias row in this script maps"))
    # Reader -> prose: an accepted spelling with no alias row, or with one that
    # points at a kind the prose does not name.
    for spelling, table in accepted.items():
        if spelling not in KIND_ALIASES:
            mismatches.append(Mismatch(
                MODEL_SPEC_CC,
                f"{table} accepts `{spelling}`, which has no prose row in "
                f"{SPEC_KINDS} section 2 and no alias row in this script: "
                f"add one of the two"))
            continue
        kind, _ = KIND_ALIASES[spelling]
        if kind is not None and kind not in prose:
            mismatches.append(Mismatch(
                MODEL_SPEC_CC,
                f"{table} accepts `{spelling}`, whose alias row names "
                f"`{kind}`; {SPEC_KINDS} section 2 has no such kind"))
    # An alias row that names a spelling the reader no longer accepts.
    for spelling in KIND_ALIASES:
        if spelling not in accepted:
            mismatches.append(Mismatch(
                MODEL_SPEC_CC,
                f"this script maps `{spelling}`, which no row table accepts: "
                f"the alias row is stale"))
    return mismatches


def check_seam_inventory(root: Path) -> list[Mismatch]:
    """`include/nanochat/kernels.h` against every `caps.json` `primitives[]`.

    Three directions fail: a seam symbol with no tier row (the seam grew a
    symbol and nobody said who supplies it), a `primitives[]` entry with no
    symbol (the declaration promises a primitive the seam does not name), and
    a primitive-tier symbol with no `primitives[]` entry (the seam owes a
    primitive the declaration does not claim).
    """
    seam = parse_seam(_read(root, KERNELS_H))
    if not seam:
        return [Mismatch(KERNELS_H, "no seam symbol was found")]
    mismatches: list[Mismatch] = []
    for symbol in seam:
        if symbol not in SEAM_TIERS:
            mismatches.append(Mismatch(
                KERNELS_H,
                f"`{symbol}` has no tier row in this script: add a tier row "
                f"(utilities, primitives, stateful, override, or composition) "
                f"naming the caps.json array that must declare it"))
    for symbol in SEAM_TIERS:
        if symbol not in seam:
            mismatches.append(Mismatch(
                KERNELS_H,
                f"this script classifies `{symbol}`, which "
                f"include/nanochat/kernels.h does not declare: the tier row "
                f"is stale"))
    for path in CAPS_JSON:
        text = _read(root, path)
        if not text:
            mismatches.append(Mismatch(path, "the file is missing"))
            continue
        try:
            declaration = json.loads(text)
        except json.JSONDecodeError as error:
            mismatches.append(Mismatch(path, f"not JSON: {error}"))
            continue
        for device in declaration.get("devices", []):
            primitives = device.get("primitives", [])
            where = f"{path} ({device.get('name', '?')}.primitives)"
            declared: set[str] = set()
            for index, name in enumerate(primitives):
                symbols = resolve_primitive(name)
                for symbol in symbols:
                    if symbol not in seam:
                        mismatches.append(Mismatch(
                            where,
                            f"row {index + 1} (`{name}`) resolves to "
                            f"`{symbol}`, which include/nanochat/kernels.h "
                            f"does not declare"))
                        continue
                    tier = SEAM_TIERS.get(symbol, ("", ""))[0]
                    if tier != "primitives":
                        mismatches.append(Mismatch(
                            where,
                            f"row {index + 1} (`{name}`) resolves to "
                            f"`{symbol}`, whose tier row says `{tier}`"))
                    declared.add(symbol)
            for symbol, (tier, _) in SEAM_TIERS.items():
                if tier == "primitives" and symbol not in declared:
                    mismatches.append(Mismatch(
                        where,
                        f"`{symbol}` is a primitive-tier seam symbol with no "
                        f"primitives[] entry"))
    return mismatches


def check(root: Path) -> int:
    """Compare the hand-written rows against the parse. The exit code."""
    rows = capability_rows(root)
    if not rows:
        print(f"{SPEC_BITS}: no capability row was found", file=sys.stderr)
        return 1
    mismatches = check_capability_names(root, rows)
    mismatches += check_caps_json(root, rows)
    mismatches += check_block_kinds(root)
    mismatches += check_seam_inventory(root)
    if mismatches:
        for mismatch in mismatches:
            print(f"FAIL: {mismatch}", file=sys.stderr)
        return 1
    absent = [row.name for row in rows if row.constant is None]
    devices = sum(len(json.loads(_read(root, path)).get("devices", []))
                  for path in CAPS_JSON if _read(root, path))
    print(f"checked {len(rows) - len(absent)} rows of {CAPABILITY_NAMES_H} and "
          f"{devices} features[] array(s) in {len(CAPS_JSON)} declarations: "
          f"name, bit, and order agree with {SPEC_BITS} section 3 and "
          f"{TENSOR_H}", file=sys.stderr)
    for name in absent:
        print(f"recorded: `{name}` is a vocabulary bit with no constant in "
              f"{TENSOR_H}, so no table here carries it", file=sys.stderr)
    kinds = parse_prose_kinds(_read(root, SPEC_KINDS))
    seam = parse_seam(_read(root, KERNELS_H))
    print(f"checked {len(kinds)} kinds of {SPEC_KINDS} section 2 against "
          f"{MODEL_SPEC_CC}, and {len(seam)} seam symbols of {KERNELS_H} "
          f"against the primitives[] of {len(CAPS_JSON)} declarations: the two "
          f"directions agree", file=sys.stderr)
    for spelling, (kind, reason) in KIND_ALIASES.items():
        if kind is None:
            print(f"recorded: {MODEL_SPEC_CC} accepts `{spelling}`, which has "
                  f"no prose row: {reason}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the capability rows that are data, and check the "
                    "rows that are prose, the block kinds, and the seam "
                    "inventory.")
    parser.add_argument("--write", action="store_true",
                        help="rewrite every generated region")
    parser.add_argument("--check", action="store_true",
                        help="check the hand-written rows, the block kinds, "
                             "and the seam inventory; write nothing")
    parser.add_argument("--root", default=str(REPO_ROOT),
                        help="the repository root (default: this checkout)")
    arguments = parser.parse_args(argv)
    if arguments.write == arguments.check:
        parser.error("name exactly one of --write and --check")
    root = Path(arguments.root).resolve()
    if arguments.write:
        return write_regions(root)
    return check(root)


if __name__ == "__main__":
    sys.exit(main())
