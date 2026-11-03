#!/usr/bin/env python3
"""Check the three C ABI mirrors of ``include/nanochat/capi.h``, and
generate the two that are mechanical.

``docs/api-surface.md`` R11b: one header is mirrored three times, and every
mirror is a copy of the header's names, order, and types. A copy that a hand
keeps is a copy that drifts, so the header is the authority for all three:

* ``python/nanochat_cpp/_core.py`` -- its ``_LAYOUT`` table holds the
  ``ctypes`` fields of every ``ctypes.Structure`` that mirrors a C struct, and
  its ``_SIGNATURES`` table gives every C function its argument and result
  types. ``--check`` compares the layout table with the header through
  ``_core.layout()``, which runs the module: it is standard-library only and
  loads no shared library. The table itself stays hand-written, keyed by
  class and carrying the real ``ctypes`` types, so a reader sees the layout
  and Python type-checks it.
* ``bindings/nanochat_capi_test.cc`` -- the ``static_assert`` block that pins
  each flat C mirror field against the C++ struct it copies.

What stays hand-written is what the header cannot say: the class docstrings,
the layout table, the prose around each generated region, and the ``declare``
call that applies the signature table. The field copy code in
``bindings/nanochat_capi.cc`` holds the semantics of a conversion and is not a
mirror of the layout, so it stays hand-written too.

Every generated region sits between a ``BEGIN GENERATED`` and an ``END
GENERATED`` marker pair, so the prose around it survives a rewrite. ``--write``
replaces the bytes between the markers and touches nothing else, and running it
twice leaves the tree unchanged. The marker line's own indentation is the
region's indentation, and its key names what the region mirrors:
``_SIGNATURES`` for the function table and ``static_assert`` for the drift
block. The layout table carries no marker, because no run of this script
writes it.

``--check`` writes nothing. It names the file, the structure, and the field for
a disagreement in either direction: a field the header declares and the layout
table lacks, a field the table carries and the header dropped, a type that
differs, and a row out of the header's order. A structure the header declares
and the table does not classify, and a structure the table names and the header
never declared, fail the same way.

The parser handles the forms this header uses: a scalar, a fixed ``char``
array, a nested struct by value, an enum typedef (a C enum is an ``int``), a
pointer to one of those, an opaque handle, and a pointer to a pointer. Anything
else stops the run with the offending line, because a guess in a generator is a
lie in three files at once.

Usage::

    # Rewrite every generated region from the header.
    python3 tools/gen_capi_mirrors.py --write

    # Check every mirror against the header. Exits non-zero on a mismatch.
    python3 tools/gen_capi_mirrors.py --check

The script is standard-library only and hermetic: it reads the header and the
two mirrors, runs the layout table of the second one, and writes only the
generated regions of the two mirrors.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import re
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The header that is the single authority for all three mirrors.
CAPI_H = "include/nanochat/capi.h"

#: The two mirrors, each holding generated regions.
CORE_PY = "python/nanochat_cpp/_core.py"
CAPI_TEST_CC = "bindings/nanochat_capi_test.cc"

#: The marker text, without the comment prefix each language adds.
BEGIN = "BEGIN GENERATED (tools/gen_capi_mirrors.py --write)"
END = "END GENERATED (tools/gen_capi_mirrors.py --write)"

#: The comment prefix of each mirrored file. The header is not mirrored, so it
#: carries no marker.
PREFIX = {".py": "#", ".cc": "//"}

#: The key of the ``_SIGNATURES`` region, which mirrors every C function.
SIGNATURES = "_SIGNATURES"

#: The key of the drift block region, which pins the C mirrors against C++.
STATIC_ASSERT = "static_assert"

#: ``clang-format`` is switched off inside the C++ region: a generated region
#: must be a fixed point of the generator, not of the formatter.
FORMAT_OFF = "// clang-format off"
FORMAT_ON = "// clang-format on"

#: The last column a generated Python line may occupy, and one less than the
#: 80-column limit `.clang-format` sets for the C++ block.
WIDTH = 79

#: A C scalar and the ``ctypes`` type that carries it.
SCALARS = {
    "char": "ctypes.c_char",
    "double": "ctypes.c_double",
    "float": "ctypes.c_float",
    "int": "ctypes.c_int",
    "int8_t": "ctypes.c_int8",
    "int16_t": "ctypes.c_int16",
    "int32_t": "ctypes.c_int32",
    "int64_t": "ctypes.c_int64",
    "size_t": "ctypes.c_size_t",
    "uint8_t": "ctypes.c_uint8",
    "uint16_t": "ctypes.c_uint16",
    "uint32_t": "ctypes.c_uint32",
    "uint64_t": "ctypes.c_uint64",
    "unsigned int": "ctypes.c_uint",
}

#: A C enum typedef. The C standard leaves the width to the implementation, and
#: this toolchain sizes every one of them as an ``int``, which is what the
#: mirrors already carry.
ENUM_TYPES = ("nanochat_precision", "nanochat_stage", "nanochat_status")

#: The Python class names that differ from the mechanical rule, which strips
#: the ``nanochat_`` prefix and upper-camels the rest. Each row records a name
#: the module already published, so renaming it would break a caller.
CLASS_SPELLINGS = {
    # The struct mirrors two C++ types at once, so the class names the pair.
    "nanochat_optim_config": "OptimizerConfig",
    # These three are the sub-configs of `nanochat_mesh_train_config`, and the
    # class carries that prefix so a reader sees the driver they belong to.
    "nanochat_mesh_data_config": "MeshTrainDataConfig",
    "nanochat_mesh_output_config": "MeshTrainOutputConfig",
    "nanochat_mesh_run_config": "MeshTrainRunConfig",
}

#: The ctypes text that differs from the mechanical rule, keyed by function and
#: by the 0-based index of the argument. A `char*` out-buffer is the one form
#: whose mirror is not `c_char_p`: that type reads a buffer as a C string, and
#: `nanochat_decode` fills it with bytes that a NUL terminates, so the mirror
#: carries the address and not the string.
ARGUMENT_OVERRIDES = {
    ("nanochat_decode", 3): "ctypes.c_void_p",
}

#: The drift modes of the ``static_assert`` block.
EQUAL = "equal"  # sizeof(C field) == sizeof(C++ field)
FLOOR = "floor"  # sizeof(C field) >= sizeof(C++ field): C carries the wider type
SAME = "same"  # std::is_same<decltype(C field), decltype(C++ field)>
SKIP = "skip"  # no assert: this row's note says why


@dataclass(frozen=True)
class Row:
    """One drift row: a C field and how the block compares it to C++."""

    field: str
    mode: str
    note: str = ""


#: The C++ struct each pinned C mirror copies, with one row per field of that C
#: struct in header order. `docs/python.md` section 13 states the rule the rows
#: follow: a scalar is a size equality, a C mirror that carries a wider type
#: than its C++ field is a size floor, and a pointer is a `decltype` equality.
DRIFT_ROWS = {
    "nanochat_config": ("nanochat::Config", (
        Row("num_layers", EQUAL),
        Row("num_heads", EQUAL),
        Row("num_kv_heads", EQUAL),
        Row("hidden_dim", EQUAL),
        Row("seq_len", EQUAL),
        Row("vocab_size", EQUAL),
        Row("padded_vocab_size", EQUAL),
        Row("rope_base", EQUAL),
        Row("window_pattern", SKIP,
            "the C field is a `const char*` buffer; the C++ field owns a "
            "`std::string`"),
        Row("value_embedding", FLOOR,
            "the C mirror widens the C++ `bool` to an `int`"),
    )),
    "nanochat_param": ("nanochat::ParamView", (
        Row("name", SKIP,
            "the C++ view holds the same pointer, and this block pins the "
            "layout and not the pointer's meaning"),
        Row("value", SKIP,
            "the C mirror carries `void*`; the C++ field is a `ComputeType*`"),
        Row("grad", SKIP,
            "the C mirror carries `void*`; the C++ field is a `GradientType*`"),
        Row("count", EQUAL),
        Row("rows", EQUAL),
        Row("cols", EQUAL),
    )),
    "nanochat_device": ("nanochat::Caps", (
        Row("device_index", SKIP,
            "both sides are a plain `int`, and this block pins the fields "
            "whose type could drift"),
        Row("compute_major", SKIP, "both sides are a plain `int`"),
        Row("compute_minor", SKIP, "both sides are a plain `int`"),
        Row("total_memory_bytes", EQUAL),
        Row("device_name", SAME),
    )),
    "nanochat_focus": ("nanochat::ScoreFocus", (
        Row("position", SAME),
        Row("ids", SAME),
        Row("count", SAME),
    )),
    "nanochat_generate_params": ("nanochat::GenerateParams", (
        Row("num_samples", SAME),
        Row("max_tokens", SAME),
        Row("temperature", SAME),
        Row("top_k", SAME),
        Row("seed", SAME),
        Row("stop_id", SAME),
        Row("bos_id", SAME),
        Row("stop_ids", SAME),
    )),
    "nanochat_checkpoint_metadata": ("nanochat::CheckpointMetadata", (
        Row("schema", EQUAL),
        Row("stage", EQUAL),
        Row("step", EQUAL),
        Row("world_size", FLOOR,
            "the C mirror widens the C++ `uint32_t` to an `int64_t`"),
        Row("precision", EQUAL),
        Row("config_json", SKIP,
            "the C field is one JSON string; the C++ field is a `Config`"),
        Row("tokenizer_id", SKIP,
            "the C++ field owns a `std::string`"),
        Row("git_revision", SKIP, "the C++ field owns a `std::string`"),
        Row("created_utc", SKIP, "the C++ field owns a `std::string`"),
        Row("notes", SKIP,
            "the C++ field holds a `std::vector`, not a buffer of pointers"),
        Row("note_count", SKIP,
            "the count is the extent of the C buffer; the C++ vector carries "
            "its own size"),
    )),
}

#: The C mirror structs this block does not pin, and why no field of one has a
#: row. Every struct the header declares must appear in this table or in
#: `DRIFT_ROWS`, so a new mirror has to be classified before it lands.
NO_LAYOUT_MIRROR = {
    "nanochat_params":
        "`nanochat::ParamBreakdown` pins the five counts and the shim adds "
        "`flops_per_token` from `EstimateFlopsPerToken`",
    "nanochat_score_result":
        "`nanochat::ScoreResult` holds vectors, so no field has the C "
        "buffer's type",
    "nanochat_sequences":
        "no C++ struct: `src/generate.cc` fills model-owned buffers",
    "nanochat_generate_multi_params":
        "`nanochat::GenerateMultiPrompt` holds a vector and a nested params "
        "struct",
    "nanochat_optim_config":
        "one C struct mirrors `nanochat::OptimizerConfig` and "
        "`nanochat::SchedulerConfig`",
    "nanochat_metrics":
        "`nanochat::MeshMetrics` holds a vector of per-replica losses",
    "nanochat_mesh_data_config":
        "the data half of `nanochat::MeshTrainConfig`, whose strings are "
        "`std::string`",
    "nanochat_mesh_run_config":
        "the step half of `nanochat::MeshTrainConfig`, spread over its "
        "scalars",
    "nanochat_mesh_output_config":
        "the output half of `nanochat::MeshTrainConfig`, whose paths are "
        "`std::string`",
    "nanochat_mesh_train_config":
        "one C struct mirrors the flattened `nanochat::MeshTrainConfig` and "
        "its three sub-configs",
    "nanochat_mesh_train_summary":
        "`nanochat::MeshTrainSummary` holds a `std::string` and a vector, and "
        "the C field is a fixed buffer",
    "nanochat_record":
        "no C++ struct: the container reader fills the record",
}

#: The C++ enum each mirrored C enum spells out, and the exception to the
#: mechanical rule that strips `nanochat_` and upper-camels the rest. An
#: enumerator `NANOCHAT_<ENUM>_<WORD>` becomes `k` + `<Word>`.
ENUM_MIRRORS = {
    "nanochat_stage": "nanochat::Stage",
    "nanochat_precision": "nanochat::ComputePrecision",
}

#: The C enums with no C++ counterpart, and why the block pins no value of one.
NO_ENUM_MIRROR = {
    "nanochat_status":
        "the C surface's own status code; no C++ enum carries it",
}


class HeaderError(Exception):
    """A header form this generator does not understand."""

    def __init__(self, path: str, line: int, text: str, detail: str) -> None:
        super().__init__(f"{path}:{line}: the generator does not understand "
                         f"`{text}`: {detail}")


class TableError(Exception):
    """A drift table that does not classify what the header declares."""


# --- The header parse -----------------------------------------------------


@dataclass(frozen=True)
class Form:
    """One C declarator: a base type, its pointer depth, and an array extent."""

    base: str
    pointers: int
    array: int | None
    line: int
    text: str

    def describe(self) -> str:
        """The declarator as the header writes it, for a failure message."""
        stars = "*" * self.pointers
        extent = "" if self.array is None else f"[{self.array}]"
        return f"{self.base}{stars}{extent}"


@dataclass(frozen=True)
class Field:
    """One field of a C struct."""

    name: str
    form: Form


@dataclass(frozen=True)
class Parameter:
    """One parameter of a C function."""

    name: str
    form: Form


@dataclass(frozen=True)
class Struct:
    """One C struct definition, in declaration order."""

    name: str
    fields: tuple[Field, ...]
    line: int


@dataclass(frozen=True)
class Enum:
    """One C enum definition, in declaration order."""

    name: str
    values: tuple[tuple[str, int], ...]
    line: int


@dataclass(frozen=True)
class Function:
    """One C function prototype."""

    name: str
    result: Form
    parameters: tuple[Parameter, ...]
    line: int
    text: str


@dataclass(frozen=True)
class Header:
    """Everything the three mirrors are generated from."""

    structs: tuple[Struct, ...]
    opaque: tuple[str, ...]
    enums: tuple[Enum, ...]
    functions: tuple[Function, ...]


def _strip_comments(path: str, text: str) -> str:
    """The text with every comment blanked and every newline kept.

    A blanked byte becomes a space, so an offset into the result is an offset
    into the source and a failure can name the line it met. The preprocessor
    lines go, and so does the `extern "C"` linkage block: neither declares a
    type that a mirror could carry.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            while index < length and text[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end == -1:
                raise HeaderError(path, _line_of(text, index), "/*",
                                  "an unterminated block comment")
            for offset in range(index, end + 2):
                out.append("\n" if text[offset] == "\n" else " ")
            index = end + 2
            continue
        out.append(text[index])
        index += 1
    clean = "".join(out)
    # A preprocessor line is not a declaration and carries no mirror.
    lines = ["" if line.lstrip().startswith("#") else line
             for line in clean.split("\n")]
    linkage = False
    for index, line in enumerate(lines):
        if line.strip() == 'extern "C" {':
            lines[index] = ""
            linkage = True
            continue
        if linkage and line.strip() == "}":
            lines[index] = ""
            linkage = False
    if linkage:
        raise HeaderError(path, _line_of(clean, clean.find('extern "C" {')),
                          'extern "C" {', "the linkage block never closes")
    return "\n".join(lines)


def _line_of(text: str, offset: int) -> int:
    """The 1-based line an offset falls on."""
    return text.count("\n", 0, offset) + 1


def _declarations(path: str, text: str) -> list[tuple[int, str]]:
    """The top-level declarations, as `(line, text)` in source order.

    A declaration ends at the `;` that closes it at brace depth zero, which
    keeps a struct or enum body inside one chunk.
    """
    chunks: list[tuple[int, str]] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise HeaderError(path, _line_of(text, index), "}",
                                  "a brace closes nothing")
        elif character == ";" and depth == 0:
            leading = len(text[start:index + 1]) - len(
                text[start:index + 1].lstrip())
            chunks.append((_line_of(text, start + leading),
                           text[start + leading:index + 1]))
            start = index + 1
    tail = text[start:].strip()
    if tail:
        raise HeaderError(path, _line_of(text, start), tail,
                          "a declaration does not end with `;`")
    if depth != 0:
        raise HeaderError(path, _line_of(text, start), "{",
                          "a brace never closes")
    return chunks


def _parse_declarator(path: str, text: str, line: int, what: str,
                      named: bool = True) -> Form:
    """One declarator: `[const] TYPE [*]* [NAME] [[EXTENT]]`.

    `named` is false for a return type, which carries no name.
    """
    original = " ".join(text.split())
    remaining = original
    array: int | None = None
    extent = re.match(r"^(.*?)\[\s*(\d+)\s*\]$", remaining)
    if extent is not None:
        array = int(extent.group(2))
        remaining = extent.group(1).strip()
    if named:
        declared = re.match(r"^(.*?)([A-Za-z_][A-Za-z_0-9]*)$", remaining)
        if declared is None:
            raise HeaderError(path, line, original,
                              f"no {what} name ends the declarator")
        prefix = declared.group(1)
    else:
        prefix = remaining
    pointers = 0
    while True:
        stripped = prefix.rstrip()
        if stripped.endswith("*"):
            pointers += 1
            prefix = stripped[:-1]
            continue
        # A `const` after a star qualifies the pointer itself.
        qualified = re.match(r"^(.*?)\bconst$", stripped)
        if qualified is not None and pointers:
            prefix = qualified.group(1)
            continue
        break
    words = [word for word in prefix.replace("*", " ").split()
             if word != "const"]
    if words[:1] == ["struct"]:
        words = words[1:]
    if len(words) != 1:
        raise HeaderError(path, line, original,
                          f"the {what} is not one type name")
    return Form(base=words[0], pointers=pointers, array=array, line=line,
                text=original)


def _parse_fields(path: str, body: str, line: int, name: str) -> tuple[Field,
                                                                     ...]:
    """Every field of one struct body, in declaration order.

    `line` is the line the body starts on, so a failure names the line of the
    field and not the line of the struct.
    """
    fields: list[Field] = []
    cursor = 0
    for piece in body.split(";"):
        start = cursor
        cursor += len(piece) + 1
        if not piece.strip():
            continue
        field_line = line + body.count("\n", 0, start)
        if "{" in piece or "}" in piece:
            raise HeaderError(path, field_line, " ".join(piece.split()),
                              "a struct body holds no nested block")
        form = _parse_declarator(path, piece, field_line, "field")
        fields.append(Field(name=_last_identifier(piece), form=form))
    if not fields:
        raise HeaderError(path, line, name, "the struct declares no field")
    return tuple(fields)


def _last_identifier(text: str) -> str:
    """The declared name of a declarator, which is its last identifier."""
    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)
    return names[-1]


def parse_header(path: str, text: str) -> Header:
    """The structs, opaque handles, enums, and functions of the header.

    Each declaration is matched against the source chunk it came from, not
    against a re-spaced copy, so a failure can name the line of the field and
    not only the line of the declaration.
    """
    clean = _strip_comments(path, text)
    structs: list[Struct] = []
    opaque: list[str] = []
    enums: list[Enum] = []
    functions: list[Function] = []
    for line, chunk in _declarations(path, clean):
        body = re.match(
            r"^typedef\s+struct\s*([A-Za-z_][A-Za-z_0-9]*)?\s*\{(.*)\}\s*"
            r"([A-Za-z_][A-Za-z_0-9]*)\s*;$", chunk, re.DOTALL)
        if body is not None:
            name = body.group(3)
            tag = body.group(1)
            if tag is not None and tag != name:
                raise HeaderError(path, line, " ".join(chunk.split()),
                                  f"the struct tag `{tag}` is not its name "
                                  f"`{name}`")
            fields_line = line + chunk.count("\n", 0, chunk.find(body.group(2)))
            structs.append(Struct(name=name,
                                  fields=_parse_fields(path, body.group(2),
                                                       fields_line, name),
                                  line=line))
            continue
        handle = re.match(
            r"^typedef\s+struct\s+([A-Za-z_][A-Za-z_0-9]*)\s+"
            r"([A-Za-z_][A-Za-z_0-9]*)\s*;$", chunk)
        if handle is not None:
            if handle.group(1) != handle.group(2):
                raise HeaderError(path, line, " ".join(chunk.split()),
                                  "an opaque handle must name one type twice")
            opaque.append(handle.group(1))
            continue
        enumeration = re.match(
            r"^typedef\s+enum\s*\{(.*)\}\s*([A-Za-z_][A-Za-z_0-9]*)\s*;$",
            chunk, re.DOTALL)
        if enumeration is not None:
            values: list[tuple[str, int]] = []
            cursor = 0
            for item in enumeration.group(1).split(","):
                item_line = line + chunk.count(
                    "\n", 0, chunk.find(enumeration.group(1)) + cursor)
                cursor += len(item) + 1
                if not item.strip():
                    continue
                pair = re.match(
                    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(-?\d+)\s*$", item)
                if pair is None:
                    raise HeaderError(path, item_line, item.strip(),
                                      "an enumerator must be NAME = INTEGER")
                values.append((pair.group(1), int(pair.group(2))))
            if not values:
                raise HeaderError(path, line, " ".join(chunk.split()),
                                  "the enum declares no value")
            enums.append(Enum(name=enumeration.group(2), values=tuple(values),
                              line=line))
            continue
        prototype = re.match(
            r"^(.*?)\b([A-Za-z_][A-Za-z_0-9]*)\s*\((.*)\)\s*;$", chunk,
            re.DOTALL)
        if prototype is None:
            raise HeaderError(path, line, " ".join(chunk.split()),
                              "not a typedef, an opaque handle, or a function")
        result = _parse_declarator(path, prototype.group(1), line, "result",
                                   named=False)
        parameters: list[Parameter] = []
        arguments = prototype.group(3)
        if arguments.strip() and arguments.strip() != "void":
            for argument in arguments.split(","):
                if not argument.strip():
                    raise HeaderError(path, line, " ".join(chunk.split()),
                                      "an argument list holds an empty entry")
                parameters.append(Parameter(
                    name=_last_identifier(argument),
                    form=_parse_declarator(path, argument, line, "argument")))
        functions.append(Function(name=prototype.group(2), result=result,
                                  parameters=tuple(parameters), line=line,
                                  text=" ".join(chunk.split())))
    return Header(structs=tuple(structs), opaque=tuple(opaque),
                  enums=tuple(enums), functions=tuple(functions))


# --- The Python mirror ----------------------------------------------------


def _upper_camel(suffix: str) -> str:
    """The upper-camel spelling of a `nanochat_`-stripped name."""
    return "".join(part.capitalize() for part in suffix.split("_"))


def class_name(struct: str) -> str:
    """The Python class that mirrors one C struct or opaque handle."""
    if struct in CLASS_SPELLINGS:
        return CLASS_SPELLINGS[struct]
    return _upper_camel(struct.removeprefix("nanochat_"))


def handle_name(struct: str) -> str:
    """The module-level alias that carries one opaque C handle."""
    return class_name(struct) + "Handle"


def render_type(form: Form, header: Header, surface: str) -> str:
    """The ``ctypes`` text of one argument or one result of a function.

    `surface` is ``argument`` for a parameter and ``result`` for a return type;
    the two differ in one place, because a ``void`` result is ``None``. A
    struct field is not rendered here: it goes through `canonical_type`, which
    is the spelling `_core.layout()` reports.
    """
    structs = {struct.name for struct in header.structs}
    opaque = set(header.opaque)

    def atom(base: str) -> str:
        if base in SCALARS:
            return SCALARS[base]
        if base in ENUM_TYPES:
            return "ctypes.c_int"
        if base in structs:
            return class_name(base)
        if base == "void":
            return "None" if surface == "result" else ""
        if base in opaque:
            raise HeaderError(CAPI_H, form.line, form.text,
                              "an opaque handle is a pointer everywhere")
        raise HeaderError(CAPI_H, form.line, form.text,
                          f"no ctypes type mirrors `{base}`")

    if form.array is not None:
        raise HeaderError(CAPI_H, form.line, form.text,
                          "an array parameter or result is not a form here")
    if form.pointers == 0:
        return atom(form.base)
    if form.pointers == 1:
        if form.base == "char":
            return "ctypes.c_char_p"
        if form.base == "void":
            return "ctypes.c_void_p"
        if form.base in opaque:
            return handle_name(form.base)
        if form.base in structs:
            return f"ctypes.POINTER({class_name(form.base)})"
        return f"ctypes.POINTER({atom(form.base)})"
    if form.pointers == 2:
        if form.base == "char":
            return "ctypes.POINTER(ctypes.c_char_p)"
        if form.base in opaque:
            return f"ctypes.POINTER({handle_name(form.base)})"
        if form.base in structs:
            return f"ctypes.POINTER(ctypes.POINTER({class_name(form.base)}))"
        raise HeaderError(CAPI_H, form.line, form.text,
                          "a pointer to a pointer holds a handle, a struct, "
                          "or a string here")
    raise HeaderError(CAPI_H, form.line, form.text,
                      f"a pointer depth of {form.pointers} is not a form here")


def _ctypes_name(text: str) -> str:
    """The class name of one ``ctypes`` type, as ``_core.layout()`` spells it.

    The two sides have to agree on the aliases ``ctypes`` defines: on a 64-bit
    Linux host ``ctypes.c_int32`` is ``ctypes.c_int`` and ``ctypes.c_int64`` is
    ``ctypes.c_long``, so the canonical spelling is the name of the class a
    header type resolves to, and not the alias the header uses for it.
    """
    return getattr(ctypes, text.removeprefix("ctypes.")).__name__


def canonical_type(form: Form, header: Header) -> str:
    """The ``_core.layout()`` spelling of one struct field of the header.

    `_core` holds a real ``ctypes`` object per field, so a scalar is spelled by
    its class name, a mirrored structure by its C name, and a pointer or an
    array by its element, which is what makes ``POINTER(Record)`` read as
    ``POINTER(nanochat_record)`` on both sides.
    """
    structs = {struct.name for struct in header.structs}

    def atom(base: str) -> str:
        if base in SCALARS:
            return _ctypes_name(SCALARS[base])
        if base in ENUM_TYPES:
            return _ctypes_name("ctypes.c_int")
        if base in structs:
            return base
        raise HeaderError(CAPI_H, form.line, form.text,
                          f"no canonical spelling for the field type `{base}`")

    if form.array is not None:
        if form.base not in SCALARS:
            raise HeaderError(CAPI_H, form.line, form.text,
                              f"no canonical spelling for an array of "
                              f"`{form.base}`")
        return f"{atom(form.base)} * {form.array}"
    if form.pointers == 0:
        return atom(form.base)
    if form.pointers == 1:
        if form.base == "char":
            return _ctypes_name("ctypes.c_char_p")
        if form.base == "void" or form.base in header.opaque:
            return _ctypes_name("ctypes.c_void_p")
        return f"POINTER({atom(form.base)})"
    if form.pointers == 2:
        if form.base == "char":
            return f"POINTER({_ctypes_name('ctypes.c_char_p')})"
        if form.base in structs:
            return f"POINTER(POINTER({form.base}))"
        raise HeaderError(CAPI_H, form.line, form.text,
                          "a pointer to a pointer is not a mirrored field "
                          "here")
    raise HeaderError(CAPI_H, form.line, form.text,
                      f"a pointer depth of {form.pointers} is not a mirrored "
                      f"field here")


def _argument_types(function: Function, header: Header) -> list[str]:
    """The ctypes text of one function's arguments, in order."""
    rendered: list[str] = []
    for index, parameter in enumerate(function.parameters):
        override = ARGUMENT_OVERRIDES.get((function.name, index))
        rendered.append(override if override is not None
                        else render_type(parameter.form, header, "argument"))
    return rendered


def _tuple_text(arguments: list[str]) -> str:
    """One argument tuple, with the comma a one-element tuple needs."""
    if not arguments:
        return "()"
    if len(arguments) == 1:
        return f"({arguments[0]},)"
    return "(" + ", ".join(arguments) + ")"


def _signature_lines(name: str, arguments: list[str], result: str,
                     indent: str) -> list[str]:
    """One ``_SIGNATURES`` row, wrapped to `WIDTH` columns."""
    single = (f'{indent}("{name}", {_tuple_text(arguments)}, {result}),')
    if len(single) <= WIDTH:
        return [single]
    lines = [f'{indent}("{name}",']
    inner = indent + "    "
    continuation = indent + "     "
    current = inner + "("
    first = True
    alone = len(arguments) == 1
    for index, argument in enumerate(arguments):
        last = index + 1 == len(arguments)
        token = argument + (",)" if alone else (")" if last else ","))
        candidate = current + ("" if first else " ") + token
        if not first and len(candidate) > WIDTH:
            lines.append(current)
            current = continuation + token
        else:
            current = candidate
            first = False
    lines.append(current + ",")
    lines.append(f"{inner}{result}),")
    return lines


def signatures_body(header: Header, indent: str) -> list[str]:
    """The ``_SIGNATURES`` region, as body lines."""
    lines = [f"{indent}{SIGNATURES} = ("]
    for function in header.functions:
        lines.extend(_signature_lines(
            function.name, _argument_types(function, header),
            render_type(function.result, header, "result"), indent + "    "))
    lines.append(f"{indent})")
    return lines


# --- The C++ drift mirror -------------------------------------------------


def _enum_prefix(enum: str) -> str:
    """The enumerator prefix of a C enum: `nanochat_stage` -> `NANOCHAT_STAGE`."""
    return enum.upper()


def _enumerator_name(enum: str, enumerator: str) -> str:
    """The C++ spelling of one C enumerator: `NANOCHAT_STAGE_BASE` -> `kBase`."""
    prefix = _enum_prefix(enum) + "_"
    if not enumerator.startswith(prefix):
        raise TableError(f"`{enumerator}` is not a value of `{enum}`: it does "
                         f"not start with `{prefix}`")
    word = enumerator.removeprefix(prefix)
    return "k" + _upper_camel(word.lower())


def _string_lines(indent: str, message: str, tail: str) -> list[str]:
    """A C++ string literal, split across lines that fit the column limit."""
    lines: list[str] = []
    current = ""
    for word in message.split():
        candidate = word if not current else current + " " + word
        if current and len(indent) + len(candidate) + 2 + len(tail) > WIDTH + 1:
            lines.append(f'{indent}"{current} "')
            current = word
        else:
            current = candidate
    lines.append(f'{indent}"{current}"{tail}')
    return lines


def _assert_size_lines(left: str, operator: str, right: str,
                       message: str) -> list[str]:
    """One size assertion, in the aligned shape the block already uses.

    The aligned shape is what the file carried before the block was generated;
    a row whose names outgrow the column limit falls back to one indented
    argument per line, which is the shape `clang-format` picks for the same
    row.
    """
    aligned = [
        f"static_assert({left} {operator}",
        f"                  {right},",
    ]
    indent = "              "
    if max(len(line) for line in aligned) > WIDTH + 1:
        aligned = ["static_assert(", f"    {left} {operator}",
                   f"        {right},"]
        indent = "    "
    return aligned + _string_lines(indent, message, ");")


def _assert_same_lines(struct: str, cpp: str, field: str,
                       message: str) -> list[str]:
    """One `decltype` equality, in the shape the C++ block already uses."""
    left = f"decltype({struct}::{field})"
    right = f"decltype({cpp}::{field})"
    aligned = [
        f"static_assert(std::is_same<{left},",
        f"                           {right}>::value,",
    ]
    indent = "              "
    if max(len(line) for line in aligned) > WIDTH + 1:
        aligned = ["static_assert(", f"    std::is_same<{left},",
                   f"                 {right}>::value,"]
        indent = "    "
    return aligned + _string_lines(indent, message, ");")


def _assert_enum_lines(cpp: str, mirror: str, enumerator: str) -> list[str]:
    """One enumerator equality: a C code and the C++ value it must keep."""
    left = f"static_cast<int>({cpp}::{mirror})"
    aligned = [f"static_assert({left} == {enumerator},"]
    indent = "              "
    if len(aligned[0]) > WIDTH + 1:
        aligned = ["static_assert(", f"    {left} ==",
                   f"        {enumerator},"]
        indent = "    "
    return aligned + _string_lines(
        indent, f"{enumerator} drifted from {cpp}", ");")


def static_assert_body(header: Header, indent: str) -> list[str]:
    """The drift block, as body lines: one assert per pinned field and value."""
    lines = [f"{indent}{FORMAT_OFF}"]
    for struct in header.structs:
        if struct.name in NO_LAYOUT_MIRROR:
            continue
        if struct.name not in DRIFT_ROWS:
            raise TableError(
                f"`{struct.name}` is declared in {CAPI_H} and has no drift "
                f"row: add it to DRIFT_ROWS or to NO_LAYOUT_MIRROR")
        cpp, rows = DRIFT_ROWS[struct.name]
        if tuple(row.field for row in rows) != tuple(
                field.name for field in struct.fields):
            raise TableError(
                f"the drift rows of `{struct.name}` are not its fields in "
                f"header order: {CAPI_H} declares "
                f"{[field.name for field in struct.fields]}")
        for row in rows:
            if row.mode in (EQUAL, FLOOR):
                lines.extend(_assert_size_lines(
                    f"sizeof({struct.name}::{row.field})",
                    "==" if row.mode == EQUAL else ">=",
                    f"sizeof({cpp}::{row.field})",
                    f"{struct.name}.{row.field} drifted from {cpp}"))
            elif row.mode == SAME:
                lines.extend(_assert_same_lines(
                    struct.name, cpp, row.field,
                    f"{struct.name}.{row.field} drifted from {cpp}"))
            elif row.mode != SKIP:
                raise TableError(f"`{row.mode}` is not a drift mode")
    for struct in header.structs:
        if struct.name not in NO_LAYOUT_MIRROR:
            continue
        if not NO_LAYOUT_MIRROR[struct.name].strip():
            raise TableError(f"`{struct.name}` has no reason to be unpinned")
    for enum in header.enums:
        if enum.name in NO_ENUM_MIRROR:
            continue
        if enum.name not in ENUM_MIRRORS:
            raise TableError(
                f"`{enum.name}` is declared in {CAPI_H} and has no enum row: "
                f"add it to ENUM_MIRRORS or to NO_ENUM_MIRROR")
        cpp = ENUM_MIRRORS[enum.name]
        for enumerator, _ in enum.values:
            lines.extend(_assert_enum_lines(cpp, _enumerator_name(
                enum.name, enumerator), enumerator))
    lines.append(f"{indent}{FORMAT_ON}")
    return lines


# --- The regions ----------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One marked region: what it mirrors, where it sits, and its body."""

    path: str
    key: str
    indent: str
    marker_line: int
    body: tuple[str, ...]


def find_regions(path: str, text: str) -> list[Region]:
    """Every marked region of one file, in source order."""
    prefix = PREFIX[Path(path).suffix]
    begin = re.compile(rf"^(?P<indent>\s*){re.escape(prefix)} {re.escape(BEGIN)}"
                       r"(?:\s+(?P<key>\S+))?\s*$")
    end = re.compile(rf"^(?P<indent>\s*){re.escape(prefix)} {re.escape(END)}"
                     r"\s*$")
    lines = text.split("\n")
    regions: list[Region] = []
    index = 0
    while index < len(lines):
        opening = begin.match(lines[index])
        if opening is None:
            index += 1
            continue
        key = opening.group("key")
        if key is None:
            raise TableError(f"{path}:{index + 1}: the marker names no region")
        for closing in range(index + 1, len(lines)):
            if end.match(lines[closing]) is not None:
                regions.append(Region(
                    path=path, key=key, indent=opening.group("indent"),
                    marker_line=index + 1,
                    body=tuple(lines[index + 1:closing])))
                index = closing + 1
                break
        else:
            raise TableError(f"{path}:{index + 1}: the marker pair for "
                             f"`{key}` has no END")
    return regions


def expected_body(region: Region, header: Header) -> list[str]:
    """The body `--write` puts in one region."""
    if region.path == CORE_PY and region.key == SIGNATURES:
        return signatures_body(header, region.indent)
    if region.path == CAPI_TEST_CC and region.key == STATIC_ASSERT:
        return static_assert_body(header, region.indent)
    raise TableError(f"{region.path}:{region.marker_line}: `{region.key}` is "
                     f"not a region this script generates")


def committed_regions(root: Path) -> list[Region]:
    """Every marked region of the two mirrors, with the bytes it holds.

    A region this script owns with no marker pair is a failure here, not a
    silent skip: a mirror nobody marks is a mirror nobody maintains.
    """
    regions: list[Region] = []
    for path in (CORE_PY, CAPI_TEST_CC):
        regions.extend(find_regions(path, (root / path).read_text(
            encoding="utf-8")))
    keys = {region.key for region in regions if region.path == CORE_PY}
    if SIGNATURES not in keys:
        raise TableError(f"{CORE_PY}: no region mirrors `{SIGNATURES}`; add "
                         f"the marker pair by hand")
    if not any(region.path == CAPI_TEST_CC for region in regions):
        raise TableError(f"{CAPI_TEST_CC}: no region mirrors the drift block; "
                         f"add the marker pair by hand")
    return regions


# --- The layout table -----------------------------------------------------


def core_layout(root: Path) -> dict[str, list[tuple[str, str]]]:
    """The layout table of `python/nanochat_cpp/_core.py`, as it reports it.

    The module is standard-library only and loads no shared library, so it can
    be run here, and its own table is then the one the check compares with the
    header instead of a second copy of it. The bytes on disk are compiled
    directly: a ``__pycache__`` entry cannot answer for a table that changed,
    because a same-size edit inside one second leaves the cached bytecode
    looking fresh.
    """
    path = root / CORE_PY
    module = types.ModuleType("_nanochat_capi_core")
    module.__file__ = str(path)
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
             module.__dict__)
        return module.layout()
    except OSError as error:
        raise TableError(f"{CORE_PY}: the module cannot be read: "
                         f"{error}") from error
    # A broken table is a failed check, not a crash.
    except Exception as error:
        raise TableError(f"{CORE_PY}: `_LAYOUT` does not yield a layout: "
                         f"{error}") from error


# --- The comparison -------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """One failed comparison, in the `path:line: message` shape of `lint`."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def layout_mismatches(
        header: Header,
        mirror: dict[str, list[tuple[str, str]]]) -> list[Mismatch]:
    """The disagreements between the header and the layout table of `_core`.

    `mirror` is what `_core.layout()` returns. The two directions are covered
    apart from the rows: a structure the header declares and the table does not
    classify, and a structure the table names and the header never declared.
    """
    mismatches: list[Mismatch] = []
    declared = {struct.name for struct in header.structs}
    for name in mirror:
        if name not in declared:
            mismatches.append(Mismatch(
                CORE_PY, f"the `_LAYOUT` table names `{name}`, and {CAPI_H} "
                         f"declares no such structure"))
    for struct in header.structs:
        rows = mirror.get(struct.name)
        if rows is None:
            mismatches.append(Mismatch(
                CORE_PY, f"`{struct.name}`: the `_LAYOUT` table does not "
                         f"classify it, and {CAPI_H} declares it"))
            continue
        position = {name: index for index, (name, _) in enumerate(rows)}
        expected = {field.name: index
                    for index, field in enumerate(struct.fields)}
        for index, field in enumerate(struct.fields):
            spelled = canonical_type(field.form, header)
            at = position.get(field.name)
            if at is None:
                mismatches.append(Mismatch(
                    CORE_PY, f"`{struct.name}.{field.name}`: the `_LAYOUT` "
                             f"table has no row for it, and {CAPI_H} declares "
                             f"it as `{field.form.describe()}` ({spelled})"))
            elif at != index:
                mismatches.append(Mismatch(
                    CORE_PY, f"`{struct.name}.{field.name}`: the `_LAYOUT` "
                             f"table puts it at position {at + 1}, and "
                             f"{CAPI_H} declares it at position {index + 1}"))
            elif rows[at][1] != spelled:
                mismatches.append(Mismatch(
                    CORE_PY, f"`{struct.name}.{field.name}`: the `_LAYOUT` "
                             f"table spells it `{rows[at][1]}`, and {CAPI_H} "
                             f"declares `{field.form.describe()}` "
                             f"({spelled})"))
        for name, _ in rows:
            if name not in expected:
                mismatches.append(Mismatch(
                    CORE_PY, f"`{struct.name}.{name}`: the `_LAYOUT` table "
                             f"names it, and {CAPI_H} declares no such "
                             f"field"))
    return mismatches


def _canonical(text: str) -> str:
    """One type expression with its spacing normalized, for comparison."""
    try:
        return ast.unparse(ast.parse(text, mode="eval"))
    except SyntaxError:
        return " ".join(text.split())


def _signature_rows(body: tuple[str, ...], where: str,
                    ) -> list[tuple[str, tuple[str, ...], str]]:
    """The `(name, arguments, result)` rows of a committed `_SIGNATURES`."""
    text = textwrap.dedent("\n".join(body))
    try:
        module = ast.parse(text)
        assignment = module.body[0]
        rows: list[tuple[str, tuple[str, ...], str]] = []
        for entry in assignment.value.elts:
            name = ast.literal_eval(entry.elts[0])
            arguments = tuple(_canonical(ast.unparse(argument))
                              for argument in entry.elts[1].elts)
            rows.append((name, arguments, _canonical(
                ast.unparse(entry.elts[2]))))
        return rows
    except (SyntaxError, AttributeError, IndexError, ValueError) as error:
        raise TableError(f"{where}: the region is not a `_SIGNATURES` table: "
                         f"{error}") from error


_ASSERT_ROW = re.compile(
    r"static_assert\(sizeof\((?P<size_struct>\w+)::(?P<size_field>\w+)\)\s*"
    r"(?P<operator>==|>=)")
_ASSERT_SAME = re.compile(
    r"static_assert\(std::is_same<decltype\((?P<same_struct>\w+)::"
    r"(?P<same_field>\w+)\)")
_ASSERT_ENUM = re.compile(
    r"static_assert\(static_cast<int>\((?P<cpp>[\w:]+)::(?P<mirror>\w+)\)\s*"
    r"==\s*(?P<enumerator>\w+)")


def _assert_rows(body: tuple[str, ...]) -> dict[tuple[str, str], str]:
    """The `(owner, field)` rows of a committed drift block, with their mode."""
    text = "\n".join(body)
    rows: dict[tuple[str, str], str] = {}
    for match in _ASSERT_ROW.finditer(text):
        mode = EQUAL if match.group("operator") == "==" else FLOOR
        rows[(match.group("size_struct"), match.group("size_field"))] = mode
    for match in _ASSERT_SAME.finditer(text):
        rows[(match.group("same_struct"), match.group("same_field"))] = SAME
    for match in _ASSERT_ENUM.finditer(text):
        rows[(match.group("cpp"), match.group("enumerator"))] = "enum"
    return rows


def _expected_assert_rows(header: Header) -> dict[tuple[str, str], str]:
    """The drift rows the table pins, keyed the way the block writes them."""
    rows: dict[tuple[str, str], str] = {}
    for struct in header.structs:
        if struct.name in NO_LAYOUT_MIRROR:
            continue
        _, table = DRIFT_ROWS[struct.name]
        for row in table:
            if row.mode == SKIP:
                continue
            rows[(struct.name, row.field)] = row.mode
    for enum in header.enums:
        if enum.name in NO_ENUM_MIRROR:
            continue
        cpp = ENUM_MIRRORS[enum.name]
        for enumerator, _ in enum.values:
            rows[(cpp, enumerator)] = "enum"
    return rows


def _row_mismatches(region: Region, header: Header) -> list[Mismatch]:
    """The row-level disagreements between one region and the header.

    This is the detailed half of the check. It compares row by row rather than
    byte by byte, so a failure names the structure and the field that moved
    instead of reporting that a region changed.
    """
    where = f"{region.path}:{region.marker_line}"
    emitted = tuple(expected_body(region, header))
    if region.key == SIGNATURES:
        found = _signature_rows(region.body, where)
        expected = _signature_rows(emitted, where)
        mismatches: list[Mismatch] = []
        for name, arguments, result in expected:
            match = [row for row in found if row[0] == name]
            if not match:
                mismatches.append(Mismatch(
                    where, f"`{name}` has no row in {SIGNATURES}: {CAPI_H} "
                           f"declares it"))
                continue
            got_arguments, got_result = match[0][1], match[0][2]
            if len(got_arguments) != len(arguments):
                mismatches.append(Mismatch(
                    where, f"`{name}`: the mirror declares "
                           f"{len(got_arguments)} argument(s), {CAPI_H} "
                           f"declares {len(arguments)}"))
                continue
            for index, (got, want) in enumerate(zip(got_arguments, arguments)):
                if got != want:
                    mismatches.append(Mismatch(
                        where, f"`{name}`: argument {index + 1} is `{got}` in "
                               f"the mirror and `{want}` in {CAPI_H}"))
            if got_result != result:
                mismatches.append(Mismatch(
                    where, f"`{name}`: the result is `{got_result}` in the "
                           f"mirror and `{result}` in {CAPI_H}"))
        for name in {row[0] for row in found} - {row[0] for row in expected}:
            mismatches.append(Mismatch(
                where, f"`{name}` has a row in {SIGNATURES}, and {CAPI_H} "
                       f"declares no such function"))
        return mismatches
    # `expected_body` accepted the region, so this one is the drift block.
    found = _assert_rows(region.body)
    expected = _assert_rows(emitted)
    mismatches = []
    for (owner, field), mode in expected.items():
        got = found.get((owner, field))
        if got is None:
            mismatches.append(Mismatch(
                where, f"`{owner}.{field}`: the drift block has no "
                       f"`{mode}` assertion for it"))
        elif got != mode:
            mismatches.append(Mismatch(
                where, f"`{owner}.{field}`: the drift block asserts "
                       f"`{got}`, and the row is `{mode}`"))
    for owner, field in found:
        if (owner, field) not in expected:
            mismatches.append(Mismatch(
                where, f"`{owner}.{field}`: the drift block asserts a row "
                       f"that the header does not supply"))
    return mismatches


def check(root: Path) -> int:
    """Compare every mirror against the header. The exit code."""
    text = (root / CAPI_H).read_text(encoding="utf-8")
    header = parse_header(CAPI_H, text)
    if not header.structs or not header.functions:
        print(f"{CAPI_H}: the parse found no structure or no function",
              file=sys.stderr)
        return 1
    mismatches: list[Mismatch] = layout_mismatches(header, core_layout(root))
    for region in committed_regions(root):
        if list(region.body) == expected_body(region, header):
            continue
        rows = _row_mismatches(region, header)
        mismatches.extend(rows if rows else [Mismatch(
            f"{region.path}:{region.marker_line}",
            "the region is not what --write emits; run "
            "`tools/gen_capi_mirrors.py --write`")])
    if mismatches:
        for mismatch in mismatches:
            print(f"FAIL: {mismatch}", file=sys.stderr)
        return 1
    fields = sum(len(struct.fields) for struct in header.structs)
    pinned = len(_expected_assert_rows(header))
    print(f"checked {len(header.structs)} structures ({fields} fields), "
          f"{len(header.functions)} functions, and {pinned} drift rows of "
          f"{CORE_PY} and {CAPI_TEST_CC}: the three mirrors agree with "
          f"{CAPI_H}", file=sys.stderr)
    for struct in header.structs:
        if struct.name in NO_LAYOUT_MIRROR:
            print(f"recorded: `{struct.name}` has no drift row: "
                  f"{NO_LAYOUT_MIRROR[struct.name]}", file=sys.stderr)
    for enum in header.enums:
        if enum.name in NO_ENUM_MIRROR:
            print(f"recorded: `{enum.name}` has no enum assertion: "
                  f"{NO_ENUM_MIRROR[enum.name]}", file=sys.stderr)
    return 0


def write(root: Path) -> int:
    """Rewrite every generated region in place. The exit code."""
    header = parse_header(str(root / CAPI_H),
                          (root / CAPI_H).read_text(encoding="utf-8"))
    regions = committed_regions(root)
    for path in dict.fromkeys(region.path for region in regions):
        source = root / path
        text = source.read_text(encoding="utf-8")
        lines = text.split("\n")
        updated = list(lines)
        for region in sorted((entry for entry in regions
                              if entry.path == path),
                             key=lambda entry: entry.marker_line, reverse=True):
            end = region.marker_line + len(region.body)
            updated[region.marker_line:end] = expected_body(region, header)
        if updated != lines:
            source.write_text("\n".join(updated), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the ctypes mirror, the signature table, and the "
                    "drift block from include/nanochat/capi.h, or check them.")
    parser.add_argument("--write", action="store_true",
                        help="rewrite every generated region")
    parser.add_argument("--check", action="store_true",
                        help="check every mirror against the header; write "
                             "nothing")
    parser.add_argument("--root", default=str(REPO_ROOT),
                        help="the repository root (default: this checkout)")
    arguments = parser.parse_args(argv)
    if arguments.write == arguments.check:
        parser.error("name exactly one of --write and --check")
    root = Path(arguments.root).resolve()
    try:
        return write(root) if arguments.write else check(root)
    except (HeaderError, TableError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
