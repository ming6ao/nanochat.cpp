"""Repository rule for NVIDIA CUTLASS (docs/kernels.md Part 3 section 5).

CUTLASS supplies the collective builder, the Hopper warp-specialized mainloop,
the Tensor Memory Accelerator copy atoms, the pipeline, the epilogue, and the
tile schedulers that `backends/cuda/gemm/cutlass_bf16.cu` drives. It is a
header-only library, so this rule fetches the pinned archive and generates one
`cc_library` over its `include/` tree. Nothing is compiled here.

**The tree is fetched, not tracked, and that is a correctness rule.** The tree's
lint pass (`python/nanochat_cpp/lint.py`) checks every source that
`git ls-files` reports: no bare include, no C-style cast, no `using namespace`,
ASCII only. The CUTLASS headers break all four of those rules on purpose, and
they are not ours to reformat. An external repository keeps them out of the lint
pass, out of the format gate, and out of a diff.

**`include/` only.** The archive also holds `test/`, `examples/`, `python/`, and
the media files, which take 169 MB against the 28.5 MB of `include/`. Only
`include/` and `LICENSE.txt` are extracted, so the fetch stays small and the
glob stays narrow.

The commit is pinned twice: by URL and by SHA-256. A host with no network, or a
host that wants to build against a local checkout, sets
`NANOCHAT_CUTLASS_PATH` to a directory that holds `include/`, and the rule copies
it instead of downloading.

The generated BUILD states `includes = ["include"]`, which Bazel turns into
`-isystem` for a dependent, so CUTLASS's own warnings do not reach a consumer's
warning policy. That matters: every build of the tree turns a warning into an
error, and the headers emit about fifteen thousand warnings.
"""

# The pinned CUTLASS. `docs/kernels.md` Part 3 section 4 names this commit as the
# reference for the Hopper path, and the tag beside it is `v4.8.0-23`, whose
# `cutlass/version.h` reports 4.8.1.
_CUTLASS_COMMIT = "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"
_CUTLASS_SHA256 = "73911a768a82242c142453ecd86a83c2ddf046388d83c04e9e36df0f83b9aa0e"

# The two members the generated repository keeps from the archive.
_WANTED = ("include", "LICENSE.txt")


def _build_file():
    return "\n".join([
        'load("@rules_cc//cc:defs.bzl", "cc_library")',
        "",
        'package(default_visibility = ["//visibility:public"])',
        "",
        "# Header only: the rule builds no source, and a dependent compiles the",
        "# headers it includes. `includes` gives Bazel's -isystem, so the",
        "# vendor warnings stay out of the dependent's warning policy.",
        "# The tree is 494 .h, 299 .hpp, and 45 .inl files. The .inl files are not",
        "# optional: the sparse and block-scaled mainloops include their builder",
        "# configuration from one, and a missing glob entry is a fatal include.",
        "cc_library(",
        '    name = "cutlass",',
        '    hdrs = glob(["include/**/*.h", "include/**/*.hpp", "include/**/*.inl"]),',
        '    includes = ["include"],',
        ")",
        "",
        'exports_files(["LICENSE.txt"])',
        "",
    ])


def _copy_local(repository_ctx, root):
    for member in _WANTED:
        source = repository_ctx.path(root + "/" + member)
        if not source.exists:
            fail(("NANOCHAT_CUTLASS_PATH is %s, but %s is not there. Point it " +
                  "at a CUTLASS checkout that holds include/ and LICENSE.txt.")
                 % (root, member))
        repository_ctx.symlink(source, member)


def _extract_archive(repository_ctx, url):
    archive = "cutlass.tar.gz"
    repository_ctx.download(output = archive, url = url, sha256 = _CUTLASS_SHA256)
    # The archive's one top-level directory is named after the commit, so
    # `--strip-components=1` and the member list keep the two wanted trees and
    # nothing else. `tar` is the host's, which every supported host has.
    result = repository_ctx.execute([
        "tar",
        "-xzf",
        archive,
        "--strip-components=1",
        "cutlass-" + _CUTLASS_COMMIT + "/include",
        "cutlass-" + _CUTLASS_COMMIT + "/LICENSE.txt",
    ])
    repository_ctx.delete(archive)
    if result.return_code != 0:
        fail("cutlass.bzl: tar could not extract %s:\n%s" % (archive, result.stderr))


def _cutlass_repo_impl(repository_ctx):
    root = repository_ctx.os.environ.get("NANOCHAT_CUTLASS_PATH", "")
    if root:
        _copy_local(repository_ctx, root)
    else:
        _extract_archive(
            repository_ctx,
            "https://github.com/NVIDIA/cutlass/archive/%s.tar.gz" % _CUTLASS_COMMIT,
        )
    repository_ctx.file("BUILD.bazel", _build_file())
    repository_ctx.file("NOTICE", "\n".join([
        "CUTLASS, NVIDIA CORPORATION. All rights reserved.",
        "Licensed under the BSD 3-Clause License. See LICENSE.txt.",
        "Pinned at commit %s." % _CUTLASS_COMMIT,
        "",
    ]))


cutlass_repository = repository_rule(
    implementation = _cutlass_repo_impl,
    environ = ["NANOCHAT_CUTLASS_PATH"],
    doc = "The pinned CUTLASS headers as one `cc_library` (docs/build.md).",
)
