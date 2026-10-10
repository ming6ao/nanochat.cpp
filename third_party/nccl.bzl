"""Repository rule for NCCL (docs/distributed-plan.md phase 1).

NCCL is a system library, not a Bazel module. The rule looks for the headers in
`NANOCHAT_NCCL_PATH`, then in a few standard prefixes. It always creates an
`nccl` target.

When the headers are absent, a CUDA build fails at compile time with a missing
`nccl.h`. The CPU build never reaches that target, because the CUDA backend
selects it away (docs/distributed-plan.md phase 0).
"""

def _nccl_repo_impl(repository_ctx):
    prefixes = []
    env = repository_ctx.os.environ.get("NANOCHAT_NCCL_PATH", "")
    if env:
        prefixes.append(env)
    prefixes += ["/usr", "/usr/local", "/opt/nccl"]

    prefix = None
    for candidate in prefixes:
        if repository_ctx.path(candidate + "/include/nccl.h").exists:
            prefix = candidate
            break

    linkopts = ["-lnccl"]
    if prefix != None:
        for name in ["nccl.h", "nccl_net.h"]:
            header = prefix + "/include/" + name
            if repository_ctx.path(header).exists:
                # Copy the header into the repository, so the Bazel sandbox
                # always has it. A symlink to a system path can escape the
                # sandbox.
                repository_ctx.file("include/" + name, repository_ctx.read(header))
        lib_dir = prefix + "/lib"
        if repository_ctx.path(lib_dir).exists:
            linkopts = [
                "-L" + lib_dir,
                "-Wl,-rpath," + lib_dir,
                "-lnccl",
            ]
    else:
        # Keep the `includes` directory valid when NCCL is absent. The CUDA
        # build then fails at the `#include <nccl.h>` line with a clear error.
        repository_ctx.file("include/.keep", "")

    repository_ctx.file(
        "BUILD.bazel",
        "\n".join([
            'load("@rules_cc//cc:defs.bzl", "cc_library")',
            'package(default_visibility = ["//visibility:public"])',
            "cc_library(",
            '    name = "nccl",',
            '    hdrs = glob(["include/*.h"]),',
            '    includes = ["include"],',
            "    linkopts = %s," % linkopts,
            ")",
        ]) + "\n",
    )

nccl_repository = repository_rule(
    implementation = _nccl_repo_impl,
    environ = ["NANOCHAT_NCCL_PATH"],
)
