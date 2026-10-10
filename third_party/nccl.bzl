"""Repository rule for NCCL (docs/distributed-plan.md phase 1).

NCCL is a system library, not a Bazel module. The rule looks for the headers in
`NANOCHAT_NCCL_PATH`, then in a few standard prefixes. It copies the header and
the shared library into the repository, so the Bazel sandbox always has them.

The generated BUILD defines `nccl`. When the library is present it uses a
`cc_import`, so Bazel treats the shared object as a link input and an rpath
entry. When NCCL is absent, the CUDA build fails at the `#include <nccl.h>` line
with a clear error. The CPU build never reaches the target, because the CUDA
backend selects it away (docs/distributed-plan.md phase 0).
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

    lib_name = None
    if prefix != None:
        for name in ["nccl.h", "nccl_net.h"]:
            header = prefix + "/include/" + name
            if repository_ctx.path(header).exists:
                repository_ctx.file(
                    "include/" + name,
                    repository_ctx.read(header),
                )
        lib_dir = repository_ctx.path(prefix + "/lib")
        if lib_dir.exists:
            unversioned = prefix + "/lib/libnccl.so"
            versioned = None
            for entry in lib_dir.readdir():
                name = entry.basename
                if name == "libnccl.so":
                    unversioned = str(entry)
                elif name.startswith("libnccl.so.") and versioned == None:
                    versioned = str(entry)
            chosen = unversioned if repository_ctx.path(unversioned).exists else versioned
            if chosen != None:
                lib_name = chosen.split("/")[-1]
                repository_ctx.download(
                    url = "file://" + chosen,
                    output = "lib/" + lib_name,
                )
    else:
        # Keep the `includes` directory valid when NCCL is absent.
        repository_ctx.file("include/.keep", "")

    lines = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]
    if lib_name != None:
        lines += [
            "cc_import(",
            '    name = "nccl_lib",',
            '    shared_library = "lib/%s",' % lib_name,
            ")",
            "cc_library(",
            '    name = "nccl",',
            '    hdrs = glob(["include/*.h"]),',
            '    includes = ["include"],',
            '    deps = [":nccl_lib"],',
            ")",
        ]
    else:
        lines += [
            "cc_library(",
            '    name = "nccl",',
            '    hdrs = glob(["include/*.h"]),',
            '    includes = ["include"],',
            '    linkopts = ["-lnccl"],',
            ")",
        ]
    repository_ctx.file("BUILD.bazel", "\n".join(lines) + "\n")

nccl_repository = repository_rule(
    implementation = _nccl_repo_impl,
    environ = ["NANOCHAT_NCCL_PATH"],
)
