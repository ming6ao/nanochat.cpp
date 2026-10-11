"""Repository rule for libibverbs (docs/distributed-native-plan.md phase 0).

The RDMA transport is the second transport of the native collective. libibverbs
is a system library, not a Bazel module, so this rule looks for the header in
`NANOCHAT_RDMA_PATH`, then in a few standard prefixes, and copies the header and
the shared library into the repository.

The generated BUILD defines `ibverbs`. When the library is present it uses a
`cc_import`; when it is absent the `cc_library` still has the `-libverbs`
linkopt, so a target that references it fails at link with a clear error. No
target references it by default, because the RDMA transport is optional and the
transport selection falls back to host staging when no adapter is present.
"""

def _rdma_repo_impl(repository_ctx):
    prefixes = []
    env = repository_ctx.os.environ.get("NANOCHAT_RDMA_PATH", "")
    if env:
        prefixes.append(env)
    prefixes += ["/usr", "/usr/local", "/opt/rdma"]

    prefix = None
    for candidate in prefixes:
        if repository_ctx.path(candidate + "/include/infiniband/verbs.h").exists:
            prefix = candidate
            break

    lib_name = None
    if prefix != None:
        header = prefix + "/include/infiniband/verbs.h"
        repository_ctx.file(
            "include/infiniband/verbs.h",
            repository_ctx.read(header),
        )
        lib_dir = repository_ctx.path(prefix + "/lib")
        if lib_dir.exists:
            unversioned = prefix + "/lib/libibverbs.so"
            versioned = None
            for entry in lib_dir.readdir():
                name = entry.basename
                if name == "libibverbs.so":
                    unversioned = str(entry)
                elif name.startswith("libibverbs.so.") and versioned == None:
                    versioned = str(entry)
            chosen = unversioned if repository_ctx.path(unversioned).exists else versioned
            if chosen != None:
                lib_name = chosen.split("/")[-1]
                repository_ctx.download(
                    url = "file://" + chosen,
                    output = "lib/" + lib_name,
                )
    else:
        repository_ctx.file("include/infiniband/.keep", "")

    lines = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]
    if lib_name != None:
        lines += [
            "cc_import(",
            '    name = "ibverbs_lib",',
            '    shared_library = "lib/%s",' % lib_name,
            ")",
            "cc_library(",
            '    name = "ibverbs",',
            '    hdrs = glob(["include/infiniband/*.h"]),',
            '    includes = ["include"],',
            '    deps = [":ibverbs_lib"],',
            ")",
        ]
    else:
        lines += [
            "cc_library(",
            '    name = "ibverbs",',
            '    hdrs = glob(["include/infiniband/*.h"]),',
            '    includes = ["include"],',
            '    linkopts = ["-libverbs"],',
            ")",
        ]
    repository_ctx.file("BUILD.bazel", "\n".join(lines) + "\n")

rdma_repository = repository_rule(
    implementation = _rdma_repo_impl,
    environ = ["NANOCHAT_RDMA_PATH"],
)
