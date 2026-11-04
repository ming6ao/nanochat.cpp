#ifndef NANOCHAT_SANDBOX_H_
#define NANOCHAT_SANDBOX_H_

#include <cstdio>
#include <cstdlib>

// Host-side guard that fails fast when an executable is launched outside the
// resource sandbox, instead of letting it consume the whole machine. There is
// no compute logic here and no vendor dependency, so it is safe to include from
// any entry point.
//
// tools/nanochat sets NANOCHAT_SANDBOX to the profile name. The `none` backend
// sets it to `off`, which records a deliberate unsandboxed run on a host
// without cgroup v2. One variable states the whole rule. See docs/sandbox.md.

namespace nanochat {

// The sentinel set by tools/nanochat, or "" when the launcher did not run.
// The value `off` is an accepted sentinel like any profile name.
inline const char* SandboxProfile() {
  const char* profile = std::getenv("NANOCHAT_SANDBOX");
  return profile != nullptr ? profile : "";
}

// Call at the top of every executable entry point (train, eval, generate, and
// every test main). Exits non-zero when the process was not launched through
// tools/nanochat, printing the corrective command. `task` names the command in
// that message, for example "train" or "test".
inline void RequireSandboxOrDie(const char* task) {
  if (*SandboxProfile() != '\0') return;
  std::fprintf(stderr,
               "fatal: %s must run through tools/nanochat (no sandbox "
               "sentinel set).\n"
               "  run: tools/nanochat help\n"
               "  no cgroup v2: NANOCHAT_SANDBOX=off %s ...\n"
               "See docs/sandbox.md.\n",
               task, task);
  std::exit(2);
}

}  // namespace nanochat

#endif  // NANOCHAT_SANDBOX_H_
