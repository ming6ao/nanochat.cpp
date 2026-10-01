#ifndef NANOCHAT_SANDBOX_H_
#define NANOCHAT_SANDBOX_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Host-side guard that fails fast when an executable is launched outside the
// resource sandbox, instead of letting it consume the whole machine. There is
// no compute logic here and no vendor dependency, so it is safe to include from
// any entry point.
//
// tools/nanochat sets NANOCHAT_SANDBOX to the profile name; see docs/sandbox.md.

namespace nanochat {

// The profile name set by tools/nanochat, or "" when unsandboxed.
inline const char* SandboxProfile() {
  const char* profile = std::getenv("NANOCHAT_SANDBOX");
  return profile != nullptr ? profile : "";
}

// True when NANOCHAT_ALLOW_UNSANDBOXED is set to a non-false value.
inline bool SandboxOverrideSet() {
  const char* value = std::getenv("NANOCHAT_ALLOW_UNSANDBOXED");
  if (value == nullptr || *value == '\0') return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

// Call at the top of every executable entry point (train, eval, generate, and
// every test main). Exits non-zero when the process was not launched through
// tools/nanochat, printing the corrective command. `task` names the command in
// that message, for example "train" or "test".
inline void RequireSandboxOrDie(const char* task) {
  if (*SandboxProfile() != '\0' || SandboxOverrideSet()) return;
  std::fprintf(stderr,
               "fatal: %s must run through tools/nanochat (no sandbox "
               "configured).\n"
               "  run: tools/nanochat help\n"
               "  one-off override: NANOCHAT_ALLOW_UNSANDBOXED=1 %s ...\n"
               "See docs/sandbox.md.\n",
               task, task);
  std::exit(2);
}

}  // namespace nanochat

#endif  // NANOCHAT_SANDBOX_H_
