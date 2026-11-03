#ifndef NANOCHAT_SRC_KERNEL_PARAMS_CHECK_H_
#define NANOCHAT_SRC_KERNEL_PARAMS_CHECK_H_

// Debug-only precondition checks at the kernel seam (docs/api-surface.md R3).
//
// The seam has no failure channel (`include/nanochat/kernels.h`): a bad
// parameter is a wrong number, not a status, and the oracle is the release
// gate. This header gives the compositions in `src/ops.cc` one way to state
// what the family they call assumes, so a debug build names the boundary and
// the condition that a caller broke instead of returning a plausible wrong
// value.
//
// A release build pays nothing. Under NDEBUG the macro is `((void)0)`: not a
// branch that is usually taken, but no code at all, so the parameter is not
// even read. The check calls no kernel, reads no device memory, and waits for
// nothing, so a debug build keeps the asynchrony of a release build.

#ifdef NDEBUG

#define NANOCHAT_CHECK_KERNEL_PARAMS(family, condition) ((void)0)

#else

#include <cstdio>
#include <cstdlib>

// Aborts unless `condition` holds. `family` names the seam family the call
// belongs to, spelled as a string literal, and the condition is echoed as
// written, so the message carries both the boundary and the failed assumption.
#define NANOCHAT_CHECK_KERNEL_PARAMS(family, condition)                       \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "nanochat: kernel parameters: %s: %s\n", (family), \
                   #condition);                                               \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

#endif  // NDEBUG

#endif  // NANOCHAT_SRC_KERNEL_PARAMS_CHECK_H_
