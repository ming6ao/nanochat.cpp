// The kernel body of one entry of the CUTLASS bring-up probe (docs/kernels.md
// Part 3 section 5). **One compiler action per entry**: the build compiles this
// source once for each index of `NANOCHAT_FOR_EACH_PROBE_ENTRY` and passes the
// index as `NANOCHAT_PROBE_ENTRY`. One process therefore holds one Hopper
// kernel body, which costs 42 s and 3.0 GiB here.
//
// **Taking the address is the point.** `cutlass::device_kernel<Kernel>` is a
// `__global__` function template, so the body is compiled only where something
// odr-uses it. A translation unit that merely names the type compiles the class
// and skips the mainloop, and the probe's first build shipped an object with no
// kernel in it for that reason. A mainloop that does not build fails here, and
// one that builds leaves SASS for `cuobjdump`.

#include "dev/cutlass/cutlass_probe_rows.h"
#include "dev/cutlass/cutlass_probe_types.h"

#if !defined(NANOCHAT_PROBE_ENTRY)
#error \
    "dev/cutlass/cutlass_probe_entry.cu is compiled once for each entry; the target states local_defines = [\"NANOCHAT_PROBE_ENTRY=<index>\"]"
#endif

namespace nanochat {
namespace cutlass_probe {

//: The entry name of this index, which `cutlass_probe_rows.h` declares. The two
//: levels of macro paste the index into the name.
#define NANOCHAT_PROBE_ENTRY_NAME_INNER(index) CutlassProbeEntry##index
#define NANOCHAT_PROBE_ENTRY_NAME(index) NANOCHAT_PROBE_ENTRY_NAME_INNER(index)

//: The entry's kernel address, and the reason this translation unit exists.
const void* NANOCHAT_PROBE_ENTRY_NAME(NANOCHAT_PROBE_ENTRY)() {
  using Entry = EntryType<NANOCHAT_PROBE_ENTRY>;
  return reinterpret_cast<const void*>(
      &cutlass::device_kernel<typename Entry::Kernel>);
}

#undef NANOCHAT_PROBE_ENTRY_NAME
#undef NANOCHAT_PROBE_ENTRY_NAME_INNER

}  // namespace cutlass_probe
}  // namespace nanochat
