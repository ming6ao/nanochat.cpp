#ifndef NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_ROWS_H_
#define NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_ROWS_H_

// The entry list of the CUTLASS bring-up probe, and the kernel address each row
// reads. This header names no CUDA type, so a host target can read it.
//
// `cutlass_probe_entry.cu` is compiled once for each entry, with
// `NANOCHAT_PROBE_ENTRY` naming the entry, so one `nvcc` process holds one
// kernel body. `cutlass_probe.cu` reads the addresses through these
// declarations. A missing index is a link error, and an extra one is unused.

namespace nanochat {
namespace cutlass_probe {

//: The rows the report holds: the five tiles, then the eight seam cases.
constexpr int kRowCount = 13;

//: The kernel bodies the build compiles: one per distinct row type.
constexpr int kEntryCount = 12;

//: The entry indexes. An index is the value a target passes as
//: `NANOCHAT_PROBE_ENTRY`.
#define NANOCHAT_FOR_EACH_PROBE_ENTRY(X) \
  X(0)                                   \
  X(1)                                   \
  X(2)                                   \
  X(3)                                   \
  X(4)                                   \
  X(5)                                   \
  X(6)                                   \
  X(7)                                   \
  X(8)                                   \
  X(9)                                   \
  X(10)                                  \
  X(11)

//: The kernel address of one entry, as the target that compiled it defines it.
#define NANOCHAT_DECLARE_PROBE_ENTRY(index) \
  const void* CutlassProbeEntry##index();
NANOCHAT_FOR_EACH_PROBE_ENTRY(NANOCHAT_DECLARE_PROBE_ENTRY)
#undef NANOCHAT_DECLARE_PROBE_ENTRY

}  // namespace cutlass_probe
}  // namespace nanochat

#endif  // NANOCHAT_DEV_CUTLASS_CUTLASS_PROBE_ROWS_H_
