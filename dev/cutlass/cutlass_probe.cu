// The report of the CUTLASS bring-up probe (docs/kernels.md Part 3 section 5):
// the two row tables, the row-to-entry map, and the window test. It names every
// row's type and emits no kernel body, so one action costs 17 s and 1.3 GiB
// against 42 s and 3.0 GiB for a body. The assertions below keep the tables,
// the row count, and the entry count in step.

#include <cstddef>

#include "dev/cutlass/cutlass_probe.h"
#include "dev/cutlass/cutlass_probe_rows.h"
#include "dev/cutlass/cutlass_probe_types.h"

namespace nanochat {
namespace cutlass_probe {
namespace {

//: The window one Hopper block may ask for: 227 KiB of the 228 KiB the SM has.
constexpr std::size_t kWindowBytes = 227 * 1024;

//: The two tables together are the row list, tiles first. A new row is a change
//: here, to `kEntryOfRow`, to `Entries`, and to the build's entry list.
constexpr int kTileRows = 5;
constexpr int kCaseRows = 8;

const TileReport kTiles[] = {
    {"128x256x64", 128, 256, 64, 2, Tiles::T0::kStages, Tiles::T0::kSharedBytes,
     Tiles::T0::kThreads, Tiles::T0::kSharedBytes <= kWindowBytes},
    {"128x256x32", 128, 256, 32, 2, Tiles::T1::kStages, Tiles::T1::kSharedBytes,
     Tiles::T1::kThreads, Tiles::T1::kSharedBytes <= kWindowBytes},
    {"128x128x64", 128, 128, 64, 2, Tiles::T2::kStages, Tiles::T2::kSharedBytes,
     Tiles::T2::kThreads, Tiles::T2::kSharedBytes <= kWindowBytes},
    {"128x128x128", 128, 128, 128, 2, Tiles::T3::kStages,
     Tiles::T3::kSharedBytes, Tiles::T3::kThreads,
     Tiles::T3::kSharedBytes <= kWindowBytes},
    {"256x128x64", 256, 128, 64, 2, Tiles::T4::kStages, Tiles::T4::kSharedBytes,
     Tiles::T4::kThreads, Tiles::T4::kSharedBytes <= kWindowBytes},
};

const CaseReport kCases[] = {
    {"bf16 store, plain A, plain B", false, false, false,
     Cases::F0::kSharedBytes, Cases::F0::kThreads},
    {"bf16 store, transposed A, plain B", true, false, false,
     Cases::F1::kSharedBytes, Cases::F1::kThreads},
    {"bf16 store, plain A, transposed B", false, true, false,
     Cases::F2::kSharedBytes, Cases::F2::kThreads},
    {"bf16 store, transposed A, transposed B", true, true, false,
     Cases::F3::kSharedBytes, Cases::F3::kThreads},
    {"fp32 store, plain A, plain B", false, false, true,
     Cases::W0::kSharedBytes, Cases::W0::kThreads},
    {"fp32 store, transposed A, plain B", true, false, true,
     Cases::W1::kSharedBytes, Cases::W1::kThreads},
    {"fp32 store, plain A, transposed B", false, true, true,
     Cases::W2::kSharedBytes, Cases::W2::kThreads},
    {"fp32 store, transposed A, transposed B", true, true, true,
     Cases::W3::kSharedBytes, Cases::W3::kThreads},
};

//: The entry each row reads. Rows 0 and 5 name one type, so they share entry 0.
constexpr int kEntryOfRow[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11};

//: The kernel entry of each entry, in index order.
const void* const kEntries[] = {
#define NANOCHAT_PROBE_ENTRY_ADDRESS(index) CutlassProbeEntry##index(),
    NANOCHAT_FOR_EACH_PROBE_ENTRY(NANOCHAT_PROBE_ENTRY_ADDRESS)
#undef NANOCHAT_PROBE_ENTRY_ADDRESS
};

//: The length of one of the tables above.
template <class T, std::size_t kN>
constexpr int CountOf(const T (&)[kN]) {
  return static_cast<int>(kN);
}

//: Every row names an entry that exists.
constexpr bool EntriesInRange() {
  for (int entry : kEntryOfRow) {
    if (entry < 0 || entry >= kEntryCount) return false;
  }
  return true;
}

static_assert(CountOf(kTiles) == kTileRows, "the tile table is the wrong size");
static_assert(CountOf(kCases) == kCaseRows, "the case table is the wrong size");
static_assert(kTileRows + kCaseRows == kRowCount,
              "the row count is the tables");
static_assert(CountOf(kEntryOfRow) == kRowCount, "one entry for each row");
static_assert(CountOf(kEntries) == kEntryCount, "one address for each entry");
static_assert(EntriesInRange(), "a row names an entry that does not exist");

}  // namespace

int TileCount() { return CountOf(kTiles); }

const TileReport& Tile(int index) { return kTiles[index]; }

const void* TileEntry(int index) { return kEntries[kEntryOfRow[index]]; }

int LiveCases() { return CountOf(kCases); }

const CaseReport& Case(int index) { return kCases[index]; }

const void* CaseEntry(int index) {
  return kEntries[kEntryOfRow[kTileRows + index]];
}

}  // namespace cutlass_probe
}  // namespace nanochat
