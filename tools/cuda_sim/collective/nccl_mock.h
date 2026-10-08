#ifndef NANOCHAT_TOOLS_CUDA_SIM_COLLECTIVE_NCCL_MOCK_H_
#define NANOCHAT_TOOLS_CUDA_SIM_COLLECTIVE_NCCL_MOCK_H_

// The collective library mock for the S0 simulator (docs/simulator.md
// sections 6.3 and 6.4).
//
// The mock implements the NCCL subset the gradient-sync seam needs -- the
// communicator lifecycle and `ncclAllReduce` -- with **weak** definitions, so
// it links without a vendor library and a real NCCL can still override it.
//
// Every rank is a separate process. The mock rendezvouses through a POSIX
// shared-memory segment named by the communicator's unique id, so the ranks
// need no daemon and no network. The wait is bounded: a rank whose peers never
// arrive returns `ncclTimeout` instead of hanging, which is what turns the
// deadlock case in docs/simulator.md section 12.3 (invariant 12) into a test
// rather than a hang.
//
// The coordinator half merges the per-rank records and checks the cross-rank
// rules in docs/simulator.md section 6.4. It is deliberately data-in,
// verdict-out, so the same code judges a four-process run and a hand-built
// timeline in a unit test.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The NCCL surface (the subset the seam uses, plus the reserved names)
// ---------------------------------------------------------------------------

using ncclResult_t = int;
constexpr ncclResult_t ncclSuccess = 0;
constexpr ncclResult_t ncclUnhandledCudaError = 1;
constexpr ncclResult_t ncclSystemError = 2;
constexpr ncclResult_t ncclInternalError = 3;
constexpr ncclResult_t ncclInvalidArgument = 4;
constexpr ncclResult_t ncclInvalidUsage = 5;
constexpr ncclResult_t ncclRemoteError = 6;
constexpr ncclResult_t ncclInProgress = 7;
constexpr ncclResult_t ncclTimeout = 8;

struct ncclComm;
using ncclComm_t = ncclComm*;

// NCCL_UNIQUE_ID_BYTES.
constexpr int kNcclUniqueIdBytes = 128;
struct ncclUniqueId {
  char internal[kNcclUniqueIdBytes];
};

struct ncclDataType_t {
  int value;
};
struct ncclRedOp_t {
  int value;
};

// The data-type and reduction codes the seam can pass. The mock maps them to
// an element size so the buffer-range check is real.
int NcclDataTypeSize(ncclDataType_t type);
const char* NcclDataTypeName(ncclDataType_t type);
const char* NcclRedOpName(ncclRedOp_t op);

// ---------------------------------------------------------------------------
// The mock's own surface
// ---------------------------------------------------------------------------

namespace nanochat {
namespace sim {

// The profile the run simulates, from `NANOCHAT_SIM_PROFILE`. Empty when unset.
std::string SimProfile();

// The wall-clock bound a collective waits for its peers. From
// `NANOCHAT_SIM_TIMEOUT_MS`, default 2000.
int PeerTimeoutMs();

// One record of one collective call on one rank. The coordinator judges these.
struct CollectiveRecord {
  int rank = 0;
  int world_size = 1;
  std::string communicator;
  std::string call;  // "ncclCommInitRank", "ncclAllReduce", ...
  std::uint64_t count = 0;
  std::string data_type;
  std::string red_op;
  int root = 0;
  bool timed_out = false;
  // The `ncclGroupStart` nesting depth when the call was issued. 0 means the
  // call was not inside a group.
  int group_depth = 0;
  std::string verdict;  // "ok", "timeout", "error: ..."
};

// Merges the per-rank records and checks the cross-rank rules
// (docs/simulator.md section 6.4). Returns true and leaves `error` empty when
// the timeline is consistent; otherwise returns false and describes the first
// violation in `error`.
bool JudgeCollectiveRecords(const std::vector<CollectiveRecord>& records,
                            std::string* error);

// Appends one record to the run's collective log, when `NANOCHAT_SIM_LOG` is
// set. One JSON object per line; pointer values are never written.
void AppendCollectiveRecord(const CollectiveRecord& record);

// Reads a log written by `AppendCollectiveRecord` back into records. This is
// the merge half of the multi-rank model: each rank appends to the one shared
// log, and the coordinator reads it back. Returns false (and sets `error`)
// only when the file cannot be opened; a malformed line is skipped.
bool ReadCollectiveLog(const std::string& path,
                       std::vector<CollectiveRecord>* records,
                       std::string* error);

// The current `ncclGroupStart` nesting depth of this process. 0 when balanced.
// The mock refuses an unmatched `ncclGroupEnd`; this lets a test read the depth
// directly.
int GroupDepth();

}  // namespace sim
}  // namespace nanochat

// ---------------------------------------------------------------------------
// Weak definitions. A real NCCL, if linked, wins.
// ---------------------------------------------------------------------------

extern "C" {

ncclResult_t ncclGetVersion(int* version) __attribute__((weak));
const char* ncclGetErrorString(ncclResult_t result) __attribute__((weak));
ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id) __attribute__((weak));
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId unique_id, int rank)
    __attribute__((weak));
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist)
    __attribute__((weak));
ncclResult_t ncclCommDestroy(ncclComm_t comm) __attribute__((weak));
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count)
    __attribute__((weak));
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)
    __attribute__((weak));
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                           std::size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, void* stream)
    __attribute__((weak));
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                           std::size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, void* stream) __attribute__((weak));
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                           std::size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, void* stream) __attribute__((weak));
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               std::size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, void* stream)
    __attribute__((weak));
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, std::size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, void* stream) __attribute__((weak));
ncclResult_t ncclSend(const void* sendbuff, std::size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      void* stream) __attribute__((weak));
ncclResult_t ncclRecv(void* recvbuff, std::size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      void* stream) __attribute__((weak));
ncclResult_t ncclGroupStart(void) __attribute__((weak));
ncclResult_t ncclGroupEnd(void) __attribute__((weak));
}

#endif  // NANOCHAT_TOOLS_CUDA_SIM_COLLECTIVE_NCCL_MOCK_H_
