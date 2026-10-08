// The collective library mock (docs/simulator.md sections 6.3 and 6.4).
//
// Weak definitions of the NCCL subset the gradient-sync seam needs, plus the
// coordinator that judges the cross-rank timeline. See `nccl_mock.h` for the
// design and the bounded-wait rationale.

#include "tools/cuda_sim/collective/nccl_mock.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

// The `ncclGroupStart`/`ncclGroupEnd` nesting depth of this process, and the
// communicator the rank-less group calls are attributed to. A process in the
// seam has one communicator, so this is enough to record group records against
// the right rank for the coordinator.
int g_group_depth = 0;
ncclComm* g_current_comm = nullptr;

}  // namespace

namespace nanochat {
namespace sim {
namespace Internal {

constexpr int kMaxRanks = 16;

// One call slot per collective call; the ranks agree on the slot because each
// keeps its own call counter and they call in the same order. 32 slots is more
// than the seam ever issues in a run.
constexpr int kMaxCalls = 32;
constexpr std::size_t kMaxElements = 1u << 20;
// The largest elementwise payload the rendezvous segment can hold. The
// coordinator checks a record's `count * element_size` against it, so an
// oversized collective is reported as a verdict instead of a timeout.
constexpr std::uint64_t kMaxCollectiveBytes =
    static_cast<std::uint64_t>(kMaxElements) * sizeof(float);

struct CallSlot {
  std::atomic<std::uint32_t> ready;
  std::atomic<std::uint32_t> released;
  std::atomic<std::uint32_t> failed;
};

struct SharedRegion {
  std::atomic<std::uint32_t> magic;
  CallSlot calls[kMaxCalls];
  float payload[kMaxRanks][kMaxElements];
};

constexpr std::uint32_t kMagic = 0x4e43434cu;  // "NCCL"

std::string EnvOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? value : fallback;
}

bool DeadlinePassed(const std::chrono::steady_clock::time_point& deadline) {
  return std::chrono::steady_clock::now() >= deadline;
}

void WaitBriefly() {
  std::this_thread::sleep_for(std::chrono::microseconds(50));
}

// The element size of a data type named as the log spells it. 0 for an unknown
// name, which the coordinator reports.
int ElementSizeByName(const std::string& name) {
  if (name == "ncclInt8" || name == "ncclUint8") return 1;
  if (name == "ncclFloat16" || name == "ncclBfloat16") return 2;
  if (name == "ncclInt32" || name == "ncclUint32" || name == "ncclFloat32") {
    return 4;
  }
  if (name == "ncclInt64" || name == "ncclUint64" || name == "ncclFloat64") {
    return 8;
  }
  return 0;
}

}  // namespace Internal

std::string SimProfile() { return Internal::EnvOr("NANOCHAT_SIM_PROFILE", ""); }

int PeerTimeoutMs() {
  const std::string value = Internal::EnvOr("NANOCHAT_SIM_TIMEOUT_MS", "2000");
  const int parsed = std::atoi(value.c_str());
  return parsed > 0 ? parsed : 2000;
}

int GroupDepth() { return g_group_depth; }

int NcclDataTypeSize(ncclDataType_t type) {
  switch (type.value) {
    case 0:  // ncclInt8
    case 1:  // ncclUint8
      return 1;
    case 2:  // ncclInt32
    case 3:  // ncclUint32
    case 6:  // ncclFloat16
      return type.value == 6 ? 2 : 4;
    case 4:  // ncclInt64
    case 5:  // ncclUint64
    case 7:  // ncclFloat32
    case 8:  // ncclFloat64
      return (type.value == 7) ? 4 : 8;
    case 9:  // ncclBfloat16
      return 2;
    default:
      return 0;
  }
}

const char* NcclDataTypeName(ncclDataType_t type) {
  switch (type.value) {
    case 0:
      return "ncclInt8";
    case 2:
      return "ncclInt32";
    case 4:
      return "ncclInt64";
    case 6:
      return "ncclFloat16";
    case 7:
      return "ncclFloat32";
    case 8:
      return "ncclFloat64";
    case 9:
      return "ncclBfloat16";
    default:
      return "ncclUnknown";
  }
}

const char* NcclRedOpName(ncclRedOp_t op) {
  switch (op.value) {
    case 0:
      return "ncclSum";
    case 1:
      return "ncclProd";
    case 2:
      return "ncclMax";
    case 3:
      return "ncclMin";
    case 4:
      return "ncclAvg";
    default:
      return "ncclUnknown";
  }
}

void AppendCollectiveRecord(const CollectiveRecord& record) {
  const char* path = std::getenv("NANOCHAT_SIM_LOG");
  if (path == nullptr || *path == '\0') return;
  std::FILE* file = std::fopen(path, "a");
  if (file == nullptr) return;
  std::fprintf(file,
               "{\"kind\":\"collective\",\"rank\":%d,\"world_size\":%d,"
               "\"comm\":\"%s\",\"call\":\"%s\",\"count\":%llu,"
               "\"data_type\":\"%s\",\"red_op\":\"%s\",\"root\":%d,"
               "\"timed_out\":%s,\"group_depth\":%d,\"verdict\":\"%s\"}\n",
               record.rank, record.world_size, record.communicator.c_str(),
               record.call.c_str(),
               static_cast<unsigned long long>(record.count),
               record.data_type.c_str(), record.red_op.c_str(), record.root,
               record.timed_out ? "true" : "false", record.group_depth,
               record.verdict.c_str());
  std::fclose(file);
}

bool JudgeCollectiveRecords(const std::vector<CollectiveRecord>& records,
                            std::string* error) {
  auto fail = [error](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };

  if (records.empty()) return fail("the timeline holds no records");

  // World-size consistency, validated before any per-rank indexing so a
  // record cannot index past the `issued` vector.
  const int world = records.front().world_size;
  for (const CollectiveRecord& record : records) {
    if (record.world_size != world) {
      return fail("world_size differs across records (" +
                  std::to_string(record.world_size) + " vs " +
                  std::to_string(world) + ")");
    }
    if (record.rank < 0 || record.rank >= world) {
      return fail("rank " + std::to_string(record.rank) +
                  " is outside world_size " + std::to_string(world));
    }
  }

  // Group nesting: `ncclGroupStart`/`ncclGroupEnd` must balance on every rank.
  std::vector<int> depth(static_cast<std::size_t>(world), 0);
  for (const CollectiveRecord& record : records) {
    const std::size_t rank = static_cast<std::size_t>(record.rank);
    if (record.call == "ncclGroupStart") {
      depth[rank] += 1;
    } else if (record.call == "ncclGroupEnd") {
      depth[rank] -= 1;
      if (depth[rank] < 0) {
        return fail("rank " + std::to_string(record.rank) +
                    " closes a group it never opened");
      }
    }
  }
  for (int rank = 0; rank < world; ++rank) {
    if (depth[static_cast<std::size_t>(rank)] != 0) {
      return fail("rank " + std::to_string(rank) +
                  " leaves an ncclGroupStart unclosed");
    }
  }

  // Communicator identity: every rank names the same communicators.
  std::string communicator;
  bool have_communicator = false;
  for (const CollectiveRecord& record : records) {
    if (record.call != "ncclAllReduce") continue;
    if (!have_communicator) {
      communicator = record.communicator;
      have_communicator = true;
    } else if (record.communicator != communicator) {
      return fail("communicator identity differs across ranks");
    }
  }

  // Collective matching: the count, type, op, and root agree across the ranks
  // that issue the all-reduce, the buffer range fits, and every rank issues it.
  std::vector<int> issued(static_cast<std::size_t>(world), 0);
  std::string data_type;
  std::string red_op;
  std::uint64_t count = 0;
  bool have_signature = false;
  for (const CollectiveRecord& record : records) {
    if (record.call != "ncclAllReduce") continue;
    if (record.timed_out) {
      return fail("rank " + std::to_string(record.rank) +
                  " timed out waiting for a peer (deadlock)");
    }
    const int element_bytes = Internal::ElementSizeByName(record.data_type);
    if (element_bytes == 0) {
      return fail("unknown data type '" + record.data_type + "'");
    }
    const std::uint64_t bytes =
        record.count * static_cast<std::uint64_t>(element_bytes);
    if (bytes > Internal::kMaxCollectiveBytes) {
      return fail("rank " + std::to_string(record.rank) +
                  " count exceeds the collective buffer");
    }
    if (!have_signature) {
      data_type = record.data_type;
      red_op = record.red_op;
      count = record.count;
      have_signature = true;
    } else if (record.data_type != data_type || record.red_op != red_op ||
               record.count != count || record.root != 0) {
      return fail("collective signature differs across ranks");
    }
    issued[static_cast<std::size_t>(record.rank)] += 1;
  }
  if (have_signature) {
    for (int rank = 0; rank < world; ++rank) {
      if (issued[static_cast<std::size_t>(rank)] == 0) {
        return fail("rank " + std::to_string(rank) +
                    " never issued the all-reduce (deadlock)");
      }
    }
  }
  if (error != nullptr) error->clear();
  return true;
}

namespace {

// Reads the integer value that follows `"key":` in one JSON line. Returns
// false when the key is absent.
bool JsonInt(const std::string& line, const std::string& key, long long* out) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t at = line.find(needle);
  if (at == std::string::npos) return false;
  const char* begin = line.c_str() + at + needle.size();
  char* end = nullptr;
  const long long value = std::strtoll(begin, &end, 10);
  if (end == begin) return false;
  *out = value;
  return true;
}

// Reads the string value that follows `"key":"` in one JSON line.
bool JsonString(const std::string& line, const std::string& key,
                std::string* out) {
  const std::string needle = "\"" + key + "\":\"";
  const std::size_t at = line.find(needle);
  if (at == std::string::npos) return false;
  const std::size_t begin = at + needle.size();
  const std::size_t end = line.find('"', begin);
  if (end == std::string::npos) return false;
  *out = line.substr(begin, end - begin);
  return true;
}

bool JsonBool(const std::string& line, const std::string& key, bool* out) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t at = line.find(needle);
  if (at == std::string::npos) return false;
  *out = line.compare(at + needle.size(), 4, "true") == 0;
  return true;
}

}  // namespace

bool ReadCollectiveLog(const std::string& path,
                       std::vector<CollectiveRecord>* records,
                       std::string* error) {
  if (records == nullptr) {
    if (error != nullptr) *error = "no output vector";
    return false;
  }
  std::FILE* file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    if (error != nullptr) *error = "cannot open " + path;
    return false;
  }
  char buffer[2048];
  while (std::fgets(buffer, sizeof(buffer), file) != nullptr) {
    const std::string line(buffer);
    if (line.find("\"kind\":\"collective\"") == std::string::npos) continue;
    CollectiveRecord record;
    long long value = 0;
    if (JsonInt(line, "rank", &value)) record.rank = static_cast<int>(value);
    if (JsonInt(line, "world_size", &value)) {
      record.world_size = static_cast<int>(value);
    }
    JsonString(line, "comm", &record.communicator);
    JsonString(line, "call", &record.call);
    if (JsonInt(line, "count", &value)) {
      record.count = static_cast<std::uint64_t>(value);
    }
    JsonString(line, "data_type", &record.data_type);
    JsonString(line, "red_op", &record.red_op);
    if (JsonInt(line, "root", &value)) record.root = static_cast<int>(value);
    JsonBool(line, "timed_out", &record.timed_out);
    if (JsonInt(line, "group_depth", &value)) {
      record.group_depth = static_cast<int>(value);
    }
    JsonString(line, "verdict", &record.verdict);
    records->push_back(std::move(record));
  }
  std::fclose(file);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace sim
}  // namespace nanochat

// ---------------------------------------------------------------------------
// The weak NCCL surface
// ---------------------------------------------------------------------------

struct ncclComm {
  int rank = 0;
  int world_size = 1;
  int device = 0;
  bool destroyed = false;
  std::uint64_t calls = 0;
  std::string id;
  void* base = nullptr;
  std::size_t bytes = 0;
};

namespace {

nanochat::sim::Internal::SharedRegion* RegionOf(ncclComm_t comm) {
  return static_cast<nanochat::sim::Internal::SharedRegion*>(comm->base);
}

// Opens (creating if needed) the rendezvous segment for `id`.
bool AttachRegion(ncclComm_t comm) {
  const std::size_t bytes = sizeof(nanochat::sim::Internal::SharedRegion);
  const std::string name = "/nanochat_sim_" + comm->id;
  const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) return false;
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    close(fd);
    return false;
  }
  void* base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) return false;
  auto* region = static_cast<nanochat::sim::Internal::SharedRegion*>(base);
  if (region->magic.load() != nanochat::sim::Internal::kMagic) {
    std::memset(base, 0, bytes);
    region->magic.store(nanochat::sim::Internal::kMagic);
  }
  comm->base = base;
  comm->bytes = bytes;
  return true;
}

// The element-wise sum of the ranks' payloads, plus the bounded wait. Returns
// false when a peer never arrives.
bool AllReduceRound(ncclComm_t comm, const void* sendbuff, void* recvbuff,
                    std::size_t count, int element_bytes) {
  nanochat::sim::Internal::SharedRegion* region = RegionOf(comm);
  if (comm->calls >= nanochat::sim::Internal::kMaxCalls) return false;
  if (count > nanochat::sim::Internal::kMaxElements) return false;
  nanochat::sim::Internal::CallSlot& slot =
      region->calls[static_cast<std::size_t>(comm->calls)];
  ++comm->calls;

  const std::size_t bytes = count * static_cast<std::size_t>(element_bytes);
  float* mine = region->payload[static_cast<std::size_t>(comm->rank)];
  std::memcpy(mine, sendbuff, bytes);

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(nanochat::sim::PeerTimeoutMs());
  slot.ready.fetch_add(1);
  while (slot.ready.load() != static_cast<std::uint32_t>(comm->world_size)) {
    if (nanochat::sim::Internal::DeadlinePassed(deadline)) {
      slot.failed.store(1);
      return false;
    }
    nanochat::sim::Internal::WaitBriefly();
  }

  for (std::size_t i = 0; i < count; ++i) {
    float sum = 0.0f;
    for (int rank = 0; rank < comm->world_size; ++rank) {
      sum += region->payload[static_cast<std::size_t>(rank)][i];
    }
    static_cast<float*>(recvbuff)[i] = sum;
  }

  // Let the peers finish reading before the segment is reused.
  slot.released.fetch_add(1);
  while (slot.released.load() != static_cast<std::uint32_t>(comm->world_size)) {
    if (nanochat::sim::Internal::DeadlinePassed(deadline)) {
      slot.failed.store(1);
      return false;
    }
    nanochat::sim::Internal::WaitBriefly();
  }
  return true;
}

// Records one call and appends it to the run log.
void Record(ncclComm_t comm, const char* call, std::size_t count,
            ncclDataType_t data_type, ncclRedOp_t red_op, bool ok) {
  nanochat::sim::CollectiveRecord record;
  record.rank = comm->rank;
  record.world_size = comm->world_size;
  record.communicator = comm->id;
  record.call = call;
  record.count = count;
  record.data_type = nanochat::sim::NcclDataTypeName(data_type);
  record.red_op = nanochat::sim::NcclRedOpName(red_op);
  record.root = 0;  // ncclAllReduce has no root
  record.timed_out = !ok;
  record.group_depth = g_group_depth;
  record.verdict = ok ? "ok" : "timeout";
  nanochat::sim::AppendCollectiveRecord(record);
}

// Records a group call against the current communicator. The group calls take
// no communicator, so the rank is whatever `ncclCommInitRank` stored last.
void RecordGroup(const char* call, const char* verdict) {
  if (g_current_comm == nullptr || g_current_comm->destroyed) return;
  nanochat::sim::CollectiveRecord record;
  record.rank = g_current_comm->rank;
  record.world_size = g_current_comm->world_size;
  record.communicator = g_current_comm->id;
  record.call = call;
  record.group_depth = g_group_depth;
  record.verdict = verdict;
  nanochat::sim::AppendCollectiveRecord(record);
}

}  // namespace

extern "C" {

ncclResult_t ncclGetVersion(int* version) {
  if (version == nullptr) return ncclInvalidArgument;
  *version = 21000;  // 2.10.0, the version the mock pretends to be
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclSystemError:
      return "system error";
    case ncclInvalidArgument:
      return "invalid argument";
    case ncclInvalidUsage:
      return "invalid usage";
    case ncclTimeout:
      return "peer timeout (simulated deadlock)";
    default:
      return "unknown error";
  }
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id) {
  if (unique_id == nullptr) return ncclInvalidArgument;
  const char* from_env = std::getenv("NANOCHAT_SIM_COMM_ID");
  const std::string seed =
      (from_env != nullptr && *from_env != '\0') ? from_env : "nanochat-sim";
  std::snprintf(unique_id->internal, kNcclUniqueIdBytes, "%s", seed.c_str());
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId unique_id, int rank) {
  if (comm == nullptr || nranks <= 0 || rank < 0 || rank >= nranks) {
    return ncclInvalidArgument;
  }
  if (nranks > nanochat::sim::Internal::kMaxRanks) return ncclInvalidArgument;
  auto* handle = new (std::nothrow) ncclComm();
  if (handle == nullptr) return ncclSystemError;
  handle->rank = rank;
  handle->world_size = nranks;
  handle->id.assign(unique_id.internal,
                    strnlen(unique_id.internal, kNcclUniqueIdBytes));
  if (!AttachRegion(handle)) {
    delete handle;
    return ncclSystemError;
  }
  *comm = handle;
  g_current_comm = handle;
  nanochat::sim::CollectiveRecord record;
  record.rank = rank;
  record.world_size = nranks;
  record.communicator = handle->id;
  record.call = "ncclCommInitRank";
  record.verdict = "ok";
  nanochat::sim::AppendCollectiveRecord(record);
  return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  if (comm == nullptr || ndev <= 0) return ncclInvalidArgument;
  ncclUniqueId unique_id;
  ncclGetUniqueId(&unique_id);
  for (int device = 0; device < ndev; ++device) {
    ncclResult_t status =
        ncclCommInitRank(&comm[device], ndev, unique_id,
                         devlist != nullptr ? devlist[device] : device);
    if (status != ncclSuccess) return status;
  }
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->destroyed) return ncclInvalidUsage;
  comm->destroyed = true;
  if (comm->base != nullptr) {
    munmap(comm->base, comm->bytes);
    comm->base = nullptr;
  }
  if (g_current_comm == comm) g_current_comm = nullptr;
  // The handle is intentionally retained rather than freed, so a
  // use-after-destroy is reported by the `destroyed` guard instead of reading
  // freed memory. A test creates a handful of communicators per process.
  return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) return ncclInvalidArgument;
  if (comm->destroyed) return ncclInvalidUsage;
  *count = comm->world_size;
  return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return ncclInvalidArgument;
  if (comm->destroyed) return ncclInvalidUsage;
  *rank = comm->rank;
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                           std::size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, void* stream) {
  (void)stream;
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) {
    return ncclInvalidArgument;
  }
  if (comm->destroyed) return ncclInvalidUsage;
  const int element_bytes = nanochat::sim::NcclDataTypeSize(datatype);
  if (element_bytes == 0) return ncclInvalidArgument;
  if (count == 0) return ncclSuccess;
  // The seam only ever sums fp32 contiguous buffers.
  if (datatype.value != 7) return ncclInvalidUsage;
  if (op.value != 0) return ncclInvalidUsage;

  const bool ok =
      AllReduceRound(comm, sendbuff, recvbuff, count, element_bytes);
  Record(comm, "ncclAllReduce", count, datatype, op, ok);
  return ok ? ncclSuccess : ncclTimeout;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                           std::size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, void* stream) {
  (void)count;
  (void)datatype;
  (void)root;
  (void)comm;
  (void)stream;
  // Reserved for later; the seam has no broadcast today.
  std::memcpy(recvbuff, sendbuff, 0);
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                           std::size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, void* stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)sendcount;
  (void)datatype;
  (void)comm;
  (void)stream;
  return ncclInternalError;  // reserved
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               std::size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, void* stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)recvcount;
  (void)datatype;
  (void)op;
  (void)comm;
  (void)stream;
  return ncclInternalError;  // reserved
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, std::size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, void* stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)op;
  (void)root;
  (void)comm;
  (void)stream;
  return ncclInternalError;  // reserved
}

ncclResult_t ncclSend(const void* sendbuff, std::size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      void* stream) {
  (void)sendbuff;
  (void)count;
  (void)datatype;
  (void)peer;
  (void)comm;
  (void)stream;
  return ncclInternalError;  // reserved
}

ncclResult_t ncclRecv(void* recvbuff, std::size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      void* stream) {
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)peer;
  (void)comm;
  (void)stream;
  return ncclInternalError;  // reserved
}

ncclResult_t ncclGroupStart(void) {
  ++g_group_depth;
  RecordGroup("ncclGroupStart", "ok");
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
  if (g_group_depth <= 0) {
    RecordGroup("ncclGroupEnd", "error: unbalanced ncclGroupEnd");
    return ncclInvalidUsage;
  }
  --g_group_depth;
  RecordGroup("ncclGroupEnd", "ok");
  return ncclSuccess;
}

}  // extern "C"
