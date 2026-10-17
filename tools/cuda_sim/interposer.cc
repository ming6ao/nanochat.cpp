// The API interposer (docs/simulator.md section 5).
//
// A shared library that exports the CUDA runtime, the cuBLAS entry points, and
// the fat-binary hooks. `tools/nanochat simulate --suite api` preloads it with
// `LD_PRELOAD`, so `backends/cuda/device.cu`, `backends/cuda/gemm.cu`, and the
// compiler-generated launch stubs bind to it instead of to the toolkit. Host
// memory backs `cudaMalloc`; the API calls are recorded and validated against
// the simulated device profile.
//
// The interposer does not compute. Each call is recorded as one JSON object
// holding the call name, the numeric arguments, the verdict, and the profile.
// Pointer *values* are never written: the test asserts named invariants over
// the log (docs/simulator.md section 12.2), not byte equality with a committed
// file, and a log without addresses stays comparable across runs.
//
// Two contracts are load-bearing, because the callers rely on them:
//   * `cublasCreate_v2` writes a non-null handle. `gemm.cu` creates the handle
//     only when the stored one is null, so a null handle would recreate it on
//     every call.
//   * `__cudaRegisterFatBinary` returns a non-null token. The registration
//     hooks pass it back.

#include "tools/cuda_sim/cuda_sim_abi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nanochat/device_profile.h"

namespace {

// ---------------------------------------------------------------------------
// The recorder
// ---------------------------------------------------------------------------

// The state the interposer keeps for one cublasLt matmul descriptor and one
// matrix layout. The opaque handle is the map key, so the descriptors of one
// call are checked against each other.
struct LtDescriptor {
  cublasComputeType_t compute_type = 0;
  cudaDataType_t scale_type = 0;
  cublasOperation_t trans_a = 0;
  cublasOperation_t trans_b = 0;
  const float* a_scale = nullptr;
  const float* b_scale = nullptr;
  bool fast_accum = false;
};

struct LtLayout {
  cudaDataType_t type = 0;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::int64_t ld = 0;
};

struct InterposerState {
  std::mutex mutex;
  std::string log;
  int records = 0;
  bool canary_written = false;
  // Live `cudaMalloc` allocations, so a copy or a free can be range-checked.
  std::map<void*, std::size_t> allocations;
  // The largest dynamic shared-memory size each function was granted.
  std::map<const void*, std::size_t> shared_attributes;
  // The live cuBLAS handle, and whether `cublasCreate_v2` ran.
  cublasHandle_t cublas_handle = nullptr;
  bool cublas_created = false;
  bool cublas_destroyed = false;
  // The live cublasLt handle and the descriptors created against it.
  cublasLtHandle_t lt_handle = nullptr;
  std::map<void*, LtDescriptor> lt_descriptors;
  std::map<void*, LtLayout> lt_layouts;
  std::map<void*, bool> lt_preferences;
  // The fat-binary token the loader passes back.
  void* fat_binary_token = nullptr;
};

InterposerState& State() {
  static InterposerState* state = new InterposerState();
  return *state;
}

std::string EnvOr(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? value : "";
}

std::string ProfileName() { return EnvOr("NANOCHAT_SIM_PROFILE"); }

const nanochat::DeviceProfile* Profile() {
  static const std::string name = ProfileName();
  if (name.empty()) return nullptr;
  static const nanochat::DeviceProfile* profile =
      nanochat::FindDeviceProfile(name);
  return profile;
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Appends one record. The first record is the binding canary: it names the
// simulated profile, which is the proof that the mock ran at all. The caller
// must not hold the state lock.
void Record(const std::string& call, const std::string& arguments,
            const std::string& verdict) {
  InterposerState& state = State();
  std::string batch;
  bool first = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.canary_written) {
      state.canary_written = true;
      first = true;
      batch += "{\"kind\":\"canary\",\"profile\":\"" +
               JsonEscape(ProfileName()) + "\"}\n";
      ++state.records;
    }
    batch += "{\"kind\":\"api\",\"call\":\"" + JsonEscape(call) +
             "\",\"args\":{" + arguments + "},\"verdict\":\"" +
             JsonEscape(verdict) + "\",\"profile\":\"" +
             JsonEscape(ProfileName()) + "\"}\n";
    ++state.records;
    state.log += batch;
  }

  const std::string path = EnvOr("NANOCHAT_SIM_LOG");
  if (!path.empty()) {
    // The very first record (the canary) truncates the log, so a stale file
    // from a previous run cannot precede it. Later records append. This matters
    // here because the real CUDA runtime's initializer makes the first call
    // before `main`, so a test cannot truncate the file itself without
    // discarding the canary.
    std::FILE* file = std::fopen(path.c_str(), first ? "w" : "a");
    if (file != nullptr) {
      std::fputs(batch.c_str(), file);
      std::fclose(file);
    }
  }
}

// True when the range [pointer, pointer + bytes) lies inside a live
// allocation. `pointer` is compared, never printed.
bool InLiveAllocation(void* pointer, std::size_t bytes) {
  InterposerState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto* begin = static_cast<const char*>(pointer);
  const auto* end = begin + bytes;
  for (const auto& entry : state.allocations) {
    const auto* first = static_cast<const char*>(entry.first);
    const auto* last = first + entry.second;
    if (begin >= first && end <= last) return true;
  }
  return false;
}

int MaxThreadsPerBlock() {
  const nanochat::DeviceProfile* profile = Profile();
  return profile != nullptr ? profile->max_threads_per_block : 1024;
}

std::size_t MaxSharedPerBlock() {
  const nanochat::DeviceProfile* profile = Profile();
  return profile != nullptr ? profile->max_shared_memory_per_block : 48 * 1024;
}

}  // namespace

// ---------------------------------------------------------------------------
// The control surface
// ---------------------------------------------------------------------------

extern "C" {

void nanochat_sim_reset(void) {
  InterposerState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.log.clear();
  state.records = 0;
  state.canary_written = false;
  state.allocations.clear();
  state.shared_attributes.clear();
  state.cublas_handle = nullptr;
  state.cublas_created = false;
  state.cublas_destroyed = false;
  state.lt_handle = nullptr;
  state.lt_descriptors.clear();
  state.lt_layouts.clear();
  state.lt_preferences.clear();
  state.fat_binary_token = nullptr;
}

const char* nanochat_sim_log(void) {
  InterposerState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  static std::string* snapshot = new std::string();
  *snapshot = state.log;
  return snapshot->c_str();
}

int nanochat_sim_record_count(void) {
  InterposerState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.records;
}

const char* nanochat_sim_profile(void) {
  static const std::string* name = new std::string(ProfileName());
  return name->c_str();
}

// ---------------------------------------------------------------------------
// The CUDA runtime
// ---------------------------------------------------------------------------

cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
  if (pointer == nullptr) return kCudaErrorInvalidValue;
  void* memory = std::malloc(bytes > 0 ? bytes : 1);
  if (memory == nullptr) return kCudaErrorInvalidValue;
  {
    InterposerState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.allocations[memory] = bytes;
  }
  *pointer = memory;
  Record("cudaMalloc", "\"bytes\":" + std::to_string(bytes), "ok");
  return kCudaSuccess;
}

cudaError_t cudaFree(void* pointer) {
  if (pointer == nullptr) {
    Record("cudaFree", "\"null\":true", "ok");
    return kCudaSuccess;
  }
  InterposerState& state = State();
  bool tracked = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.allocations.find(pointer);
    tracked = found != state.allocations.end();
    if (tracked) state.allocations.erase(found);
  }
  if (!tracked) {
    Record("cudaFree", "\"tracked\":false",
           "error: pointer is not a live allocation");
    return kCudaErrorInvalidDevicePointer;
  }
  std::free(pointer);
  Record("cudaFree", "\"tracked\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                       cudaMemcpyKindAbi kind) {
  const bool dst_known = InLiveAllocation(dst, bytes);
  const bool src_known = InLiveAllocation(const_cast<void*>(src), bytes);
  // The direction names which side is the device buffer: the device end must
  // be inside a live allocation; the host end need not be.
  bool ok = true;
  switch (kind) {
    case kCudaMemcpyHostToDevice:
      ok = dst_known;
      break;
    case kCudaMemcpyDeviceToHost:
      ok = src_known;
      break;
    case kCudaMemcpyDeviceToDevice:
      ok = dst_known && src_known;
      break;
    default:
      ok = true;  // a host-to-host copy carries no device pointer
      break;
  }
  if (bytes > 0 && dst != nullptr && src != nullptr) {
    std::memcpy(dst, src, bytes);
  }
  Record("cudaMemcpy",
         "\"bytes\":" + std::to_string(bytes) +
             ",\"kind\":" + std::to_string(kind) +
             ",\"dst_device\":" + (dst_known ? "true" : "false") +
             ",\"src_device\":" + (src_known ? "true" : "false"),
         ok ? "ok" : "error: copy range is outside a live allocation");
  return ok ? kCudaSuccess : kCudaErrorInvalidDevicePointer;
}

cudaError_t cudaMemset(void* pointer, int value, std::size_t bytes) {
  if (bytes == 0) {
    Record("cudaMemset", "\"bytes\":0", "ok");
    return kCudaSuccess;
  }
  // The bounds check is load-bearing: a memset outside a live allocation is the
  // same defect as a bad copy, so it must be reported, not silently discarded.
  if (!InLiveAllocation(pointer, bytes)) {
    Record("cudaMemset", "\"bytes\":" + std::to_string(bytes),
           "error: memset range is outside a live allocation");
    return kCudaErrorInvalidDevicePointer;
  }
  std::memset(pointer, value, bytes);
  Record("cudaMemset", "\"bytes\":" + std::to_string(bytes), "ok");
  return kCudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
  if (device != nullptr) *device = 0;
  Record("cudaGetDevice", "", "ok");
  return kCudaSuccess;
}

// The simulator models one device. `backends/cuda/device.cu` enumerates the
// local devices with this call, so the count must be positive, or the device
// list stays empty.
cudaError_t cudaGetDeviceCount(int* count) {
  if (count == nullptr) {
    Record("cudaGetDeviceCount", "", "error: null count pointer");
    return kCudaErrorInvalidValue;
  }
  *count = 1;
  Record("cudaGetDeviceCount", "\"count\":1", "ok");
  return kCudaSuccess;
}

// Only device 0 exists in the simulation. A bind to another ordinal fails, so a
// wrong device index cannot pass silently.
cudaError_t cudaSetDevice(int device) {
  const bool ok = device == 0;
  Record("cudaSetDevice", "\"device\":" + std::to_string(device),
         ok ? "ok" : "error: the simulator has one device (ordinal 0)");
  return ok ? kCudaSuccess : kCudaErrorInvalidDevice;
}

cudaError_t cudaGetDeviceProperties_v2(cudaDevicePropPrefix* prop,
                                       int /*device*/) {
  if (prop == nullptr) return kCudaErrorInvalidValue;
  std::memset(prop, 0, sizeof(*prop));
  const nanochat::DeviceProfile* profile = Profile();
  if (profile != nullptr) {
    std::snprintf(prop->name, sizeof(prop->name), "%s", profile->name);
    prop->totalGlobalMem = profile->total_memory_bytes;
    prop->sharedMemPerBlock = 48 * 1024;
    prop->warpSize = profile->warp_size;
    prop->maxThreadsPerBlock = profile->max_threads_per_block;
    prop->maxThreadsDim[0] = profile->max_threads_per_block;
    prop->maxThreadsDim[1] = profile->max_threads_per_block;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->major = profile->compute_major;
    prop->minor = profile->compute_minor;
    Record("cudaGetDeviceProperties_v2",
           "\"major\":" + std::to_string(prop->major) +
               ",\"minor\":" + std::to_string(prop->minor),
           "ok");
  } else {
    Record("cudaGetDeviceProperties_v2", "", "ok: no simulated profile");
  }
  return kCudaSuccess;
}

const char* cudaGetErrorString(cudaError_t status) {
  switch (status) {
    case kCudaSuccess:
      return "no error";
    case kCudaErrorInvalidValue:
      return "invalid argument";
    case kCudaErrorInvalidDevicePointer:
      return "invalid device pointer";
    case kCudaErrorInvalidConfiguration:
      return "invalid configuration argument";
    case kCudaErrorInvalidDevice:
      return "invalid device ordinal";
    default:
      return "unknown error";
  }
}

cudaError_t cudaGetLastError(void) {
  Record("cudaGetLastError", "", "ok");
  return kCudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  Record("cudaDeviceSynchronize", "", "ok");
  return kCudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block,
                             void** /*args*/, std::size_t shared,
                             void* stream) {
  const std::size_t threads =
      static_cast<std::size_t>(block.x) * block.y * block.z;
  if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
    Record("cudaLaunchKernel", "\"grid\":0", "error: a grid dimension is zero");
    return kCudaErrorInvalidConfiguration;
  }
  if (threads == 0) {
    Record("cudaLaunchKernel", "\"block_threads\":0",
           "error: a block dimension is zero");
    return kCudaErrorInvalidConfiguration;
  }
  if (threads > static_cast<std::size_t>(MaxThreadsPerBlock())) {
    Record("cudaLaunchKernel", "\"block_threads\":" + std::to_string(threads),
           "error: block size exceeds the profile limit");
    return kCudaErrorInvalidConfiguration;
  }
  std::size_t granted = kDefaultSharedMemoryLimit;
  {
    InterposerState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto found = state.shared_attributes.find(func);
    if (found != state.shared_attributes.end()) granted = found->second;
  }
  if (shared > granted) {
    Record("cudaLaunchKernel",
           "\"shared_bytes\":" + std::to_string(shared) +
               ",\"granted_bytes\":" + std::to_string(granted),
           "error: dynamic shared memory above the granted limit needs "
           "cudaFuncSetAttribute");
    return kCudaErrorInvalidConfiguration;
  }
  (void)stream;
  Record("cudaLaunchKernel",
         "\"grid_x\":" + std::to_string(grid.x) +
             ",\"block_threads\":" + std::to_string(threads) +
             ",\"shared_bytes\":" + std::to_string(shared),
         "ok");
  return kCudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void* func, int attribute, int value) {
  if (attribute != kCudaFuncAttributeMaxDynamicSharedMemorySize) {
    Record("cudaFuncSetAttribute", "\"attribute\":" + std::to_string(attribute),
           "ok: attribute not modelled");
    return kCudaSuccess;
  }
  if (value < 0 || static_cast<std::size_t>(value) > MaxSharedPerBlock()) {
    Record("cudaFuncSetAttribute", "\"value\":" + std::to_string(value),
           "error: value exceeds the profile limit");
    return kCudaErrorInvalidValue;
  }
  {
    InterposerState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.shared_attributes[func] = static_cast<std::size_t>(value);
  }
  Record("cudaFuncSetAttribute",
         "\"value\":" + std::to_string(value) + ",\"legacy\":false", "ok");
  return kCudaSuccess;
}

void** __cudaRegisterFatBinary(void* fat_cubin) {
  (void)fat_cubin;
  InterposerState& state = State();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.fat_binary_token == nullptr) {
      // Non-null: the loader passes the token back to the other hooks.
      state.fat_binary_token = static_cast<void*>(&state);
    }
  }
  Record("__cudaRegisterFatBinary", "", "ok");
  return static_cast<void**>(&state.fat_binary_token);
}

void __cudaRegisterFatBinaryEnd(void** handle) {
  (void)handle;
  Record("__cudaRegisterFatBinaryEnd", "", "ok");
}

void __cudaUnregisterFatBinary(void** handle) {
  (void)handle;
  Record("__cudaUnregisterFatBinary", "", "ok");
}

void __cudaRegisterFunction(void** handle, const void* host_function,
                            const char* device_function,
                            const char* device_name, int thread_limit,
                            void* tid, void* bid, void* block_dim,
                            void* grid_dim, int* warp_size) {
  (void)handle;
  (void)device_function;
  (void)tid;
  (void)bid;
  (void)block_dim;
  (void)grid_dim;
  (void)warp_size;
  // The name is kept, the host address is not: a log without addresses stays
  // comparable across runs (docs/simulator.md section 12.2).
  Record("__cudaRegisterFunction",
         "\"name\":\"" + JsonEscape(device_name != nullptr ? device_name : "") +
             "\",\"thread_limit\":" + std::to_string(thread_limit) +
             ",\"host_null\":" + (host_function == nullptr ? "true" : "false"),
         "ok");
}

// ---------------------------------------------------------------------------
// The reserved stream, event, and graph calls
// ---------------------------------------------------------------------------
//
// Recorded and accepted. The production backend uses none of them today; they
// are here so the symbol gate can prove the mock covers the family.

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  if (stream != nullptr) *stream = reinterpret_cast<cudaStream_t>(1);
  Record("cudaStreamCreate", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t /*stream*/) {
  Record("cudaStreamDestroy", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t /*stream*/) {
  Record("cudaStreamSynchronize", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t /*stream*/, int /*mode*/) {
  Record("cudaStreamBeginCapture", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t /*stream*/, cudaGraph_t* graph) {
  if (graph != nullptr) *graph = reinterpret_cast<cudaGraph_t>(1);
  Record("cudaStreamEndCapture", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
  if (event != nullptr) *event = reinterpret_cast<cudaEvent_t>(1);
  Record("cudaEventCreate", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t /*event*/) {
  Record("cudaEventDestroy", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t /*event*/, cudaStream_t /*stream*/) {
  Record("cudaEventRecord", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t /*event*/) {
  Record("cudaEventSynchronize", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t /*start*/,
                                 cudaEvent_t /*end*/) {
  if (milliseconds != nullptr) *milliseconds = 0.0f;
  Record("cudaEventElapsedTime", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t /*graph*/,
                                 void* /*a*/, void* /*b*/, std::size_t /*c*/) {
  if (exec != nullptr) *exec = reinterpret_cast<cudaGraphExec_t>(1);
  Record("cudaGraphInstantiate", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t /*exec*/, cudaStream_t /*stream*/) {
  Record("cudaGraphLaunch", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t /*graph*/) {
  Record("cudaGraphDestroy", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t /*exec*/) {
  Record("cudaGraphExecDestroy", "\"reserved\":true", "ok");
  return kCudaSuccess;
}

// ---------------------------------------------------------------------------
// cuBLAS
// ---------------------------------------------------------------------------

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (handle == nullptr) return kCublasStatusInvalidValue;
  InterposerState& state = State();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    // Non-null: `gemm.cu` creates the handle only when the stored one is null.
    state.cublas_handle = static_cast<void*>(&state);
    state.cublas_created = true;
    state.cublas_destroyed = false;
  }
  *handle = state.cublas_handle;
  Record("cublasCreate_v2", "\"handle_non_null\":true", "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  InterposerState& state = State();
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    known = handle != nullptr && handle == state.cublas_handle;
    if (known) {
      state.cublas_handle = nullptr;
      state.cublas_destroyed = true;
    }
  }
  Record("cublasDestroy", "", known ? "ok" : "error: unknown handle");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t /*stream*/) {
  InterposerState& state = State();
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    known = handle != nullptr && handle == state.cublas_handle;
  }
  Record("cublasSetStream", "", known ? "ok" : "error: unknown handle");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

const char* cublasGetStatusString(cublasStatus_t status) {
  switch (status) {
    case kCublasStatusSuccess:
      return "CUBLAS_STATUS_SUCCESS";
    case kCublasStatusNotInitialized:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case kCublasStatusInvalidValue:
      return "CUBLAS_STATUS_INVALID_VALUE";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

// The classic single GEMM. Recorded; the leading-dimension rule is checked.
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* a, int lda,
                              const float* b, int ldb, const float* beta,
                              float* c, int ldc) {
  (void)alpha;
  (void)beta;
  (void)a;
  (void)b;
  (void)c;
  InterposerState& state = State();
  bool handle_ok = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    handle_ok = handle != nullptr && handle == state.cublas_handle;
  }
  if (!handle_ok) {
    Record("cublasSgemm_v2", "", "error: cuBLAS call before cublasCreate_v2");
    return kCublasStatusNotInitialized;
  }
  const int rows_a = transa == kCublasOpN ? m : k;
  const int rows_b = transb == kCublasOpN ? k : n;
  const bool dimensions_ok = lda >= rows_a && ldb >= rows_b && ldc >= m;
  Record(
      "cublasSgemm_v2",
      "\"m\":" + std::to_string(m) + ",\"n\":" + std::to_string(n) +
          ",\"k\":" + std::to_string(k) + ",\"lda\":" + std::to_string(lda) +
          ",\"ldb\":" + std::to_string(ldb) +
          ",\"ldc\":" + std::to_string(ldc) + ",\"legacy\":true",
      dimensions_ok ? "ok" : "error: leading dimension below the operand row");
  return dimensions_ok ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

// The shared body of the two extended-GEMM entry points. The CUDA API defines
// `cublasGemmEx` as the strided-batched call restricted to one batch and zero
// stride, and `gemm.cu` reaches it for a single-batch GEMM whose operands are
// not all fp32 (the bf16 mixed path, docs/precision.md). Both entry points
// therefore run the same rules and record the same shape; only the recorded
// name and the batch count differ.
static cublasStatus_t GemmExCommon(
    const char* call, cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc,
    int batch_count, cublasComputeType_t compute_type, cublasGemmAlgo_t algo) {
  InterposerState& state = State();
  bool handle_ok = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    handle_ok = handle != nullptr && handle == state.cublas_handle;
  }
  if (!handle_ok) {
    Record(call, "",
           "error: cuBLAS call before cublasCreate_v2 (or after "
           "cublasDestroy)");
    return kCublasStatusNotInitialized;
  }

  const nanochat::DeviceProfile* profile = Profile();
  const bool tensor_cores = profile != nullptr && profile->has_tensor_cores;
  const bool tensor_op_algo = algo == kCublasGemmDefaultTensorOp;
  const bool legacy = tensor_op_algo || algo != kCublasGemmDefault;
  // The legacy-enum rule: `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and
  // `cublasGemmAlgo_t` are deprecated since CUDA 11.0, and a GEMM on a
  // tensor-core device must accumulate in fp32 (docs/simulator.md section
  // 5.3). On a tensor-core profile the only accepted accumulation is
  // `CUBLAS_COMPUTE_32F`; `CUBLAS_COMPUTE_16F` is the Pascal non-tensor-core
  // path and must not appear here.
  bool ok = true;
  std::string verdict = "ok";
  if (tensor_cores && compute_type != kCublasCompute32F) {
    ok = false;
    verdict =
        "error: a tensor-core device must accumulate with "
        "CUBLAS_COMPUTE_32F";
  }
  if (legacy) {
    verdict +=
        "; CUBLAS_GEMM_DEFAULT_TENSOR_OP and cublasGemmAlgo_t are "
        "legacy since CUDA 11.0";
  }
  // The leading-dimension rule follows the transpose flags exactly as the
  // classic entry point does: op(A) is m x k and op(B) is k x n, so a stored
  // operand's row count depends on its flag.
  const int rows_a = transa == kCublasOpN ? m : k;
  const int rows_b = transb == kCublasOpN ? k : n;
  if (!(lda >= rows_a && ldb >= rows_b && ldc >= m)) {
    ok = false;
    verdict = "error: leading dimension below the operand row";
  }
  Record(call,
         "\"m\":" + std::to_string(m) + ",\"n\":" + std::to_string(n) +
             ",\"k\":" + std::to_string(k) +
             ",\"batch_count\":" + std::to_string(batch_count) +
             ",\"compute_type\":" + std::to_string(compute_type) +
             ",\"algo\":" + std::to_string(algo) +
             ",\"tensor_cores\":" + (tensor_cores ? "true" : "false") +
             ",\"legacy_enum\":" + (legacy ? "true" : "false"),
         verdict);
  return ok ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* a,
    cublasDataType_t a_type, int lda, long long stride_a, const void* b,
    cublasDataType_t b_type, int ldb, long long stride_b, const void* beta,
    void* c, cublasDataType_t c_type, int ldc, long long stride_c,
    int batch_count, cublasComputeType_t compute_type, cublasGemmAlgo_t algo) {
  (void)alpha;
  (void)beta;
  (void)a;
  (void)b;
  (void)c;
  (void)a_type;
  (void)b_type;
  (void)c_type;
  (void)stride_a;
  (void)stride_b;
  (void)stride_c;
  return GemmExCommon("cublasGemmStridedBatchedEx", handle, transa, transb, m,
                      n, k, lda, ldb, ldc, batch_count, compute_type, algo);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* a,
                            cublasDataType_t a_type, int lda, const void* b,
                            cublasDataType_t b_type, int ldb, const void* beta,
                            void* c, cublasDataType_t c_type, int ldc,
                            cublasComputeType_t compute_type,
                            cublasGemmAlgo_t algo) {
  (void)alpha;
  (void)beta;
  (void)a;
  (void)b;
  (void)c;
  (void)a_type;
  (void)b_type;
  (void)c_type;
  return GemmExCommon("cublasGemmEx", handle, transa, transb, m, n, k, lda, ldb,
                      ldc, 1, compute_type, algo);
}

// ---------------------------------------------------------------------------
// cublasLt (docs/simulator-precision.md section 5)
// ---------------------------------------------------------------------------

namespace {

// True when the layout type is one of the two fp8 operand formats.
bool IsFp8Type(cudaDataType_t type) {
  return type == kCudaR8FE4M3 || type == kCudaR8FE5M2;
}

}  // namespace

cublasStatus_t cublasLtCreate(cublasLtHandle_t* light_handle) {
  if (light_handle == nullptr) return kCublasStatusInvalidValue;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    State().lt_handle = static_cast<void*>(&State());
  }
  *light_handle = State().lt_handle;
  Record("cublasLtCreate", "\"handle_non_null\":true", "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t light_handle) {
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = light_handle != nullptr && light_handle == State().lt_handle;
    if (known) State().lt_handle = nullptr;
  }
  Record("cublasLtDestroy", "", known ? "ok" : "error: unknown handle");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

const char* cublasLtGetStatusString(cublasStatus_t status) {
  return cublasGetStatusString(status);
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmul_desc,
                                        cublasComputeType_t compute_type,
                                        cudaDataType_t scale_type) {
  if (matmul_desc == nullptr) return kCublasStatusInvalidValue;
  void* token = new int(0);
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    LtDescriptor& desc = State().lt_descriptors[token];
    desc.compute_type = compute_type;
    desc.scale_type = scale_type;
  }
  *matmul_desc = token;
  Record("cublasLtMatmulDescCreate",
         "\"compute_type\":" + std::to_string(compute_type) +
             ",\"scale_type\":" + std::to_string(scale_type),
         "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmul_desc) {
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = State().lt_descriptors.erase(matmul_desc) > 0;
  }
  if (known) delete static_cast<int*>(matmul_desc);
  Record("cublasLtMatmulDescDestroy", "",
         known ? "ok" : "error: unknown descriptor");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmul_desc, cublasLtMatmulDescAttributes_t attribute,
    const void* buffer, std::size_t size_in_bytes) {
  InterposerState& state = State();
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    known = state.lt_descriptors.count(matmul_desc) != 0;
  }
  if (!known) {
    Record("cublasLtMatmulDescSetAttribute", "",
           "error: attribute set on an unknown descriptor");
    return kCublasStatusInvalidValue;
  }
  std::string verdict = "ok";
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    LtDescriptor& desc = state.lt_descriptors[matmul_desc];
    if (buffer == nullptr || size_in_bytes == 0) {
      verdict = "error: null or empty attribute buffer";
    } else if (attribute == kCublasLtDescTransA) {
      std::memcpy(&desc.trans_a, buffer, sizeof(desc.trans_a));
    } else if (attribute == kCublasLtDescTransB) {
      std::memcpy(&desc.trans_b, buffer, sizeof(desc.trans_b));
    } else if (attribute == kCublasLtDescAScalePointer) {
      std::memcpy(&desc.a_scale, buffer, sizeof(desc.a_scale));
    } else if (attribute == kCublasLtDescBScalePointer) {
      std::memcpy(&desc.b_scale, buffer, sizeof(desc.b_scale));
    } else if (attribute == kCublasLtDescFastAccum) {
      std::int32_t fast = 0;
      std::memcpy(&fast, buffer, sizeof(fast));
      desc.fast_accum = fast != 0;
    }
  }
  Record("cublasLtMatmulDescSetAttribute",
         "\"attribute\":" + std::to_string(attribute), verdict);
  return verdict == "ok" ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(
    cublasLtMatmulDesc_t matmul_desc, cublasLtMatmulDescAttributes_t attribute,
    void* buffer, std::size_t size_in_bytes, std::size_t* size_written) {
  (void)buffer;
  (void)size_in_bytes;
  if (size_written != nullptr) *size_written = 0;
  InterposerState& state = State();
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    known = state.lt_descriptors.count(matmul_desc) != 0;
  }
  Record("cublasLtMatmulDescGetAttribute",
         "\"attribute\":" + std::to_string(attribute),
         known ? "ok" : "error: unknown descriptor");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* layout,
                                          cudaDataType_t type,
                                          std::uint64_t rows,
                                          std::uint64_t cols, std::int64_t ld) {
  if (layout == nullptr) return kCublasStatusInvalidValue;
  void* token = new int(0);
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    LtLayout& entry = State().lt_layouts[token];
    entry.type = type;
    entry.rows = rows;
    entry.cols = cols;
    entry.ld = ld;
  }
  *layout = token;
  Record(
      "cublasLtMatrixLayoutCreate",
      "\"type\":" + std::to_string(type) + ",\"rows\":" + std::to_string(rows) +
          ",\"cols\":" + std::to_string(cols) + ",\"ld\":" + std::to_string(ld),
      "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout) {
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = State().lt_layouts.erase(layout) > 0;
  }
  if (known) delete static_cast<int*>(layout);
  Record("cublasLtMatrixLayoutDestroy", "",
         known ? "ok" : "error: unknown layout");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout,
                                                int attribute,
                                                const void* buffer,
                                                std::size_t size_in_bytes) {
  (void)buffer;
  (void)size_in_bytes;
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = State().lt_layouts.count(layout) != 0;
  }
  Record("cublasLtMatrixLayoutSetAttribute",
         "\"attribute\":" + std::to_string(attribute),
         known ? "ok" : "error: unknown layout");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(
    cublasLtMatmulPreference_t* preference) {
  if (preference == nullptr) return kCublasStatusInvalidValue;
  void* token = new int(0);
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    State().lt_preferences[token] = true;
  }
  *preference = token;
  Record("cublasLtMatmulPreferenceCreate", "", "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(
    cublasLtMatmulPreference_t preference) {
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = State().lt_preferences.erase(preference) > 0;
  }
  if (known) delete static_cast<int*>(preference);
  Record("cublasLtMatmulPreferenceDestroy", "",
         known ? "ok" : "error: unknown preference");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t preference,
    cublasLtMatmulPreferenceAttributes_t attribute, const void* buffer,
    std::size_t size_in_bytes) {
  (void)buffer;
  (void)size_in_bytes;
  bool known = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    known = State().lt_preferences.count(preference) != 0;
  }
  Record("cublasLtMatmulPreferenceSetAttribute",
         "\"attribute\":" + std::to_string(attribute),
         known ? "ok" : "error: unknown preference");
  return known ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t light_handle, cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc, cublasLtMatrixLayout_t d_desc,
    cublasLtMatmulPreference_t preference, int requested_algo_count,
    cublasLtMatmulHeuristicResult_t heuristic_results[],
    int* return_algo_count) {
  (void)operation_desc;
  (void)a_desc;
  (void)b_desc;
  (void)c_desc;
  (void)d_desc;
  (void)preference;
  if (return_algo_count != nullptr) *return_algo_count = 0;
  bool handle_ok = false;
  {
    std::lock_guard<std::mutex> lock(State().mutex);
    handle_ok = light_handle != nullptr && light_handle == State().lt_handle;
  }
  if (!handle_ok) {
    Record("cublasLtMatmulAlgoGetHeuristic", "",
           "error: cublasLt call before cublasLtCreate");
    return kCublasStatusNotInitialized;
  }
  if (heuristic_results == nullptr || requested_algo_count < 1) {
    Record("cublasLtMatmulAlgoGetHeuristic", "", "error: no result slot");
    return kCublasStatusInvalidValue;
  }
  // The mock computes nothing, so a zeroed algorithm and no workspace are
  // enough: the choice cannot change a result.
  heuristic_results[0] = cublasLtMatmulHeuristicResult_t{};
  heuristic_results[0].state = kCublasStatusSuccess;
  heuristic_results[0].workspaceSize = 0;
  if (return_algo_count != nullptr) *return_algo_count = 1;
  Record("cublasLtMatmulAlgoGetHeuristic",
         "\"requested\":" + std::to_string(requested_algo_count), "ok");
  return kCublasStatusSuccess;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t light_handle,
                              cublasLtMatmulDesc_t compute_desc,
                              const void* alpha, const void* a,
                              cublasLtMatrixLayout_t a_desc, const void* b,
                              cublasLtMatrixLayout_t b_desc, const void* beta,
                              const void* c, cublasLtMatrixLayout_t c_desc,
                              void* d, cublasLtMatrixLayout_t d_desc,
                              const cublasLtMatmulAlgo_t* algo, void* workspace,
                              std::size_t workspace_size, cudaStream_t stream) {
  (void)alpha;
  (void)a;
  (void)b;
  (void)beta;
  (void)c;
  (void)d;
  (void)d_desc;
  (void)algo;
  (void)workspace;
  (void)workspace_size;
  (void)stream;
  InterposerState& state = State();
  LtDescriptor desc;
  LtLayout a_layout;
  LtLayout b_layout;
  LtLayout c_layout;
  bool ok = true;
  std::string verdict = "ok";
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const bool handle_ok =
        light_handle != nullptr && light_handle == state.lt_handle;
    const auto desc_it = state.lt_descriptors.find(compute_desc);
    const auto a_it = state.lt_layouts.find(a_desc);
    const auto b_it = state.lt_layouts.find(b_desc);
    const auto c_it = state.lt_layouts.find(c_desc);
    if (!handle_ok) {
      ok = false;
      verdict = "error: cublasLt call before cublasLtCreate";
    } else if (desc_it == state.lt_descriptors.end() ||
               a_it == state.lt_layouts.end() ||
               b_it == state.lt_layouts.end() ||
               c_it == state.lt_layouts.end()) {
      ok = false;
      verdict = "error: unknown descriptor";
    } else {
      desc = desc_it->second;
      a_layout = a_it->second;
      b_layout = b_it->second;
      c_layout = c_it->second;
    }
  }
  const bool fp8 = IsFp8Type(a_layout.type) || IsFp8Type(b_layout.type);
  if (ok && fp8) {
    const nanochat::DeviceProfile* profile = Profile();
    if (profile != nullptr && !profile->supports_fp8) {
      ok = false;
      verdict = "error: fp8 call on a profile without fp8 support";
    } else if (desc.compute_type != kCublasCompute32F) {
      ok = false;
      verdict = "error: fp8 call must accumulate with CUBLAS_COMPUTE_32F";
    } else if (c_layout.type != kCudaR32F) {
      ok = false;
      verdict = "error: fp8 call must write an fp32 accumulator";
    } else if (desc.a_scale == nullptr || desc.b_scale == nullptr) {
      ok = false;
      verdict = "error: fp8 call is missing a scale pointer";
    } else if (!(std::isfinite(*desc.a_scale) && *desc.a_scale > 0.0f &&
                 std::isfinite(*desc.b_scale) && *desc.b_scale > 0.0f)) {
      ok = false;
      verdict = "error: fp8 scale is not finite and positive";
    }
  }
  Record("cublasLtMatmul",
         "\"a_type\":" + std::to_string(a_layout.type) +
             ",\"b_type\":" + std::to_string(b_layout.type) +
             ",\"c_type\":" + std::to_string(c_layout.type) +
             ",\"compute_type\":" + std::to_string(desc.compute_type) +
             ",\"fp8\":" + (fp8 ? "true" : "false"),
         verdict);
  return ok ? kCublasStatusSuccess : kCublasStatusInvalidValue;
}

}  // extern "C"
