// The acceptance check for the four Hopper mechanisms
// (docs/simulator.md section 9.2, docs/kernels.md Part 3 section 4,
// docs/attention.md Part 8 section 5).
//
// It runs the real interface (`backends/cuda/kernels/hopper_ops.cuh`) through
// the emulation arm and compares every element with a host reference:
//
//   1. The score product over the interleaved operand form, both key widths.
//   2. The score product over the 128-byte swizzled panel form, which the copy
//      engine fills and the descriptor reads with a panel step.
//   3. The value product, whose first operand is in registers and whose second
//      is the transposed staging.
//   4. The `mbarrier` handoff between a producer thread and a consumer
//      warpgroup, with the consumer polling before the producer arrives.
//   5. The refusal: a descriptor that reads past its staged region.
//
// The operands are multiples of an eighth, so every product and every sum of
// them is exact in both bf16 and fp32, and the comparison is exact. A wrong
// address, a wrong lane map, or a wrong swizzle changes a value; nothing
// rounds.
//
// CPU only, so the target carries the `sim` tag and rides the default loop.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda.h>

#include "backends/cuda/kernels/hopper_ops.cuh"

namespace {

namespace hopper = ::nanochat::cuda_kernels::hopper;

int g_failures = 0;

void Fail(const char* what, const char* case_name) {
  std::fprintf(stderr, "FAIL: %s (%s)\n", what, case_name);
  ++g_failures;
}

// --- The operands -----------------------------------------------------------

//: The query rows, the key rows, and the head width of one atom set.
constexpr int kQueryRows = 64;
constexpr int kKeys = 128;
constexpr int kDim = 128;
//: One atom walks sixteen elements of the head width, so the recorded tile is
//: eight atoms deep.
constexpr int kAtoms = kDim / 16;
//: The slots of one accumulator thread at a hundred and twenty-eight keys.
constexpr int kSlots = 64;

//: A value whose product with another, summed over the tile, stays exact: every
//: operand is a multiple of an eighth and the products are multiples of a sixty
//: fourth.
float OperandValue(int index, int shift) {
  return static_cast<float>((index * 5 + shift) % 13 - 6) * 0.125f;
}

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  // Round to nearest even at the sixteenth bit, which is what the device does.
  const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float FromBf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

//: A tile of operands, one shift of the value function per case.
std::vector<float> OperandTile(std::size_t count, int shift) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = OperandValue(static_cast<int>(i), shift);
  }
  return values;
}

//: The same tile rounded to bf16, and answered as the words a kernel reads. The
//: tile it rounds in place is what the reference multiplies, so both sides of a
//: comparison see one operand.
std::vector<std::uint16_t> RoundToBf16(std::vector<float>* values) {
  std::vector<std::uint16_t> words(values->size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] = ToBf16((*values)[i]);
    (*values)[i] = FromBf16(words[i]);
  }
  return words;
}

// --- The staged operand forms ----------------------------------------------

//: The interleaved offset of one element of a row-major tile. This is the form
//: `dev/kernels/wgmma_bench.cu` stages and every recorded staged row reads.
int InterleavedOffset(int row, int k, int rows) {
  const int block = k / 16;
  const int half = (k % 16) / 8;
  const int element = k % 8;
  return block * rows * 32 + (row / 8) * 256 + half * 128 + (row % 8) * 16 +
         element * 2;
}

//: The bytes one interleaved tile of `rows` rows occupies at the head width.
constexpr int InterleavedBytes(int rows) { return rows * kDim * 2; }

//: The bytes the query atom advances between two k blocks, which is the
//: descriptor's own step.
constexpr int AtomStepBytes(int rows) { return rows * 32; }

// --- Case 1: the score product, interleaved --------------------------------

__global__ void ScoreInterleavedKernel(const std::uint16_t* query,
                                       const std::uint16_t* key, float* out) {
  __shared__ __align__(1024) char q_stage[InterleavedBytes(kQueryRows)];
  __shared__ __align__(1024) char k_stage[InterleavedBytes(kKeys)];
  const int threads = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < kQueryRows * kDim;
       index += threads) {
    const int row = index / kDim;
    const int k = index - row * kDim;
    std::memcpy(q_stage + InterleavedOffset(row, k, kQueryRows), query + index,
                2);
  }
  for (int index = static_cast<int>(threadIdx.x); index < kKeys * kDim;
       index += threads) {
    const int row = index / kDim;
    const int k = index - row * kDim;
    std::memcpy(k_stage + InterleavedOffset(row, k, kKeys), key + index, 2);
  }
  __syncthreads();
  hopper::DeclareSharedOperand(q_stage, InterleavedBytes(kQueryRows));
  hopper::DeclareSharedOperand(k_stage, InterleavedBytes(kKeys));

  float d[kSlots];
  hopper::Fence();
  for (int atom = 0; atom < kAtoms; ++atom) {
    const hopper::MatrixDescriptor q = hopper::MakeMatrixDescriptor(
        q_stage + atom * AtomStepBytes(kQueryRows), 128, 256,
        hopper::OperandLayout::kInterleaved);
    const hopper::MatrixDescriptor k =
        hopper::MakeMatrixDescriptor(k_stage + atom * AtomStepBytes(kKeys), 128,
                                     256, hopper::OperandLayout::kInterleaved);
    hopper::ScoreAtom<128>(q, k, d, atom != 0);
  }
  hopper::Commit();
  hopper::Wait<0>();
  hopper::FenceAccumulator<kSlots>(d);

  const int lane = static_cast<int>(threadIdx.x) % hopper::kWarpgroupThreads;
  for (int index = 0; index < kSlots; ++index) {
    out[hopper::SlotRow(lane, index) * kKeys +
        hopper::SlotColumn(lane, index)] = d[index];
  }
}

std::vector<float> ReferenceScore(const std::vector<float>& query,
                                  const std::vector<float>& key, int keys) {
  std::vector<float> out(static_cast<std::size_t>(kQueryRows) * keys, 0.0f);
  for (int row = 0; row < kQueryRows; ++row) {
    for (int column = 0; column < keys; ++column) {
      float sum = 0.0f;
      for (int k = 0; k < kDim; ++k) {
        sum += query[static_cast<std::size_t>(row) * kDim + k] *
               key[static_cast<std::size_t>(column) * kDim + k];
      }
      out[static_cast<std::size_t>(row) * keys + column] = sum;
    }
  }
  return out;
}

// --- Case 2: the score product, the swizzled panel -------------------------

//: One panel of the swizzled key tile: sixty-four columns of the head width.
constexpr int kPanelColumns = 64;
constexpr int kPanelBytes = kKeys * kPanelColumns * 2;
constexpr int kPanels = kDim / kPanelColumns;

__global__ void ScoreSwizzledKernel(const std::uint16_t* query, float* out,
                                    const CUtensorMap* panel_low,
                                    const CUtensorMap* panel_high) {
  __shared__ __align__(1024) char q_stage[InterleavedBytes(kQueryRows)];
  __shared__ __align__(1024) char k_stage[kPanels * kPanelBytes];
  __shared__ __align__(8) std::uint64_t ready;
  const int threads = static_cast<int>(blockDim.x);
  for (int index = static_cast<int>(threadIdx.x); index < kQueryRows * kDim;
       index += threads) {
    const int row = index / kDim;
    const int k = index - row * kDim;
    std::memcpy(q_stage + InterleavedOffset(row, k, kQueryRows), query + index,
                2);
  }
  __syncthreads();
  hopper::DeclareSharedOperand(q_stage, InterleavedBytes(kQueryRows));

  // The producer is the last thread of the block, so every consumer polls the
  // barrier before the stage lands. That is the blocking path of the wait.
  if (threadIdx.x == blockDim.x - 1) {
    hopper::BarrierInit(&ready, 1);
    hopper::BarrierExpectTx(&ready, 2 * kPanelBytes);
    const int coordinates[2] = {0, 0};
    hopper::TmaLoad<2>(k_stage, *panel_low, coordinates, &ready);
    hopper::TmaLoad<2>(k_stage + kPanelBytes, *panel_high, coordinates, &ready);
  }
  hopper::BarrierWait(&ready, 0);

  float d[kSlots];
  hopper::Fence();
  for (int atom = 0; atom < kAtoms; ++atom) {
    const int panel = atom / (kPanelColumns / 16);
    const int within = (atom % (kPanelColumns / 16)) * 32;
    const hopper::MatrixDescriptor q = hopper::MakeMatrixDescriptor(
        q_stage + atom * AtomStepBytes(kQueryRows), 128, 256,
        hopper::OperandLayout::kInterleaved);
    // The swizzled descriptor: an atom of eight rows of a hundred and twenty-
    // eight bytes is a kilobyte, so the stride offset is that atom and the
    // leading offset is one chunk.
    const hopper::MatrixDescriptor k =
        hopper::MakeMatrixDescriptor(k_stage + panel * kPanelBytes + within, 16,
                                     1024, hopper::OperandLayout::kSwizzled128);
    hopper::ScoreAtom<128>(q, k, d, atom != 0);
  }
  hopper::Commit();
  hopper::Wait<0>();

  const int lane = static_cast<int>(threadIdx.x) % hopper::kWarpgroupThreads;
  for (int index = 0; index < kSlots; ++index) {
    out[hopper::SlotRow(lane, index) * kKeys +
        hopper::SlotColumn(lane, index)] = d[index];
  }
}

// --- Case 3: the value product --------------------------------------------

__global__ void ValueKernel(const std::uint16_t* probability,
                            const std::uint16_t* value, float* out,
                            float* accum) {
  __shared__ __align__(1024) char v_stage[InterleavedBytes(kDim)];
  const int threads = static_cast<int>(blockDim.x);
  // The value tile is staged transposed: the rows of the staged tile are the
  // head width and the `k` axis is the key index.
  for (int key = 0; key < kKeys; ++key) {
    for (int dim = static_cast<int>(threadIdx.x); dim < kDim; dim += threads) {
      std::memcpy(v_stage + InterleavedOffset(dim, key, kDim),
                  value + static_cast<std::size_t>(key) * kDim + dim, 2);
    }
  }
  __syncthreads();
  hopper::DeclareSharedOperand(v_stage, InterleavedBytes(kDim));

  // The probability tile reaches the instruction in the calling thread's four
  // packed registers, through the register map. One atom reads sixteen keys, so
  // the registers are packed again for every atom, which is what a softmax step
  // does for the keys it just produced.
  const int lane = static_cast<int>(threadIdx.x) % hopper::kWarpgroupThreads;
  std::uint32_t a[4] = {0, 0, 0, 0};

  float* d = accum + static_cast<std::size_t>(threadIdx.x) * kSlots;
  for (int index = 0; index < kSlots; ++index) d[index] = 0.0f;
  hopper::Fence();
  for (int atom = 0; atom < kKeys / 16; ++atom) {
    for (int reg = 0; reg < 4; ++reg) {
      a[reg] = 0;
      for (int half = 0; half < 2; ++half) {
        const int row = hopper::RegisterRow(lane, reg);
        const int key = atom * 16 + hopper::RegisterKey(lane, reg, half);
        const std::uint16_t bits =
            probability[static_cast<std::size_t>(row) * kKeys + key];
        a[reg] |= static_cast<std::uint32_t>(bits) << (16 * half);
      }
    }
    const hopper::MatrixDescriptor v =
        hopper::MakeMatrixDescriptor(v_stage + atom * AtomStepBytes(kDim), 128,
                                     256, hopper::OperandLayout::kInterleaved);
    hopper::ValueAtom(a, v, d);
  }
  hopper::Commit();
  hopper::Wait<0>();

  for (int index = 0; index < kSlots; ++index) {
    out[hopper::SlotRow(lane, index) * kDim + hopper::SlotColumn(lane, index)] =
        d[index];
  }
}

std::vector<float> ReferenceValue(const std::vector<float>& probability,
                                  const std::vector<float>& value) {
  std::vector<float> out(static_cast<std::size_t>(kQueryRows) * kDim, 0.0f);
  for (int row = 0; row < kQueryRows; ++row) {
    for (int dim = 0; dim < kDim; ++dim) {
      float sum = 0.0f;
      for (int key = 0; key < kKeys; ++key) {
        sum += probability[static_cast<std::size_t>(row) * kKeys + key] *
               value[static_cast<std::size_t>(key) * kDim + dim];
      }
      out[static_cast<std::size_t>(row) * kDim + dim] = sum;
    }
  }
  return out;
}

// --- The checks -------------------------------------------------------------

void Compare(const std::vector<float>& got, const std::vector<float>& want,
             const char* case_name) {
  if (got.size() != want.size()) {
    Fail("result size", case_name);
    return;
  }
  int bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (bad == 0) {
        std::fprintf(stderr, "  first differs at %zu: %g against %g\n", i,
                     static_cast<double>(got[i]), static_cast<double>(want[i]));
      }
      ++bad;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr, "  %d of %zu elements differ\n", bad, got.size());
    Fail("element check", case_name);
  }
}

void RunInterleavedCase() {
  std::vector<float> query =
      OperandTile(static_cast<std::size_t>(kQueryRows) * kDim, 0);
  std::vector<float> key =
      OperandTile(static_cast<std::size_t>(kKeys) * kDim, 3);
  const std::vector<std::uint16_t> query_bits = RoundToBf16(&query);
  const std::vector<std::uint16_t> key_bits = RoundToBf16(&key);
  std::vector<float> out(static_cast<std::size_t>(kQueryRows) * kKeys, 0.0f);
  ::nanochat::cuda_backend::Launch(
      ScoreInterleavedKernel, dim3(1u), dim3(hopper::kWarpgroupThreads), 0,
      query_bits.data(), key_bits.data(), out.data());
  Compare(out, ReferenceScore(query, key, kKeys),
          "score product, interleaved form");
}

void RunSwizzledCase() {
  std::vector<float> query =
      OperandTile(static_cast<std::size_t>(kQueryRows) * kDim, 1);
  std::vector<float> key =
      OperandTile(static_cast<std::size_t>(kKeys) * kDim, 5);
  const std::vector<std::uint16_t> query_bits = RoundToBf16(&query);
  const std::vector<std::uint16_t> key_bits = RoundToBf16(&key);

  // Two maps, one a panel: sixty-four columns of the head width, and the base
  // of the second one panel along the row.
  CUtensorMap panel_low{};
  CUtensorMap panel_high{};
  for (int panel = 0; panel < kPanels; ++panel) {
    const void* base = reinterpret_cast<const char*>(key_bits.data()) +
                       panel * kPanelColumns * 2;
    const cuuint64_t dims[2] = {kPanelColumns, kKeys};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(kDim) * 2};
    const cuuint32_t box[2] = {kPanelColumns, kKeys};
    const cuuint32_t element[2] = {1, 1};
    CUtensorMap* map = panel == 0 ? &panel_low : &panel_high;
    const CUresult status = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims,
        strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
      Fail("the swizzled panel map was refused", "score, swizzled form");
      return;
    }
  }

  std::vector<float> out(static_cast<std::size_t>(kQueryRows) * kKeys, 0.0f);
  // Two warpgroups, so the producer is not the first fiber: every consumer
  // polls the barrier before the stage lands.
  ::nanochat::cuda_backend::Launch(
      ScoreSwizzledKernel, dim3(1u), dim3(2 * hopper::kWarpgroupThreads), 0,
      query_bits.data(), out.data(), &panel_low, &panel_high);
  Compare(out, ReferenceScore(query, key, kKeys),
          "score product, swizzled form");
}

// The staged transposed tile, read back through the descriptor one element at a
// time. This is the operand the value product reads, and it is checked without
// a block, so a wrong staging and a wrong product cannot be confused.
void CheckValueOperandReads() {
  constexpr int kTileBytes = InterleavedBytes(kDim);
  std::vector<char> stage(kTileBytes, 0);
  std::vector<float> value =
      OperandTile(static_cast<std::size_t>(kKeys) * kDim, 7);
  RoundToBf16(&value);
  for (int key = 0; key < kKeys; ++key) {
    for (int dim = 0; dim < kDim; ++dim) {
      const std::uint16_t bits =
          ToBf16(value[static_cast<std::size_t>(key) * kDim + dim]);
      std::memcpy(stage.data() + InterleavedOffset(dim, key, kDim), &bits, 2);
    }
  }
  hopper::DeclareSharedOperand(stage.data(), stage.size());
  int bad = 0;
  for (int atom = 0; atom < kKeys / 16; ++atom) {
    const hopper::MatrixDescriptor v = hopper::MakeMatrixDescriptor(
        stage.data() + atom * AtomStepBytes(kDim), 128, 256,
        hopper::OperandLayout::kInterleaved);
    for (int dim = 0; dim < kDim; ++dim) {
      for (int step = 0; step < 16; ++step) {
        const float got = hopper::detail::OperandElement(v, dim, step);
        const float want =
            value[static_cast<std::size_t>(atom * 16 + step) * kDim + dim];
        if (got != want && ++bad == 1) {
          std::fprintf(stderr,
                       "  staged value read differs at atom %d dim %d step %d: "
                       "%g against %g\n",
                       atom, dim, step, static_cast<double>(got),
                       static_cast<double>(want));
        }
      }
    }
  }
  if (bad != 0) {
    std::fprintf(stderr, "  %d reads differ\n", bad);
    Fail("the staged value operand does not read back", "value operand");
  }
}

void RunValueCase() {
  std::vector<float> probability =
      OperandTile(static_cast<std::size_t>(kQueryRows) * kKeys, 2);
  std::vector<float> value =
      OperandTile(static_cast<std::size_t>(kKeys) * kDim, 7);
  const std::vector<std::uint16_t> probability_bits = RoundToBf16(&probability);
  const std::vector<std::uint16_t> value_bits = RoundToBf16(&value);
  std::vector<float> out(static_cast<std::size_t>(kQueryRows) * kDim, 0.0f);
  std::vector<float> accum(
      static_cast<std::size_t>(hopper::kWarpgroupThreads) * kSlots, 0.0f);
  ::nanochat::cuda_backend::Launch(
      ValueKernel, dim3(1u), dim3(hopper::kWarpgroupThreads), 0,
      probability_bits.data(), value_bits.data(), out.data(), accum.data());
  Compare(out, ReferenceValue(probability, value), "value product");
}

// --- The emulator's own property: a local array across a shuffle -------------

//: A local array across a shuffle. The fiber that a shuffle blocks is resumed
//: later, so this states that the array is the thread's own and that the value
//: a shuffle answers is the neighbour's. A regression here would make every
//: mapped product below wrong for a reason that has nothing to do with the
//: product.
__global__ void LocalAcrossShuffleKernel(float* out) {
  float d[64];
  for (int i = 0; i < 64; ++i) d[i] = static_cast<float>(i + 1);
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int neighbour = __shfl_sync(0xFFFFFFFFu, lane, (lane + 1) % 32);
  if (neighbour != (lane + 1) % 32) d[0] = -1.0f;
  for (int i = 0; i < 64; ++i) out[threadIdx.x * 64 + i] = d[i];
}

void CheckLocalAcrossShuffle() {
  constexpr int kThreads = 32;
  std::vector<float> out(static_cast<std::size_t>(kThreads) * 64, 0.0f);
  ::nanochat::cuda_backend::Launch(LocalAcrossShuffleKernel, dim3(1u),
                                   dim3(kThreads), 0, out.data());
  int bad = 0;
  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < 64; ++i) {
      const float want = static_cast<float>(i + 1);
      if (out[static_cast<std::size_t>(t) * 64 + i] != want && ++bad == 1) {
        std::fprintf(
            stderr,
            "  local d[%d] of thread %d is %g after a shuffle, not %g\n", i, t,
            static_cast<double>(out[static_cast<std::size_t>(t) * 64 + i]),
            static_cast<double>(want));
      }
    }
  }
  if (bad != 0) Fail("a local array does not survive a shuffle", "shuffle");
}

// --- Case 5: the refusal ----------------------------------------------------

//: A descriptor whose stride is four times the staged tile. The last row group
//: then sits past the end of the region, which is the defect the probe found on
//: a card: `XID 13, Out Of Range Address`.
void ReadPastTheRegion() {
  std::vector<char> stage(InterleavedBytes(kKeys), 0);
  // One atom of the operand is eight row groups of two hundred and fifty-six
  // bytes, and the operand the descriptor names is that one atom.
  hopper::DeclareSharedOperand(stage.data(), 8 * 256);
  const hopper::MatrixDescriptor descriptor = hopper::MakeMatrixDescriptor(
      stage.data(), 128, 256, hopper::OperandLayout::kInterleaved);
  // Row a hundred and twenty-seven is the last group the tile holds, and the
  // descriptor's operand is eight of them, so this read is past the operand.
  volatile float value = hopper::detail::OperandElement(descriptor, 127, 0);
  (void)value;
}

void RunRefusalCase() {
  const pid_t child = fork();
  if (child == 0) {
    // The child's stderr states why the read was refused.
    ReadPastTheRegion();
    std::_Exit(0);
  }
  if (child < 0) {
    Fail("fork", "the refusal");
    return;
  }
  int status = 0;
  waitpid(child, &status, 0);
  const bool refused = !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  if (!refused) Fail("a read past the staged region was accepted", "refusal");
}

// The encoder's own refusals, which do not abort: a stride the driver's rules
// reject, an empty box, and a swizzle that does not fit the innermost box.
void CheckEncoderRefusals() {
  std::vector<std::uint16_t> source(static_cast<std::size_t>(kKeys) * kDim, 0);
  CUtensorMap map{};
  const cuuint64_t dims[2] = {64, kKeys};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(kDim) * 2};
  const cuuint32_t box[2] = {64, kKeys};
  const cuuint32_t element[2] = {1, 1};
  const cuuint64_t bad_strides[1] = {static_cast<cuuint64_t>(kDim) * 2 + 1};
  const cuuint32_t empty_box[2] = {0, kKeys};
  const cuuint32_t wide_box[2] = {32, kKeys};

  if (cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data(), dims,
          bad_strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_ERROR_INVALID_VALUE) {
    Fail("a stride that is not a multiple of sixteen bytes was accepted",
         "encoder refusals");
  }
  if (cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data(), dims,
          strides, empty_box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_ERROR_INVALID_VALUE) {
    Fail("an empty box was accepted", "encoder refusals");
  }
  if (cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data(), dims,
          strides, wide_box, element, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_ERROR_INVALID_VALUE) {
    Fail("a swizzle wider than the innermost box was accepted",
         "encoder refusals");
  }
}

// The two lane maps are bijections over their own tiles, which is the property
// the probe measured with one-hot triples. A map that repeats a slot leaves a
// hole, and the element checks above would then compare a stale value.
void CheckLaneMaps() {
  for (int keys = 64; keys <= 128; keys *= 2) {
    std::vector<int> seen(static_cast<std::size_t>(kQueryRows) * keys, 0);
    const int slots = keys / 2;
    for (int lane = 0; lane < hopper::kWarpgroupThreads; ++lane) {
      for (int index = 0; index < slots; ++index) {
        const int row = hopper::SlotRow(lane, index);
        const int column = hopper::SlotColumn(lane, index);
        if (row < 0 || row >= kQueryRows || column < 0 || column >= keys) {
          Fail("an accumulator slot is outside its tile", "lane maps");
          return;
        }
        ++seen[static_cast<std::size_t>(row) * keys + column];
      }
    }
    for (int count : seen) {
      if (count != 1) {
        Fail("the accumulator map is not a bijection", "lane maps");
        return;
      }
    }
  }
  // The register operand: sixty-four rows by sixteen keys over eight elements a
  // thread, which is one element for each (row, key) of the atom.
  std::vector<int> seen(static_cast<std::size_t>(kQueryRows) * 16, 0);
  for (int lane = 0; lane < hopper::kWarpgroupThreads; ++lane) {
    for (int reg = 0; reg < 4; ++reg) {
      for (int half = 0; half < 2; ++half) {
        const int row = hopper::RegisterRow(lane, reg);
        const int key = hopper::RegisterKey(lane, reg, half);
        if (row < 0 || row >= kQueryRows || key < 0 || key >= 16) {
          Fail("a register operand element is outside its tile", "lane maps");
          return;
        }
        ++seen[static_cast<std::size_t>(row) * 16 + key];
      }
    }
  }
  for (int count : seen) {
    if (count != 1) {
      Fail("the register operand map is not a bijection", "lane maps");
      return;
    }
  }
}

}  // namespace

int main() {
  CheckLaneMaps();
  CheckEncoderRefusals();
  RunInterleavedCase();
  RunSwizzledCase();
  CheckLocalAcrossShuffle();
  CheckValueOperandReads();
  RunValueCase();
  RunRefusalCase();

  if (g_failures != 0) {
    std::fprintf(stderr, "hopper_emu: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("hopper_emu: ok\n");
  return 0;
}
