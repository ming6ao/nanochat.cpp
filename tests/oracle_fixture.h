#ifndef NANOCHAT_TESTS_ORACLE_FIXTURE_H_
#define NANOCHAT_TESTS_ORACLE_FIXTURE_H_

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Reader for the CPU oracle fixture, `tests/data/debug_state.bin`, produced
// offline by `tools/dump_oracle.py`. The fixture is data, so this reader has no
// torch dependency and no dependency on the model: any test (CPU, CUDA, or the
// model skeleton) can parse the same reference tensors.
//
// The same little-endian record encoding backs the evaluation fixture
// (`NANOEVL1`, written by `python/nanochat_cpp/eval_fixture.py` and
// `tools/dump_eval_fixture.py`); `Parse` accepts either magic so the score
// parity test can reuse this reader. See the module docstring in
// tools/dump_oracle.py for the exact layout. This header is header-only on
// purpose: it can be included directly, or linked through the
// `//tests:oracle_fixture` library.

namespace nanochat {
namespace oracle {

enum class DType : std::uint8_t {
  kFp32 = 0,
  kInt32 = 1,
  kFp64 = 2,
  kInt64 = 3,
  kUInt8 = 4,
};

inline std::size_t ElemSize(DType dtype) {
  switch (dtype) {
    case DType::kFp32:
    case DType::kInt32:
      return 4;
    case DType::kFp64:
    case DType::kInt64:
      return 8;
    case DType::kUInt8:
      return 1;
  }
  throw std::runtime_error("oracle: unknown dtype code");
}

// A non-owning view of one record. `data` points into the owning Fixture's
// payload storage, which stays alive for the Fixture's lifetime.
struct Tensor {
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t bytes = 0;

  std::int64_t numel() const {
    std::int64_t count = 1;
    for (std::int64_t extent : shape) count *= extent;
    return count;
  }

  bool is_fp32() const { return dtype == DType::kFp32; }
  bool is_int32() const { return dtype == DType::kInt32; }

  const float* f32() const {
    if (dtype != DType::kFp32)
      throw std::runtime_error("oracle: record is not fp32");
    return reinterpret_cast<const float*>(data);
  }

  const std::int32_t* i32() const {
    if (dtype != DType::kInt32)
      throw std::runtime_error("oracle: record is not int32");
    return reinterpret_cast<const std::int32_t*>(data);
  }

  const std::int64_t* i64() const {
    if (dtype != DType::kInt64)
      throw std::runtime_error("oracle: record is not int64");
    return reinterpret_cast<const std::int64_t*>(data);
  }

  float scalar_f32() const {
    if (numel() != 1)
      throw std::runtime_error("oracle: record is not a scalar");
    return f32()[0];
  }

  std::int64_t scalar_int() const {
    if (numel() != 1)
      throw std::runtime_error("oracle: record is not a scalar");
    if (dtype == DType::kInt32) return i32()[0];
    if (dtype == DType::kInt64) return i64()[0];
    throw std::runtime_error("oracle: integer scalar has an unexpected dtype");
  }
};

namespace detail {

inline std::uint16_t ReadU16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
}

inline std::uint32_t ReadU32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t ReadU64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
  return value;
}

}  // namespace detail

class Fixture {
 public:
  static Fixture Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("oracle: cannot open fixture '" + path + "'");
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    if (bytes.empty())
      throw std::runtime_error("oracle: fixture is empty: " + path);
    return Parse(std::move(bytes));
  }

  static Fixture Parse(std::vector<std::uint8_t> bytes) {
    Fixture fixture;
    if (bytes.size() < 16) throw std::runtime_error("oracle: truncated header");
    if (std::memcmp(bytes.data(), "NANOORC1", 8) != 0 &&
        std::memcmp(bytes.data(), "NANOEVL1", 8) != 0) {
      throw std::runtime_error(
          "oracle: bad magic (expected NANOORC1 or NANOEVL1)");
    }
    const std::uint32_t version = detail::ReadU32(bytes.data() + 8);
    if (version != 1) throw std::runtime_error("oracle: unsupported version");
    const std::uint32_t count = detail::ReadU32(bytes.data() + 12);

    std::size_t offset = 16;
    for (std::uint32_t i = 0; i < count; ++i) {
      if (offset + 2 > bytes.size())
        throw std::runtime_error("oracle: truncated name length");
      const std::uint16_t name_len = detail::ReadU16(bytes.data() + offset);
      offset += 2;
      if (offset + name_len > bytes.size())
        throw std::runtime_error("oracle: truncated name");
      std::string name(reinterpret_cast<const char*>(bytes.data() + offset),
                       name_len);
      offset += name_len;

      if (offset + 2 > bytes.size())
        throw std::runtime_error("oracle: truncated dtype");
      const DType dtype = static_cast<DType>(bytes[offset]);
      const std::uint8_t ndim = bytes[offset + 1];
      offset += 2;

      std::vector<std::int64_t> shape(ndim);
      for (std::uint8_t d = 0; d < ndim; ++d) {
        if (offset + 8 > bytes.size())
          throw std::runtime_error("oracle: truncated shape");
        shape[d] =
            static_cast<std::int64_t>(detail::ReadU64(bytes.data() + offset));
        offset += 8;
      }

      std::int64_t numel = 1;
      for (std::int64_t extent : shape) numel *= extent;
      const std::size_t payload_bytes =
          static_cast<std::size_t>(numel) * ElemSize(dtype);
      if (offset + payload_bytes > bytes.size()) {
        throw std::runtime_error("oracle: truncated payload for '" + name +
                                 "'");
      }
      fixture.payloads_.emplace_back(bytes.begin() + offset,
                                     bytes.begin() + offset + payload_bytes);
      offset += payload_bytes;

      Tensor tensor;
      tensor.dtype = dtype;
      tensor.shape = shape;
      tensor.data = fixture.payloads_.back().data();
      tensor.bytes = payload_bytes;
      fixture.tensors_.emplace(name, std::move(tensor));
      fixture.order_.push_back(std::move(name));
    }
    if (offset != bytes.size())
      throw std::runtime_error("oracle: trailing bytes");
    return fixture;
  }

  bool Has(const std::string& name) const { return tensors_.count(name) != 0; }

  const Tensor& Get(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      throw std::runtime_error("oracle: missing record '" + name + "'");
    }
    return it->second;
  }

  const std::vector<std::string>& names() const { return order_; }
  std::size_t size() const { return tensors_.size(); }

 private:
  std::vector<std::vector<std::uint8_t>> payloads_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::vector<std::string> order_;
};

}  // namespace oracle
}  // namespace nanochat

#endif  // NANOCHAT_TESTS_ORACLE_FIXTURE_H_
