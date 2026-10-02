#ifndef NANOCHAT_SRC_WORKSPACE_H_
#define NANOCHAT_SRC_WORKSPACE_H_

#include <cstddef>
#include <cstdint>

// Runtime tensor plumbing: the single-allocation, bump-pointer activation
// arena behind the fixed train graph (docs/model.md, "Workspace and memory").
//
// The graph computes every slot size on the host from the Config and
// `(batch, seq)`, reserves one contiguous block, and then carves named slots
// out of it in a fixed order. Slots are never freed individually; the whole
// arena is released when the model is destroyed or rebuilt for a different
// shape. This is the llm.c `ActivationTensors` pattern without the C macros.

namespace nanochat {

class Workspace {
 public:
  Workspace() = default;
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  // Allocates the backing store once. A later Reserve with a larger request
  // reallocates and invalidates every pointer handed out before it, so callers
  // must reserve before taking any slot.
  void Reserve(std::size_t bytes, std::size_t alignment = 64);

  // Starts a fresh bump pass over the existing allocation.
  void Reset();

  // Carves `count` elements of T out of the arena. Returns nullptr for a zero
  // count.
  template <typename T>
  T* Alloc(std::size_t count) {
    if (count == 0) return nullptr;
    return static_cast<T*>(AllocRaw(count * sizeof(T), alignof(T)));
  }

  std::size_t bytes_reserved() const { return capacity_; }
  std::size_t bytes_used() const { return offset_; }

 private:
  void* AllocRaw(std::size_t bytes, std::size_t alignment);

  void* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t offset_ = 0;
  std::size_t alignment_ = 64;
};

}  // namespace nanochat

#endif  // NANOCHAT_SRC_WORKSPACE_H_
