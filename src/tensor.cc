// Runtime tensor plumbing: the bump-pointer activation workspace. See
// workspace.h and docs/model.md. The backend owns the raw allocation
// (`kernels::Alloc`), so the same code works for the CPU reference backend and
// the CUDA backend.

#include "workspace.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "nanochat/kernels.h"

namespace nanochat {

Workspace::~Workspace() {
  if (base_ != nullptr) {
    kernels::Free(base_);
    base_ = nullptr;
  }
  capacity_ = 0;
  offset_ = 0;
}

void Workspace::Reserve(std::size_t bytes, std::size_t alignment) {
  if (alignment == 0) alignment = 64;
  if (base_ != nullptr && bytes <= capacity_) {
    alignment_ = alignment;
    return;
  }
  if (base_ != nullptr) {
    kernels::Free(base_);
    base_ = nullptr;
    capacity_ = 0;
  }
  if (bytes == 0) return;
  base_ = kernels::Alloc(bytes);
  capacity_ = bytes;
  offset_ = 0;
  alignment_ = alignment;
}

void Workspace::Reset() { offset_ = 0; }

void* Workspace::AllocRaw(std::size_t bytes, std::size_t alignment) {
  if (alignment == 0) alignment = 1;
  const std::size_t mask = alignment - 1;
  const std::size_t start = (offset_ + mask) & ~mask;
  if (start + bytes > capacity_) {
    // The host graph under-reserved its workspace. Fail loudly rather than
    // corrupting the arena silently.
    std::fprintf(stderr,
                 "nanochat: workspace overflow: need %zu bytes, have %zu "
                 "(used %zu)\n",
                 start + bytes, capacity_, offset_);
    std::abort();
  }
  offset_ = start + bytes;
  return static_cast<std::uint8_t*>(base_) + start;
}

}  // namespace nanochat
