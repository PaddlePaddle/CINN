#pragma once

#include <glog/logging.h>

#include <memory>
#include <unordered_map>

#include "cinnrt/cinn/memory.h"
#include "cinnrt/common/cinn_runtime.h"
#include "cinnrt/common/macros.h"
#include "cinnrt/common/target.h"

namespace cinnrt {

/**
 * Buffer helps to hold the memory, and offers a set of methods to help manage the memory.
 */
struct Buffer final {
  Buffer() = default;
  explicit Buffer(const cinnrt::common::Target& target) { SetTarget(target); }

  //! Resize the memory hold by this buffer *exactlly* to \p size.
  void Resize(uint32_t size);
  void Resize(uint32_t alignment, uint32_t size);

  //! Lazily resize the memory.
  void ResizeLazy(uint32_t size);
  void ResizeLazy(uint32_t alignment, uint32_t size);

  //! Resize the memory to \p size in target \p target.
  void Resize(uint32_t size, const cinnrt::common::Target& target);
  void Resize(uint32_t alignment, uint32_t size, const cinnrt::common::Target& target);

  //! Lazily resize the memory to \p size in target \p target.
  void ResizeLazy(uint32_t size, const cinnrt::common::Target& target);
  void ResizeLazy(uint32_t alignment, uint32_t size, const cinnrt::common::Target& target);

  void SetTarget(const cinnrt::common::Target& target);

  const cinn_buffer_t* data() const { return &data_; }
  cinn_buffer_t* data() { return &data_; }

  //! Free all the memory owned by this buffer.
  void Free() {
    if (!data_.memory) return;
    memory_mng_cache_->free(data_.memory);
  }

 private:
  inline void* Malloc(uint32_t size) CINN_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->malloc(size);
  }

  inline void* AlignedAlloc(uint32_t alignment, uint32_t size) CINN_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->aligned_alloc(alignment, size);
  }

 private:
  cinn_buffer_t data_;

  //! The place where this buffer locates.
  cinnrt::common::Target target_;

  //! Number of bytes of this buffer.
  uint32_t size_{};

  //! Hold the corresponding memory manager for speed.
  MemoryInterface* memory_mng_cache_{};
};

}  // namespace cinnrt
