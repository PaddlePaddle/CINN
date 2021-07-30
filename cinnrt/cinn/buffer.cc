#include "cinnrt/cinn/buffer.h"

namespace cinnrt {

void Buffer::Resize(uint32_t size) {
  if (size_ > 0) {
    Free();
    size_ = 0;
  }

  if (size_ != size) {
    data_.memory = reinterpret_cast<uint8_t*>(Malloc(size));
    size_        = size;
  }
}

void Buffer::Resize(uint32_t alignment, uint32_t size) {
  if (size_ > 0) {
    Free();
    size_ = 0;
  }

  if (size_ != size) {
    data_.memory = reinterpret_cast<uint8_t*>(AlignedAlloc(alignment, size));
    size_        = size;
  }
}

void Buffer::SetTarget(const cinnrt::common::Target& target) {
  target_           = target;
  memory_mng_cache_ = MemoryManager::Global().RetrieveSafely(target_.arch);
}

void Buffer::ResizeLazy(uint32_t size) {
  if (size <= size_) return;
  Resize(size);
}

void Buffer::ResizeLazy(uint32_t alignment, uint32_t size) {
  if (size <= size_) return;
  Resize(alignment, size);
}

void Buffer::Resize(uint32_t size, const cinnrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  Resize(size);
}

void Buffer::Resize(uint32_t alignment, uint32_t size, const cinnrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  Resize(alignment, size);
}

void Buffer::ResizeLazy(uint32_t size, const cinnrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  ResizeLazy(size);
}

void Buffer::ResizeLazy(uint32_t alignment, uint32_t size, const cinnrt::common::Target& target) {
  if (target.arch != target_.arch) {
    Free();
    SetTarget(target);
  }
  ResizeLazy(alignment, size);
}

}  // namespace cinnrt
