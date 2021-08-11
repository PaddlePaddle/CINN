#include "cinnrt/common/buffer.h"

#include <stdarg.h>
#include <stdio.h>

#include <cmath>

namespace cinnrt {

struct cinn_buffer_t* cinn_buffer_t::new_(cinn_device_kind_t device,
                                          cinn_type_t type,
                                          const std::vector<int>& shape,
                                          int align) {
  int32_t dimensions = shape.size();
  CINN_CHECK(shape.size() < CINN_BUFFER_MAX_DIMS);

  struct cinn_buffer_t* buf = (struct cinn_buffer_t*)malloc(sizeof(struct cinn_buffer_t));
  memcpy(&(buf->dims[0]), shape.data(), shape.size() * sizeof(int));
  buf->type        = type;
  buf->device      = device;
  buf->memory      = nullptr;
  buf->memory_size = 0;
  buf->lazy        = true;
  // NOTE set device_interface for each buffer.
  switch (buf->device) {
    case cinn_x86_device:
      buf->device_interface = cinn_x86_device_interface();
      break;
    case cinn_unk_device:
      fprintf(stderr, "Device type of buffer should be set, found Unk");
      abort();
      break;
    default:
      fprintf(stderr, "Not supported device type");
      abort();
  }

  buf->dimensions = dimensions;
  buf->align      = align;
  return buf;
}

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
