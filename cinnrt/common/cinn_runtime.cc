#include "cinnrt/common/cinn_runtime.h"

#include <stdarg.h>
#include <stdio.h>

#include <cmath>

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
