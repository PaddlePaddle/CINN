#include "cinn/common/test_helper.h"

namespace cinn {
namespace common {

cinn_buffer_t* BufferBuilder::Build() {
  cinn_type_t cinn_type;
  if (type_ == type_of<float>()) {
    cinn_type = cinn_float32_t();
  } else if (type_ == type_of<double>()) {
    cinn_type = cinn_float64_t();
  } else if (type_ == type_of<int32_t>()) {
    cinn_type = cinn_int32_t();
  } else if (type_ == type_of<int64_t>()) {
    cinn_type = cinn_int64_t();
  } else {
    NOT_IMPLEMENTED
  }

  auto* buffer = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_type, shape_, align_);

  cinn_buffer_malloc(nullptr, buffer);

  if (init_type_ == 0) {
    memset(buffer->host_memory, 0, buffer->memory_size);
  } else if (init_type_ == 1) {
    if (type_ == type_of<float>()) {
      RandomFloat<float>(buffer->host_memory, buffer->num_elements());
    } else if (type_ == type_of<double>()) {
      RandomFloat<double>(buffer->host_memory, buffer->num_elements());
    } else if (type_ == type_of<int32_t>()) {
      RandomInt<int32_t>(buffer->host_memory, buffer->num_elements());
    } else if (type_ == type_of<int64_t>()) {
      RandomInt<int64_t>(buffer->host_memory, buffer->num_elements());
    } else {
      NOT_IMPLEMENTED
    }
  } else {
    NOT_IMPLEMENTED
  }
  return buffer;
}

}  // namespace common
}  // namespace cinn
