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
    CINN_NOT_IMPLEMENTED
  }

  auto* buffer = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_type, shape_, align_);

  cinn_buffer_malloc(nullptr, buffer);

  switch (init_type_) {
    case InitType::kZero:
      memset(buffer->host_memory, 0, buffer->memory_size);
      break;

    case InitType::kRandom:
      if (type_ == type_of<float>()) {
        RandomFloat<float>(buffer->host_memory, buffer->num_elements());
      } else if (type_ == type_of<double>()) {
        RandomFloat<double>(buffer->host_memory, buffer->num_elements());
      } else if (type_ == type_of<int32_t>()) {
        RandomInt<int32_t>(buffer->host_memory, buffer->num_elements());
      } else if (type_ == type_of<int64_t>()) {
        RandomInt<int64_t>(buffer->host_memory, buffer->num_elements());
      }
      break;

    case InitType::kSetValue:
      if (type_ == type_of<int>()) {
        SetVal<int>(buffer->host_memory, buffer->num_elements(), init_val_);
      } else if (type_ == type_of<float>()) {
        SetVal<float>(buffer->host_memory, buffer->num_elements(), init_val_);
      } else {
        CINN_NOT_IMPLEMENTED
      }
      break;
  }

  return buffer;
}

}  // namespace common
}  // namespace cinn
