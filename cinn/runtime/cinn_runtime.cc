#include "cinn/runtime/cinn_runtime.h"

extern "C" {

int cinn_buffer_malloc(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->malloc(context, buf);
}

int cinn_buffer_free(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  return buf->device_interface->impl->free(context, buf);
}

int cinn_device_sync(void* context, struct cinn_buffer_t* buf) {
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  // ASSERT_NOT_NULL(context)
  buf->device_interface->impl->sync(context, buf);
  return 0;
}

int cinn_device_release(void* context, const struct cinn_device_interface_t* device_interface) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(device_interface)
  CINN_NOT_IMPLEMENTED
}

int cinn_buffer_copy_to_host(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->copy_to_host(context, buf);
}

int cinn_buffer_copy_to_device(void* context, struct cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  ASSERT_NOT_NULL(buf->device_interface)
  return buf->device_interface->impl->copy_to_device(context, buf);
}
int cinn_buffer_copy(void* context, struct cinn_buffer_t* src, struct cinn_buffer_t* dst) {
  // ASSERT_NOT_NULL(context);
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(dst);
  return dst->device_interface->buffer_copy(context, src, dst);
}

void* cinn_buffer_get_data_handle(struct cinn_buffer_t* buf) {
  CINN_CHECKP(buf, "%s", "buffer is null");
  return buf->host_memory;
}

void* cinn_buffer_get_data_const_handle(const struct cinn_buffer_t* buf) {
  CINN_CHECKP(buf, "%s", "buffer is null");
  return buf->host_memory;
}

cinn_type_t cinn_unk_t() { return cinn_type_t(cinn_type_unk, 0); }
cinn_type_t cinn_int32_t() { return cinn_type_t(cinn_type_int, 32); }
cinn_type_t cinn_int64_t() { return cinn_type_t(cinn_type_int, 64); }
cinn_type_t cinn_uint32_t() { return cinn_type_t(cinn_type_uint, 32); }
cinn_type_t cinn_uint64_t() { return cinn_type_t(cinn_type_uint, 64); }
cinn_type_t cinn_float32_t() { return cinn_type_t(cinn_type_float, 32); }
cinn_type_t cinn_float64_t() { return cinn_type_t(cinn_type_float, 64); }

}  // extern "C"

struct cinn_buffer_t* cinn_buffer_t::new_(cinn_device_kind_t device,
                                          cinn_type_t type,
                                          const std::vector<int>& shape,
                                          int align) {
  int32_t dimensions     = shape.size();
  cinn_dimension_t* dims = new cinn_dimension_t[dimensions];
  memcpy(dims, shape.data(), shape.size() * sizeof(int));

  struct cinn_buffer_t* x = new (struct cinn_buffer_t);
  x->type                 = type;
  x->device               = device;
  // NOTE set device_interface for each buffer.
  switch (x->device) {
    case cinn_x86_device:
      x->device_interface = &cinn_x86_device_interface;
      break;
    case cinn_unk_device:
      fprintf(stderr, "Device type of buffer should be set, found Unk");
      abort();
      break;
    default:
      fprintf(stderr, "Not supported device type");
      abort();
  }

  x->dims       = dims;
  x->dimensions = dimensions;
  x->align      = align;
  return x;
}
