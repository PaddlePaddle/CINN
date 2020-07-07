#include <stdlib.h>

#include "cinn/runtime/cinn_runtime.h"

int cinn_x86_malloc(void* context, cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  uint64_t memory_size = buf->num_elements() * buf->type.bytes();
  CINN_CHECK(memory_size > 0);
  if (buf->memory_size < memory_size) {
    if (buf->host_memory) {
      free(buf->host_memory);
    }
    int bytes = buf->type.bytes() * buf->num_elements();
    if (buf->align == 0) {
      buf->host_memory = (unsigned char*)malloc(bytes);
    } else {
      buf->host_memory = (unsigned char*)aligned_alloc(buf->align, bytes);
    }
    buf->memory_size = memory_size;
    CINN_LOG("buf.memory size is %ld\n", buf->memory_size);
  }
  ASSERT_NOT_NULL(buf->host_memory);
  return 0;
}

int cinn_x86_free(void* context, cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context);
  ASSERT_NOT_NULL(buf);
  if (buf->host_memory) {
    free(buf->host_memory);
    buf->host_memory = NULL;
  }
  return 0;
}

// All the following operations are not support by X86 device, just leave them empty.
// @{
int cinn_x86_sync(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_release(void* context) { return 0; }
int cinn_x86_copy_to_host(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_copy_to_device(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_buffer_copy(void* context, cinn_buffer_t* src, cinn_buffer_t* dst) { return 0; }
// @}

cinn_device_interface_impl_t cinn_x86_device_impl{&cinn_x86_malloc,
                                                  &cinn_x86_free,
                                                  &cinn_x86_sync,
                                                  &cinn_x86_release,
                                                  &cinn_x86_copy_to_host,
                                                  &cinn_x86_copy_to_device,
                                                  &cinn_x86_buffer_copy};

cinn_device_interface_t cinn_x86_device_interface_interface{&cinn_buffer_malloc,
                                                            &cinn_buffer_free,
                                                            &cinn_device_sync,
                                                            &cinn_device_release,
                                                            &cinn_buffer_copy_to_host,
                                                            &cinn_buffer_copy_to_device,
                                                            &cinn_buffer_copy,
                                                            &cinn_x86_device_impl};

struct cinn_device_interface_t* cinn_x86_device_interface() {
  return &cinn_x86_device_interface_interface;
}
