#include "cinn/runtime/cinn_runtime.h"

#include <gtest/gtest.h>

TEST(buffer, basic) {
  auto* buffer = cinn_buffer_t::new_(cinn_x86_device, cinn_float32_t(), {3, 10});
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(buffer->device_interface);
  ASSERT_EQ(buffer->device_interface, &cinn_x86_device_interface);
  buffer->device_interface->impl->malloc(NULL, buffer);
  auto* data = reinterpret_cast<float*>(buffer->host_memory);
  data[0]    = 0.f;
  data[1]    = 1.f;
  EXPECT_EQ(data[0], 0.f);
  EXPECT_EQ(data[1], 1.f);
}

TEST(cinn_print_debug_string, basic) {
  cinn_print_debug_string("hello world");
  cinn_print_debug_string("should be 1, %d", 1);
  int a = 1;
  cinn_print_debug_string("should be pointer, %p", &a);
  cinn_print_debug_string("should be 1, %d", a);
}

TEST(cinn_args_construct, basic) {
  cinn_pod_value_t arr[4];
  cinn_pod_value_t a0(0);
  cinn_pod_value_t a1(1);
  cinn_pod_value_t a2(2);
  cinn_pod_value_t a3(3);
  cinn_args_construct(arr, 4, &a0, &a1, &a2, &a3);
  for (int i = 0; i < 4; i++) ASSERT_EQ((int)arr[i], i);
}
