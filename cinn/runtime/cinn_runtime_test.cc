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
