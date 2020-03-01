#include "cinn/runtime/cinn_runtime.h"

#include <gtest/gtest.h>

TEST(buffer, basic) {
  auto* buffer = cinn_buffer_t::new_(cinn_x86_device);
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(buffer->device_interface);
  ASSERT_EQ(buffer->device_interface, &cinn_x86_device_interface);
}
