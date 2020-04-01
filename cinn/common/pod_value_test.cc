#include "cinn/common/pod_value.h"

#include <gtest/gtest.h>

namespace cinn {
namespace common {

TEST(PODValue, test) {
  {
    PODValue value;
    value.Set(32);
    ASSERT_EQ(int(value), 32);  // NOLINT
  }
  {
    PODValue value;
    value.Set(32.f);
    ASSERT_NEAR(float(value), 32.f, 1e-6);  // NOLINT
  }
}

TEST(PODValue, buffer) {
  PODValue value;
  cinn_buffer_t* v = nullptr;
  value.Set(v);
  ASSERT_EQ((cinn_buffer_t*)value, nullptr);
}

}  // namespace common
}  // namespace cinn
