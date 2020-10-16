#include "cinn/host_context/value.h"
#include <gtest/gtest.h>

namespace cinn {
namespace host_context {

TEST(Value, test) {
  Value x(12);
  ASSERT_EQ(x.get<int>(), 12);

  Value y(1.2f);
  ASSERT_EQ(x.get<float>(), 1.2f);

  Value z(true);
  ASSERT_EQ(x.get<bool>(), true);
}

}  // namespace host_context
}  // namespace cinn
