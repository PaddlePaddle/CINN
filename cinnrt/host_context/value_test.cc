#include "value.h"

#include <gtest/gtest.h>

namespace cinn {
namespace host_context {

TEST(ValueRef, test) {
  ValueRef x(12);
  ASSERT_EQ(x.get<int>(), 12);

  ValueRef y(1.2f);
  ASSERT_EQ(y.get<float>(), 1.2f);

  ValueRef z(true);
  ASSERT_EQ(z.get<bool>(), true);
}

}  // namespace host_context
}  // namespace cinn
