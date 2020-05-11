#include "cinn/common/cinn_value.h"

#include <gtest/gtest.h>

#include "cinn/common/common.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace common {

TEST(CINNValue, test) {
  {
    CINNValue value(32);
    ASSERT_EQ(int(value), 32);  // NOLINT
  }
  {
    CINNValue value(32.f);
    ASSERT_NEAR(float(value), 32.f, 1e-6);  // NOLINT
  }
}

TEST(CINNValue, buffer) {
  cinn_buffer_t* v = nullptr;
  CINNValue value(v);
  ASSERT_EQ((cinn_buffer_t*)value, nullptr);
}

TEST(CINNValue, Expr) {
  Expr a(1);

  {
    CINNValue value(a);
    ASSERT_TRUE(a == value);
  }

  {
    CINNValue copied = CINNValue(a);
    ASSERT_TRUE(copied == common::make_const(1));
  }
}

}  // namespace common
}  // namespace cinn
