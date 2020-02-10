#include "cinn/ir/ir_operators.h"

#include <gtest/gtest.h>

namespace cinn {
namespace ir {

TEST(ir_operators, test) {
  Expr a(1);
  Expr b = a + 1;
}

}  // namespace ir
}  // namespace cinn