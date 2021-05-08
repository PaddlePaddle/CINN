#include "cinn/ir/ir_verify.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"

namespace cinn::ir {

TEST(IrVerify, basic) {
  Expr a(1);
  Expr b(1);
  IrVerify(a + b);
}

}  // namespace cinn::ir
