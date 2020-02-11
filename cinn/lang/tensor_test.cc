#include "cinn/lang/tensor.h"

#include <gtest/gtest.h>

#include "cinn/ir/function.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {

TEST(Tensor, test) {
  const int M = 100;
  const int N = 100;
  Var i("i"), j("j");
  Tensor tensor({Expr(M), Expr(N)}, {i, j}, Float(32), Expr(1), "A");
  CHECK_EQ(tensor->type(), Float(32));

  auto tmp = tensor(Expr(i), Expr(j));
  LOG(INFO) << tmp;
}

}  // namespace ir
}  // namespace cinn
