#include "cinn/lang/tensor.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir.h"

namespace cinn {
namespace lang {

TEST(Tensor, basic) {
  Expr M(100);
  Expr N(20);

  Expr x(100);

  Var i("i"), j("j");

  Tensor tensor({M, N}, {i, j}, Float(32), Expr(0), <#initializer #>);

  auto slice = tensor(Expr(2), Expr(1));
}

}  // namespace lang
}  // namespace cinn
