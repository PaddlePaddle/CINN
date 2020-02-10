#include "cinn/lang/placeholder.h"
#include <gtest/gtest.h>

namespace cinn {
namespace lang {

TEST(placeholder, basic) {
  Expr M(100);
  Expr N(20);

  Placeholder<float> x({M, N});

  ir::Var i("i");
  ir::Var j("j");

  auto slice = x(Expr(i), Expr(j));
}

}  // namespace lang
}  // namespace cinn