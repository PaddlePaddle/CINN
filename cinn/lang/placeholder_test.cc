#include "cinn/lang/placeholder.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace lang {

TEST(placeholder, basic) {
  Expr M(100);
  Expr N(20);

  Placeholder<float> x({M, N});

  auto x_buffer = Expr(x.buffer());
  ir::Var i("i");
  ir::Var j("j");

  auto slice = x(Expr(i), Expr(j));
  LOG(INFO) << "slice " << slice;
}

}  // namespace lang
}  // namespace cinn