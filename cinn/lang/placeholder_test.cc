#include "cinn/lang/placeholder.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace lang {

TEST(placeholder, basic) {
  Expr M(100);
  Expr N(20);

  Placeholder<float> x("x", {M, N});

  ir::Var i("i");
  ir::Var j("j");

  auto slice = x(i, j);
  LOG(INFO) << "slice " << slice;
}

}  // namespace lang
}  // namespace cinn