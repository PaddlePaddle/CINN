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

TEST(placeholder, dynamic_shape) {
  Var B("B", Int(32));
  Expr N(20);

  Placeholder<float> x("x", {B, N});

  Var i("i"), j("j");
  auto slice = x(i, j);
}

}  // namespace lang
}  // namespace cinn
