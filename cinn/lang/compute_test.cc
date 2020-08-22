#include "cinn/lang/compute.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace lang {

TEST(Call, basic) {
  Expr M(100);

  Placeholder<float> x("x", {M, Expr(10)});
  Placeholder<float> y("y", {M, Expr(10)});

  std::vector<ReturnType> return_types({{Float(32), std::vector<Expr>{{M, Expr(20)}}, "C"}});
  auto tensors = CallLowered("lowered_fun0", {Expr(x), Expr(y)}, return_types);
}

}  // namespace lang
}  // namespace cinn
