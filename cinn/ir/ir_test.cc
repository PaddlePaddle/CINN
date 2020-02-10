#include "cinn/ir/ir.h"

#include <gtest/gtest.h>

#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

TEST(Expr, basic) {
  Expr a(1);
  auto b = Expr(a);
  LOG(INFO) << b.as_int32();
}

}  // namespace ir
}  // namespace cinn
