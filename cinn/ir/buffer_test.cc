#include "cinn/ir/buffer.h"

#include <gtest/gtest.h>

#include <vector>

#include "cinn/common/common.h"

namespace cinn {
namespace ir {

TEST(Buffer, basic) {
  Var ptr("buff", Float(32));
  std::vector<Expr> shape({Expr(100), Expr(20)});
  Var i("i"), j("j");
  std::vector<Expr> strides({Expr(0), Expr(0)});
  auto buffer = _Buffer_::Make(ptr, ptr->type(), shape, strides, Expr(0), "buf", "", 0, 0);

  // Check shared
  ASSERT_EQ(ref_count(buffer.get()).val(), 1);

  {
    auto buffer1 = buffer;
    ASSERT_EQ(ref_count(buffer.get()).val(), 2);
    ASSERT_EQ(ref_count(buffer1.get()).val(), 2);
  }

  ASSERT_EQ(ref_count(buffer.get()).val(), 1);
}

}  // namespace ir
}  // namespace cinn