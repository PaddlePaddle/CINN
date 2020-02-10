#include "cinn/ir/buffer.h"
#include <gtest/gtest.h>
#include <vector>

namespace cinn {
namespace ir {

TEST(Buffer, basic) {
  Var ptr("buff", Float(32));
  std::vector<Expr> shape({Expr(100), Expr(20)});
  Var i("i"), j("j");
  std::vector<Expr> strides({Expr(0), Expr(0)});
  auto buffer = _Buffer_::Make(ptr, ptr->type(), shape, strides, Expr(0), "buf", "", 0, 0);
  LOG(INFO) << "node type: " << buffer->node_type();
}

}  // namespace ir
}  // namespace cinn