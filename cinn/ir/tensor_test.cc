#include "cinn/ir/tensor.h"
#include <gtest/gtest.h>
#include "cinn/ir/function.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {

TEST(Tensor, test) {
  Var M("M"), N("N");
  Tensor tensor({M, N}, Float(32));
  CHECK_EQ(tensor->type(), Float(32));

  Var i("i"), j("j");

  auto tmp = tensor(Expr(i), Expr(j));
  LOG(INFO) << tmp;
}

TEST(Tensor, func) {
  Var M, N;
  Tensor A({M, N}, Float(32));

  PackedFunc::func_t body = [&](Args args, RetValue* ret) {
    Expr i = args[0];
    Expr j = args[1];

    Expr elem = A(i, j);
    Expr tmp  = elem + 1.f;
    ret->Set(tmp);
  };

  PackedFunc func(body);

  Var i("i"), j("j");
  Expr res = func(i, j);
  LOG(INFO) << res;
}

}  // namespace ir
}  // namespace cinn