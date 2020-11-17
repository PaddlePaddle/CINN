#include "cinn/optim/if_simplify.h"
#include <gtest/gtest.h>
#include <string>
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

TEST(IfSimplify, if_true) {
  Var n("n");
  auto e = ir::IfThenElse::Make(Expr(1) /*true*/, ir::Let::Make(n, Expr(1)), ir::Let::Make(n, Expr(2)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  ASSERT_EQ(utils::GetStreamCnt(e), "int32 n = 1");
}

TEST(IfSimplify, if_false) {
  Var n("n");
  auto e = ir::IfThenElse::Make(Expr(0) /*false*/, ir::Let::Make(n, Expr(1)), ir::Let::Make(n, Expr(2)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  ASSERT_EQ(utils::GetStreamCnt(e), "int32 n = 2");
}

TEST(IfSimplify, if_else_empty) {
  Var n("n");
  auto e = ir::IfThenElse::Make(Expr(0) /*false*/, ir::Let::Make(n, Expr(1)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  std::string target = utils::Trim(R"ROC(
{

}
)ROC");

  ASSERT_EQ(utils::GetStreamCnt(e), target);
}

}  // namespace cinn::optim
