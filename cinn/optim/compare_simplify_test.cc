#include "cinn/optim/compare_simplify.h"
#include <gtest/gtest.h>
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

TEST(CompareSimplify, basic_true) {
  Expr a = (Expr(1) > Expr(2));
  LOG(INFO) << a;

  CompareSimplify(&a);

  LOG(INFO) << a;

  ASSERT_EQ(utils::GetStreamCnt(a), "0");
}

TEST(CompareSimplify, basic_false) {
  Expr a = (Expr(1) < Expr(2));
  LOG(INFO) << a;

  CompareSimplify(&a);

  LOG(INFO) << a;

  ASSERT_EQ(utils::GetStreamCnt(a), "1");
}

}  // namespace cinn::optim
