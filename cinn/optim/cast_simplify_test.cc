#include "cinn/optim/cast_simplify.h"
#include <gtest/gtest.h>
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

TEST(CastSimplify, same_type) {
  Var n("n");
  Expr a = ir::Cast::Make(Int(32), n);
  LOG(INFO) << n->type();
  LOG(INFO) << a;
  CastSimplify(&a);
  ASSERT_EQ(utils::GetStreamCnt(a), "n");
}

TEST(CastSimplify, Imm_int) {
  Expr a = ir::Cast::Make(Int(64), Expr(1));
  Expr c = ir::Cast::Make(Int(32), a);
  LOG(INFO) << c;
  CastSimplify(&c);
  LOG(INFO) << c;
  ASSERT_EQ(utils::GetStreamCnt(c), "1");
  ASSERT_EQ(c.type(), Int(32));
}

TEST(CastSimplify, Imm_double) {
  Expr a = ir::Cast::Make(Float(64), Expr(2.33));
  Expr c = ir::Cast::Make(Int(32), a);
  LOG(INFO) << c;
  CastSimplify(&c);
  LOG(INFO) << c;
  ASSERT_EQ(utils::GetStreamCnt(c), "2");
  ASSERT_EQ(c.type(), Int(32));
}

}  // namespace cinn::optim
