#include "cinn/common/arithmatic.h"

#include <ginac/ginac.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/common/ir.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace common {
using utils::GetStreamCnt;
using utils::Join;
using utils::Trim;
using namespace ir;  // NOLINT

TEST(GiNaC, simplify) {
  using namespace GiNaC;  // NOLINT
  symbol x("x");
  symbol y("y");

  ex e = x * 0 + 1 + 2 + 3 - 100 + 30 * y - y * 21 + 0 * x;
  LOG(INFO) << "e: " << e;
}

TEST(GiNaC, diff) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");
  ex e  = (x + 1);
  ex e1 = (y + 1);

  e  = diff(e, x);
  e1 = diff(e1, x);
  LOG(INFO) << "e: " << eval(e);
  LOG(INFO) << "e1: " << eval(e1);
}

TEST(GiNaC, solve) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");

  lst eqns{2 * x + 3 == 19};
  lst vars{x};

  LOG(INFO) << "solve: " << lsolve(eqns, vars);
  LOG(INFO) << diff(2 * x + 3, x);
}

TEST(Solve, basic) {
  Var i("i", Int(32));
  Expr lhs = Expr(i) * 2;
  Expr rhs = Expr(2) * Expr(200);
  Expr res;
  bool is_positive;
  std::tie(res, is_positive) = Solve(lhs, rhs, i);
  LOG(INFO) << "res: " << res;
  EXPECT_TRUE(is_positive);
  EXPECT_TRUE(res == Expr(200));
}

TEST(Solve, basic1) {
  Var i("i", Int(32));
  Expr lhs = Expr(i) * 2;
  Expr rhs = Expr(2) * Expr(200) + 3 * Expr(i);

  Expr res;
  bool is_positive;
  std::tie(res, is_positive) = Solve(lhs, rhs, i);
  LOG(INFO) << "res " << res;
  EXPECT_TRUE(res == Expr(-400));
  EXPECT_FALSE(is_positive);
}

TEST(CAS, SimplifyPower_0) {
  {  // x^0 = 1
    Var x   = ir::_Var_::Make("x", Float(32));
    auto p0 = ir::Power::Make(x, Expr(0));
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "1");
  }
  {  // x^1 = x
    Var x   = ir::_Var_::Make("x", Float(32));
    auto p0 = ir::Power::Make(x, Expr(1));
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "x");
  }

  {  // 1^x = 1
    Var x   = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Power::Make(make_const(1), x);
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "1");
  }

  {  // 1.^x = 1.
    Var x   = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Power::Make(make_const(1.f), x);
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "1");
  }

  {  // 0^x = 0
    Var x   = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Power::Make(make_const(0), x);
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "0");
  }

  {  // 0.^x = 0
    Var x   = ir::_Var_::Make("x", Int(32));
    auto p0 = ir::Power::Make(make_const(0.f), x);
    LOG(INFO) << "p0 " << p0;
    auto p2 = detail::SimplifyPower(p0);
    LOG(INFO) << "simplified " << p2;
    EXPECT_EQ(GetStreamCnt(p2), "0");
  }
}

TEST(CAS, SimplifyPower) {
  Var x   = ir::_Var_::Make("x", Float(32));
  auto p0 = ir::Power::Make(x, Expr(2));
  LOG(INFO) << "p0 " << p0;
  auto p1 = ir::Power::Make(p0, Expr(3));

  LOG(INFO) << "power: " << p1;

  auto p2 = detail::SimplifyPower(p1);
  LOG(INFO) << "simplified: " << p2;
}

TEST(CAS, cmp) {
  detail::ExprPosCmp cmp;

  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  EXPECT_EQ(cmp(x, Expr(1)), false);
  EXPECT_EQ(cmp(Expr(1), x), true);

  // x * y * z > x * y
  EXPECT_EQ(cmp(ir::Product::Make({x, y, z}), ir::Product::Make({x, y})), false);
  // x * y * z > 10 * y * z
  EXPECT_EQ(cmp(ir::Product::Make({x, y, z}), ir::Product::Make({Expr(10), y, z})), false);
  // 1 * y * z < 10 * y * z
  EXPECT_EQ(cmp(ir::Product::Make({Expr(1), y, z}), ir::Product::Make({Expr(10), y, z})), true);
  // y^1 < y^x
  EXPECT_EQ(cmp(ir::Power::Make(y, Expr(1)), ir::Power::Make(y, x)), true);
  // y^1 < y^2
  EXPECT_EQ(cmp(ir::Power::Make(y, Expr(1)), ir::Power::Make(y, Expr(2))), true);
  // y * z^2 > x * z^1
  EXPECT_EQ(cmp(Product::Make({y, Power::Make(z, Expr(2))}), Product::Make({x, Power::Make(z, Expr(1))})), false);
  // 1*y^2 > y
  EXPECT_EQ(cmp(Product::Make({Expr(1), Power::Make(y, Expr(2))}), y), false);
  // 1*y^2 > x
  EXPECT_EQ(cmp(Product::Make({Expr(1), Power::Make(y, Expr(2))}), x), false);
  // 1*y^2 < z
  EXPECT_EQ(cmp(Product::Make({Expr(1), Power::Make(y, Expr(2))}), z), true);
}

}  // namespace common
}  // namespace cinn
