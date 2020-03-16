#include "cinn/common/arithmatic.h"
#include <ginac/ginac.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace common {

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

}  // namespace common
}  // namespace cinn
