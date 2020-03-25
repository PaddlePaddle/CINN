#include "cinn/common/cas.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/common/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace common {

using common::make_const;
using utils::GetStreamCnt;
using utils::Join;
using utils::Trim;
using namespace ir;  // NOLINT

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

TEST(CAS, number_cal) {
  // 1 * 100 * -1 + 0 + 1001
  auto u1 = Sum::Make({Product::Make({Expr(1), Expr(100), Expr(-1)}), Expr(0), Expr(1001)});
  LOG(INFO) << u1;
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

TEST(CAS, SimplifySum) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));
  // x + y + z + 0
  auto u1 = Sum::Make({x, y, z, make_const(0)});
  // x*1 + y + z + 0
  auto u2 = Sum::Make({Product::Make({x, Expr(1)}), y, z, make_const(0)});
  // z + 1 + y + x + zx
  auto u3 = CasSimplify(Sum::Make({z, Expr(1), y, x, Product::Make({z, x})}));
  // z + 1 + y + 3 + x + 0 + zx
  auto u4 = CasSimplify(Sum::Make({z, Expr(1), y, Expr(3), x, Expr(0), Product::Make({z, x})}));
  // x2 + 3zy + -3*yz + -2x + 1
  auto u5 = CasSimplify(Sum::Make({Product::Make({x, Expr(2)}),
                                   Product::Make({z, y, Expr(3)}),
                                   Product::Make({Expr(-3), y, z}),
                                   Product::Make({Expr(-2), x}),
                                   Expr(1)}));

  EXPECT_EQ(GetStreamCnt(CasSimplify(u1)), "(x + y + z)");
  EXPECT_EQ(GetStreamCnt(CasSimplify(u2)), "(x + y + z)");
  EXPECT_EQ(GetStreamCnt(u3), "(1 + x + y + z + (x * z))");
  EXPECT_EQ(GetStreamCnt(u4), "(4 + x + y + z + (x * z))");
  EXPECT_EQ(GetStreamCnt(u5), "1");
}

TEST(CAS, SimplifyProduct) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  // x * x^-1
  auto u1 = CasSimplify(Product::Make({x, Power::Make(x, Expr(-1))}));
  // zyx*(-1)
  auto u2 = CasSimplify(Product::Make({z, y, x, Expr(-1)}));
  // x^2*y*z*x*x*x^-5
  auto u3 = CasSimplify(Product::Make({Power::Make(x, Expr(2)), y, z, x, x, Power::Make(x, Expr(-5))}));
  // x^(4/2) * x^3
  auto u4 = CasSimplify(Product::Make({Power::Make(x, FracOp::Make(Expr(4), Expr(2))), Power::Make(x, Expr(3))}));

  EXPECT_EQ(GetStreamCnt(u1), "1");
  EXPECT_EQ(GetStreamCnt(u2), "(-1 * x * y * z)");
  EXPECT_EQ(GetStreamCnt(u3), "((x^-1) * y * z)");
  EXPECT_EQ(GetStreamCnt(u4), "(x^5)");
  LOG(INFO) << u4;
}

TEST(CAS, SimplifyMod) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  // 2*x % 2 = 0
  auto u1 = CasSimplify(Mod::Make(Product::Make({x, Expr(2)}), Expr(2)));
  // (x+y+z) % 2 = x%2 + y%2 + z%2
  auto u2 = CasSimplify(Mod::Make(Sum::Make({x, y, z}), Expr(2)));
  // x%2 + 1%2 + x%2
  auto u3 = CasSimplify(Sum::Make({Mod::Make(x, Expr(2)), Mod::Make(Expr(1), Expr(2)), Mod::Make(x, Expr(2))}));
  // x^3 + x % 5 + y + 1 + (4*x)%5
  auto u4 = CasSimplify(Sum::Make(
      {Power::Make(x, Expr(3)), Mod::Make(x, Expr(5)), y, Expr(1), Mod::Make(Product::Make({x, Expr(4)}), Expr(5))}));

  EXPECT_EQ(GetStreamCnt(u1), "0");
  EXPECT_EQ(GetStreamCnt(u2), "((x % 2) + (y % 2) + (z % 2))");
  EXPECT_EQ(GetStreamCnt(u3), "1");
  EXPECT_EQ(GetStreamCnt(u4), "(1 + (x^3) + y)");
}

TEST(CAS, ConvertCinnToCAS) {
  Placeholder<float> A("A", {10, 10});
  Placeholder<float> B("B", {10, 10});

  auto C =
      // A(i,j) + 0 + 1 + 2 B(i,j) + 0 B(i,j) A(i,j)
      Compute({10, 10}, [&](Expr i, Expr j) { return A(i, j) + 0.f + 1.f + 2.f * B(i, j) + 0.f * B(i, j) * A(i, j); });

  Expr body = C->body();
  LOG(INFO) << "body " << body;

  body = detail::ConvertCinnToCAS(body);
  body = CasSimplify(body);
  EXPECT_EQ(GetStreamCnt(body), "(1 + A[((i * 10) + j)] + (2 * B[((i * 10) + j)]))");
  body = detail::ConvertCasToCinn(body);
  LOG(INFO) << "convert back " << body;
  EXPECT_EQ(GetStreamCnt(body), "(1 + (A[((i * 10) + j)] + (2 * B[((i * 10) + j)])))");
}

TEST(CAS, FracOp) {
  Var x = ir::_Var_::Make("x", Int(32));
  Var y = ir::_Var_::Make("y", Int(32));
  Var z = ir::_Var_::Make("z", Int(32));

  auto u1 = AutoSimplify(Div::Make(Expr(1), x) * x);
  EXPECT_EQ(GetStreamCnt(u1), "1");
  // 64x/32 + y + 64/32
  auto u2 = AutoSimplify(Expr(64) * x / Expr(32) + y + Expr(64) / Expr(32));
  EXPECT_EQ(GetStreamCnt(u2), "(2 + ((2 * x) + y))");
  // 1/32 * y * z * 32768 * 2
  auto u3 = AutoSimplify(Expr(1) / Expr(32) * y * z * 32768 * 2);
  EXPECT_EQ(GetStreamCnt(u3), "(2048 * (y * z))");
  // 32768 * (32x + y) + y
  auto u4 = AutoSimplify(Expr(32768) * (((Expr(32) * x) + y) / 32));
  LOG(INFO) << u4;
  EXPECT_EQ(GetStreamCnt(u4), "((32768 * x) + (1024 * y))");
}

}  // namespace common
}  // namespace cinn
