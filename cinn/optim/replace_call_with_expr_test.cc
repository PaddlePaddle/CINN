#include "cinn/optim/replace_call_with_expr.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace optim {

using namespace poly;

TEST(ReplaceCallWithExpr, basic) {
  // Define A and B statement.
  std::map<std::string, Expr> tuple_to_expr;
  Var i("i"), j("j"), k("k");
  Var A_buf("A_buf");

  auto A_value       = ir::Add::Make(ir::Call::Make(Float(32), "A", {i, j, k}, ir::Call::Halide),
                               ir::Call::Make(Float(32), "B", {i, j, k}, ir::Call::Halide));
  auto B_value       = ir::Add::Make(ir::Call::Make(Float(32), "B", {i, j, k}, ir::Call::Halide), Expr(1.f));
  tuple_to_expr["A"] = ir::Store::Make(A_buf, A_value, Expr(i) * 100 * 100 + Expr(j) * 100 + Expr(k));
  tuple_to_expr["B"] = ir::Store::Make(A_buf, B_value, Expr(i) * 100 * 100 + Expr(j) * 100 + Expr(k));

  isl::ctx ctx(isl_ctx_alloc());
  Element A(isl::set(ctx, "{ A[i,j,k]: 0<i,j,k<100 }"));
  Element B(isl::set(ctx, "{ B[i,j,k]: 0<i,j,k<100 }"));

  Iterator A_i0, A_i1;
  Iterator B_i0, B_i1;

  std::tie(A_i0, A_i1) = A.Split(Iterator("i"), 4);
  std::tie(B_i0, B_i1) = B.Split(Iterator("i"), 4);

  Scheduler scheduler;
  scheduler.RegisterElement(A);
  scheduler.RegisterElement(B);
  scheduler.After(A, B, 3);

  AstGen gen(isl::set(ctx, "{:}"), {A, B}, scheduler);
  gen.SetIteratorNames({"i.outer", "i.inner", "j", "k"});
  isl::ast_node ast = gen.Build();

  Expr gened_expr;
  IslAstNodeToCinnExpr(ast, &gened_expr);
  LOG(INFO) << "gened expr " << gened_expr;

  LOG(INFO) << "A axis";
  for (auto &item : gen.axis2ast("A")) {
    LOG(INFO) << item.first << ": " << item.second;
  }
  LOG(INFO) << "B axis";
  for (auto &item : gen.axis2ast("A")) {
    LOG(INFO) << item.first << ": " << item.second;
  }

  for (auto &statement : tuple_to_expr) {
    auto axis_ast_map         = gen.axis2ast(statement.first);
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    std::map<std::string, Expr> axis;
    for (auto &item : axis_ast_map) {
      IslAstExprToCinnExpr(item.second, &axis[item.first]);
      LOG(INFO) << "axis: " << item.first << " " << axis[item.first];
    }
    ReplaceCallWithExpr(&gened_expr, statement.first, statement_candi_expr, axis);
  }

  std::string target = R"ROC(
poly_for (0, (i.outer <= 24), 1)
{
  poly_for (max(0, ((-4 * i.outer) + 1)), (i.inner <= 3), 1)
  {
    poly_for (1, (j <= 99), 1)
    {
      {
        poly_for (1, (k <= 99), 1)
        {
          A_buf[((((((4 * i.outer) + i.inner) * 100) * 100) + (j * 100)) + k)] = (A(((4 * i.outer) + i.inner), j, k) + A_buf[((((((4 * i.outer) + i.inner) * 100) * 100) + (j * 100)) + k)] = (B(((4 * i.outer) + i.inner), j, k) + 1))
        }
        poly_for (1, (k <= 99), 1)
        {
          A_buf[((((((4 * i.outer) + i.inner) * 100) * 100) + (j * 100)) + k)] = (B(((4 * i.outer) + i.inner), j, k) + 1)
        }
      }
    }
  }
}
)ROC";
  EXPECT_EQ(utils::GetStreamCnt(gened_expr), utils::Trim(target));

  LOG(INFO) << "\n" << gened_expr;
}

}  // namespace optim
}  // namespace cinn
