#include "cinn/optim/replace_call_with_expr.h"

#include <gtest/gtest.h>

#include "cinn/ir/buffer.h"
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
  ir::Buffer A_buf = ir::_Buffer_::Make("A", Float(32));

  auto A_value       = ir::Add::Make(ir::Call::Make(Float(32), "A", {i, j, k}, ir::Call::Halide),
                               ir::Call::Make(Float(32), "B", {i, j, k}, ir::Call::Halide));
  tuple_to_expr["A"] = ir::Store::Make(Expr(A_buf), A_value, Expr(i) * 100 * 100 + Expr(j) * 100 + Expr(k));
  // tuple_to_expr["B"] = ir::Store::Make(Expr(A_buf), B_value, Expr(i) * 100 * 100 + Expr(j) * 100 + Expr(k));

  isl::ctx ctx = Context::Global().isl_ctx();
  auto A       = Stage::New(isl::set(ctx, "{ A[i,j,k]: 0<i,j,k<100 }"));

  Iterator A_i0, A_i1;

  std::tie(A_i0, A_i1) = A->Split(Iterator("i"), 4);

  auto schedule = CreateSchedule({A.get()});

  AstGen gen(isl::set(ctx, "{:}"), {A.get()}, schedule->groups[0]);
  gen.SetIteratorNames({"i.outer", "i.inner", "j", "k"});
  isl::ast_node ast = gen.Build();

  Expr gened_expr;
  IslAstNodeToCinnExpr(ast, &gened_expr);
  LOG(INFO) << "gened expr " << gened_expr;

  LOG(INFO) << "A axis";
  for (auto &item : gen.axis2ast("A")) {
    LOG(INFO) << item.first << ": " << item.second;
  }

  for (auto &statement : tuple_to_expr) {
    auto axis_ast_map         = gen.axis2ast(statement.first);
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    std::map<std::string, Expr> axis;
    for (auto &item : axis_ast_map) {
      IslAstExprToCinnExpr(item.second, &axis[item.first]);
      // LOG(INFO) << "axis: " << item.first << " " << axis[item.first];
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
      poly_for (1, (k <= 99), 1)
      {
        A[((((((4 * i.outer) + i.inner) * 100) * 100) + (j * 100)) + k)] = (A(((4 * i.outer) + i.inner), j, k) + B(((4 * i.outer) + i.inner), j, k))
      }
    }
  }
}
)ROC";
  auto out           = utils::GetStreamCnt(gened_expr);
  EXPECT_EQ(out, utils::Trim(target));

  std::cout << "output:" << std::endl << out << std::endl;
}

}  // namespace optim
}  // namespace cinn
