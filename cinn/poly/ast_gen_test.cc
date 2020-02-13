#include "cinn/poly/ast_gen.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

TEST(ast_gen, basic) {
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

  auto iters = gen.axis2ast("A");
  for (auto& x : iters) {
    LOG(INFO) << x.first << " " << x.second;
  }

  ir::Expr e;
  IslAstNodeToCinnExpr(ast, &e);

  LOG(INFO) << "\n" << e;
}

}  // namespace poly
}  // namespace cinn
