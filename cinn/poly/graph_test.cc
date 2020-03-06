#include "cinn/poly/graph.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/buffer.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::Halide);
  return expr;
}

Stage* CreateStage(const std::string& name, std::vector<Expr>& args, isl::set domain) {
  auto expr = CreateCall(name, args);
  return Stage::New(domain, expr).get();
}

TEST(CreateGraph, basic) {
  auto ctx = Context::Global().isl_ctx();
  // create call (for tensor);
  Var i("i"), j("j"), k("k");
  std::vector<Expr> args({Expr(i), Expr(j), Expr(k)});

  lang::Buffer A_arr(Float(32), "A"), B_arr(Float(32), "B"), C_arr(Float(32), "C");
  Expr A_call = CreateCall("A", args);
  Expr B_call = CreateCall("B", args);
  Expr C_call = CreateCall("C", args);

  // A[] = B[] + 1
  Expr A_expr = ir::Store::Make(Expr(A_arr.buffer()), Expr(1.f), Expr(i));
  Expr B_expr = ir::Store::Make(Expr(B_arr.buffer()), A_call + 1.f, Expr(i));
  Expr C_expr = ir::Store::Make(Expr(C_arr.buffer()), B_call + A_call, Expr(i));

  // create stages
  auto A_stage = Stage::New(isl::set(ctx, "{ A[i,j,k]: 0<=i,j,k<100 }"), A_expr);
  auto B_stage = Stage::New(isl::set(ctx, "{ B[i,j,k]: 0<=i,j,k<100 }"), B_expr);
  auto C_stage = Stage::New(isl::set(ctx, "{ C[i,j,k]: 0<=i,j,k<100 }"), C_expr);

  auto graph = CreateGraph({A_stage.get(), B_stage.get(), C_stage.get()});
  LOG(INFO) << "graph:\n" << graph->Visualize();
}

}  // namespace poly
}  // namespace cinn
