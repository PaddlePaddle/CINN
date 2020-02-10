#include "cinn/poly/ast_gen.h"

#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(ast_gen, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  Element A(isl::set(ctx, "{ A[i,j,k]: 0<i,j,k<100 }"));
  Element B(isl::set(ctx, "{ B[i,j,k]: 0<i,j,k<100 }"));

  Scheduler scheduler;
  scheduler.RegisterElement(A);
  scheduler.RegisterElement(B);
  scheduler.After(A, B, 2);

  AstGen gen(isl::set(ctx, "{:}"));
  gen.SetIteratorNames({"i", "j", "k"});
  gen({A, B}, scheduler);
}

}  // namespace poly
}  // namespace cinn
