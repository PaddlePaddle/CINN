#include "cinn/poly/compute_at_transform.h"
#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(ComputeAtTransform, basic) {
  // two computation
  // s0: A(i,j)
  // s1: B(i,j,k) = A(i-1,j-1) + A(i,j) + A(i+1, j+1)

  isl::ctx ctx(isl_ctx_alloc());

  isl::set pdomain(ctx, "{ s0[i,j]: 1 <=i,j < 100 }");
  isl::set cdomain(ctx, "{ s1[i,j,k]: 0 < i,j,k<10 }");
  isl::map ptransform(ctx, "{ s0[i,j] -> s0[i/4,i%4,j] }");

  std::vector<isl::map> accesses;
  accesses.push_back(isl::map(ctx, "{ s1[i,j,k] -> s0[i-1,j-1] }"));
  accesses.push_back(isl::map(ctx, "{ s1[i,j,k] -> s0[i,j] }"));
  accesses.push_back(isl::map(ctx, "{ s1[i,j,k] -> s0[i+1,j+1] }"));

  ComputeAtTransform transform(pdomain, cdomain, accesses, ptransform, 1);  // j
  LOG(INFO) << "adjusted domain: " << transform.adjusted_pdomain();
  LOG(INFO) << "adjusted transform: " << transform.adjusted_ptransform();
}

}  // namespace poly
}  // namespace cinn
