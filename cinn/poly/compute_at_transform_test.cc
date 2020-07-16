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

TEST(ComputeAtTransform2, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set pdomain(ctx, "{ p[i,j]: 0<=i,j<100 }");
  isl::map ptransform(ctx, "{ p[i,j]->p[t0,t1,t2]: t0=i%4 and t1=i/4 and t2=j }");
  isl::set cdomain(ctx, "{ c[i,j,k]: 0<=i,j,k<50 }");
  isl::map ctransform(ctx, "{ c[i,j,k]->c[t0,t1,t2,t3]: t0=i/4 and t1=i%4 and t2=j and t3=k }");

  isl::map access(ctx, "{ c[i,j,k]->p[i,j]; c[i,j,k]->p[i+1,j]; c[i,j,k]->p[i-1,j] }");

  poly::ComputeAtTransform2 t(pdomain, cdomain, access, ptransform, ctransform, 1);
  t();

  t.DisplayC();

  isl::map pschedule(ctx, "{ p[i0,i1,i2,i3,i4] -> [t0,t1,t1t, t2,t3,t4,t5]: t0=i0 and t1=i1 and t2=i2 and t3=i3 and t4=i4 and t5=0 and t1t=0 }");
  isl::map cschedule(ctx, "[_c_0,_c_1] -> { c[i0,i1,i2,i3] -> [t0,t1,t1t,t2,t3,t4,t5]: t0=i0 and t1=i1 and t2=i2 and t3=i3 and t4=0 and t5=0 and t1t=1 }");

  t.DisplayC(pschedule.release(), cschedule.release());
}

}  // namespace poly
}  // namespace cinn
