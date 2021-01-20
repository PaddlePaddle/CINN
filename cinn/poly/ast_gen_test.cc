#include "cinn/poly/ast_gen.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

TEST(TransIdentityExtentToContextId, basic) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "{ s[i,j=0,k] : 0<=i<12 and 12<k<32 }");
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;

  ASSERT_EQ(utils::GetStreamCnt(new_set),
            "[_const_0] -> { s[i, j, k] : _const_0 <= 1 and 0 <= i <= 11 and 0 <= j <= _const_0 and 13 <= k <= 31 }");
}

TEST(TransIdentityExtentToContextId, basic1) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "[n] -> { s[i,j=0,k] : 0<=i<n and 12<k<32 }");
  LOG(INFO) << "set: " << set;
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;
}

TEST(TransIdentityExtentToContextIdForSchedule, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::union_map map(
      ctx,
      "{ C[i, j, k0] -> [r = 0, t0 = 0, d0 = 0, t1 = 0, d1 = 0, t2, d2 = 0, t3 = i, d3 = 0, t4 = k0 - 4t2, d4 = 0, t5 "
      "= j, d5 = 0] : 0 <= i <= 15 and 0 <= j <= 15 and 0 <= k0 <= 15 and -3 + k0 <= 4t2 <= k0 }");

  LOG(INFO) << "map: " << map;
  LOG(INFO) << "res: " << TransIdentityExtentToContextIdForSchedule(map);
}

}  // namespace poly
}  // namespace cinn
