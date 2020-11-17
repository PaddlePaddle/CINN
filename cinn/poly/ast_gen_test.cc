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

}  // namespace poly
}  // namespace cinn
