#include "cinn/poly/isl_utils.h"
#include <gtest/gtest.h>

namespace cinn::poly {

TEST(isl_utils, isl_set_axis_has_noparam_constant_bound) {
  isl_ctx* ctx = isl_ctx_alloc();
  {
    isl::set set(ctx, "{ s[i] : 0 < i < 2 }");
    ASSERT_TRUE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }

  {
    isl::set set(ctx, "[n] -> { s[i] : 0 < i < n }");
    ASSERT_FALSE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }

  {
    isl::set set(ctx, "[unused] -> { s[i] : 0 < i < 10 }");
    ASSERT_TRUE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }
}

}  // namespace cinn::poly