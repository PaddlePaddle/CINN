#include "cinn/poly/element.h"

#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(Element, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  Element ele(domain);
  ele.Split(Iterator("i"), 4);
}

}  // namespace poly
}  // namespace cinn
