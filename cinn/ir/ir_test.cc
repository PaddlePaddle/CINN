#include "cinn/ir/ir.h"
#include <gtest/gtest.h>

#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

TEST(ir, Add) {
  Builder builder;

  auto one = builder.make<IntImm>(Int(32), 1);
  auto two = builder.make<IntImm>(Int(32), 2);

  auto add = builder.make<Add>(one, two);

  auto cnt = utils::GetStreamCnt(add.node_type());
  ASSERT_EQ(cnt, "<node: Add>");
}

}  // namespace ir
}  // namespace cinn
