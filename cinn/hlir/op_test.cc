#include <gtest/gtest.h>
#include <string>

#include "cinn/cinn.h"

#include "cinn/hlir/op.h"

namespace cinn {
CINN_REGISTRY_ENABLE(cinn::hlir::Operator);
namespace hlir {
CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

TEST(Operator, GetAttr) {
  auto add  = Operator::Get("add");
  auto nick = Operator::GetAttr<std::string>("nick_name");
  ASSERT_EQ(add->description, "test of op Add");
  ASSERT_EQ(nick[add], "plus");
}
}  // namespace hlir
}  // namespace cinn
