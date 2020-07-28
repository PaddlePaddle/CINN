#include <gtest/gtest.h>
#include <string>

#include "cinn/cinn.h"

#include "cinn/hlir/op.h"
namespace dmlc {
DMLC_REGISTRY_ENABLE(cinn::hlir::Op);
}
namespace cinn {
namespace hlir {
HLIR_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

TEST(Op, GetAttr) {
  auto add  = Op::Get("add");
  auto nick = Op::GetAttr<std::string>("nick_name");
  ASSERT_EQ(add->description, "test of op Add");
  ASSERT_EQ(nick[add], "plus");
}
}  // namespace hlir
}  // namespace cinn
