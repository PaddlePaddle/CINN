#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"

CINN_REGISTER_HELPER(nn_ops) {
  CINN_REGISTER_OP(add)
      .describe("Add")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<std::string>("add", "add")
      .set_support_level(4);
}
