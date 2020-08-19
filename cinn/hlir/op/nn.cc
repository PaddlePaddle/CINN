#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
namespace op {
using common::CINNValue;
using common::CINNValuePack;
using common::CINNValuePackShared;
using framework::OpStrategy;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForAdd(const framework::NodeAttr &attr,
                                           const std::vector<ir::Tensor> &inputs,
                                           Type out_type,
                                           const Target &target) {
  framework::CINNCompute add_compute([](ir::Args args, ir::RetValue *ret) {
    CINNValuePackShared a = args[0];
    ir::Expr A            = a[0];
    ir::Expr B            = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    *ret =
        CINNValuePack::Make({CINNValue(ir::Expr(pe::Add(A.as_tensor_ref(), B.as_tensor_ref(), UniqName("C")).get()))});
  });

  framework::CINNSchedule add_schedule([](ir::Args args, ir::RetValue *ret) {
    CINNValuePackShared a = args[0];
    ir::Expr A            = a[0];
    *ret                  = CINNValuePack::Make({CINNValue(A)});
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(add_compute, add_schedule, "strategy.add.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(nn_ops) {
  CINN_REGISTER_OP(add)
      .describe("Add two tensors")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForAdd)
      .set_support_level(4);
}
