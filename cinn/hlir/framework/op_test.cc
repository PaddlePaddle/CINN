#include "cinn/hlir/framework/op.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "cinn/cinn.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
namespace framework {

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

TEST(Operator, GetAttr) {
  auto add      = Operator::Get("add");
  Operator temp = *add;
  auto strategy = Operator::GetAttr<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attr;
  std::vector<ir::Tensor> inputs{A, B};
  common::Type type;
  common::Target target;
  target.arch = common::Target::Arch::X86;
  auto impl   = OpStrategy::SelectImpl(strategy[add](attr, inputs, type, target));

  common::CINNValuePack cinn_input = common::_CINNValuePack_::Make({common::CINNValue(A), common::CINNValue(B)});
  common::CINNValuePack C          = impl->fcompute(cinn_input);
  C                                = impl->fschedule(C);
  for (int i = 0; i < C.get()->size(); i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("add1", inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  ASSERT_EQ(impl->name, "strategy.add.x86");
  ASSERT_EQ(add->description, "Add two tensors");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
