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

TEST(Operator, GetAttrs) {
  auto add      = Operator::Get("add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attrs;
  std::vector<ir::Tensor> inputs{A, B};
  std::vector<Type> type{Float(32)};
  common::Target target;
  target.arch = common::Target::Arch::X86;
  auto impl   = OpStrategy::SelectImpl(strategy[add](attrs, inputs, type, target));

  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A), common::CINNValue(B)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  ASSERT_EQ(rets.size(), 2UL);
  rets = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    ir::Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("add1", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  ASSERT_EQ(impl->name, "strategy.add.x86");
  ASSERT_EQ(add->description, "Add two tensors");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
