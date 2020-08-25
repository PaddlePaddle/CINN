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
using common::CINNValue;
using common::CINNValuePack;
using lang::Args;
using lang::PackedFunc;
using lang::RetValue;

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

CINN_REGISTER_OP(_add_test_)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

std::shared_ptr<OpStrategy> StrategyTest(const NodeAttr &attr,
                                         const std::vector<ir::Tensor> &inputs,
                                         common::Type out_type,
                                         const common::Target &target) {
  PackedFunc::body_t compute_body = [](Args args, RetValue *ret) {
    CINNValuePack a = args[0];
    ir::Expr A      = a[0];
    ir::Expr B      = a[1];
    auto C          = pe::Add(A.as_tensor_ref(), B.as_tensor_ref(), "C");

    auto stages = poly::CreateStages({A.as_tensor_ref(), B.as_tensor_ref(), C});

    *ret = CINNValuePack{{CINNValue(ir::Expr(C.get())), CINNValue(stages)}};
  };
  PackedFunc fcompute(compute_body);

  PackedFunc::body_t schedule_body = [](Args args, RetValue *ret) {
    CHECK_EQ(args.size(), 2UL);
    CINNValuePack a       = args[0];
    ir::Expr A            = a[0];
    poly::StageMap stages = a[1];
    stages[A.as_tensor_ref()]->Vectorize(1, 16);
    stages[A.as_tensor_ref()]->Unroll(1);
    *ret = CINNValuePack{{CINNValue(A), CINNValue(stages)}};
  };
  PackedFunc fschedule(schedule_body);

  auto strategy = std::make_shared<OpStrategy>();

  if (target.arch == common::Target::Arch ::X86) {
    strategy->AddImpl(fcompute, fschedule, "test.strategy.x86", 10);
  } else {
    strategy->AddImpl(fcompute, fschedule, "test.strategy.else", 10);
  }
  strategy->AddImpl(fcompute, fschedule, "test.strategy.lowPlevel.x86", 5);
  return strategy;
}

TEST(Operator, GetAttrs) {
  auto add      = Operator::Get("_add_test_");
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

  CINNValuePack cinn_input = CINNValuePack{{CINNValue(A), CINNValue(B)}};
  CINNValuePack C          = impl->fcompute(cinn_input);
  ASSERT_EQ(C->size(), 2UL);
  poly::StageMap stages    = C[1];
  C                        = impl->fschedule(C, stages);
  for (int i = 0; i < C.get()->size() - 1; i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }

  auto func = Lower("_add_test_", stages, inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  ASSERT_EQ(impl->name, "strategy.add.x86");
  ASSERT_EQ(add->description, "Add two tensors");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
