// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/types/optional.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {

std::shared_ptr<OpStrategy> StrategyForSelect(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const std::vector<std::vector<int>> &output_shapes,
                                              const Target &target) {
  framework::CINNCompute select_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of select compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 4U);
    Expr condition   = pack_args[0];
    Expr true_value  = pack_args[1];
    Expr false_value = pack_args[2];
    CHECK(condition.as_tensor());
    CHECK(true_value.as_tensor());
    CHECK(false_value.as_tensor());
    CHECK(pack_args[3].is_string());
    std::string tensor_name = pack_args[3].operator std::string();

    auto out =
        pe::Select(condition.as_tensor_ref(), true_value.as_tensor_ref(), false_value.as_tensor_ref(), tensor_name);
    auto stages =
        CreateStages({condition.as_tensor_ref(), true_value.as_tensor_ref(), false_value.as_tensor_ref(), out});
    *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of select op is empty! Please check.";
  strategy->AddImpl(select_compute, GetInjectiveScheduleFunc(output_shapes, target, false), "strategy.select.x86", 1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForSelect(const std::vector<framework::shape_t> &inputs_shape,
                                                    const framework::AttrMapType &attrs) {
  CHECK_GE(inputs_shape.size(), 3) << "The input's shape size is 0! Please check again.";
  CHECK(inputs_shape[0].size() == inputs_shape[1].size() && inputs_shape[1].size() == inputs_shape[2].size())
      << "input tensors n_dim is not equal!";
  CHECK(inputs_shape[0] == inputs_shape[1] && inputs_shape[1] == inputs_shape[2])
      << "input tensor shapes is not equal!";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForSelect(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_GE(inputs_type.size(), 3) << "The input's type size is less than three! Please check again.";
  CHECK(inputs_type[0].is_bool()) << "The condition tensor type should be bool";
  CHECK_EQ(inputs_type[1], inputs_type[2]) << "The true or false tensor type should be equal";
  std::vector<Type> res{inputs_type[1]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ternary_ops) {
  CINN_REGISTER_OP(select)
      .describe("This operator implements the meta op 'Select'.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSelect)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSelect))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSelect))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
