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

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/layout.h"

namespace cinn {
namespace hlir {
namespace op {

std::shared_ptr<OpStrategy> StrategyForBroadcastTo(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target) {
  auto out_shape = GetAttr<std::vector<int>>(attrs.attr_store, "out_shape", {});
  CHECK(out_shape.size());
  auto broadcast_axes = GetAttr<std::vector<int>>(attrs.attr_store, "broadcast_axes", {});
  CHECK(broadcast_axes.size());

  framework::CINNCompute broadcast_to_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of broadcast_to compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "The input tensors of broadcast_to compute is empty! Please check.";

    CHECK_GE(pack_args.size(), 2U);
    CHECK(pack_args[1].is_string());
    std::string tensor_name = pack_args[1].operator std::string();

    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out     = pe::BroadcastTo(A, out_shape, broadcast_axes, tensor_name);
    auto stages  = CreateStages({A, out});
    *ret         = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      broadcast_to_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.broadcast_to.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForBroadcastTo(const std::vector<shape_t> &inputs_shape,
                                              const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL) << "input_shape size should be one. Please Check.";
  auto out_shape = GetAttr<std::vector<int>>(attrs, "out_shape", {});
  CHECK(out_shape.size());
  auto broadcast_axes = GetAttr<std::vector<int>>(attrs, "broadcast_axes", {});
  CHECK(broadcast_axes.size());

  CHECK_EQ(inputs_shape[0].size(), broadcast_axes.size())
      << "broadcast_axes's size should be same with the input shape's size";
  CHECK_GE(out_shape.size(), broadcast_axes.size()) << "broadcast_axes's size should be no more than out_shape's size";

  return {out_shape};
}

std::vector<Type> InferDtypeForBroadcastGrad(const std::vector<Type> &inputs_type,
                                             const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 3UL);
  std::vector<Type> out_type{inputs_type[1], inputs_type[2]};
  return out_type;
}

std::vector<std::vector<std::string>> InferLayoutForBroadcastTo(const std::vector<std::vector<int>> &input_shapes,
                                                                const std::vector<std::string> &input_layouts,
                                                                const framework::NodeAttr &attrs,
                                                                const Target &target) {
  CHECK(input_layouts.size() == 1U) << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> out_layouts = {""};
  if (attrs.attr_store.count("out_layouts")) {
    out_layouts = absl::get<std::vector<std::string>>(attrs.attr_store.at("out_layouts"));
  }
  return {out_layouts, input_layouts};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(broadcast_to_op) {
  CINN_REGISTER_OP(broadcast_to)
      .describe("broadcast one tensor to the target shape")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBroadcastTo)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBroadcastTo))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBroadcast))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForBroadcastTo))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast)
      .set_support_level(4);

  return true;
}
