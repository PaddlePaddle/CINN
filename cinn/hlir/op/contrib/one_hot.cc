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

#include "cinn/hlir/op/contrib/one_hot.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {
ir::Tensor OneHot(const ir::Tensor& indices,
                  const ir::Tensor& on_value,
                  const ir::Tensor& off_value,
                  const int depth,
                  const int axis,
                  const std::string& dtype,
                  const std::string& output_name) {
  CHECK(axis == -1 || (axis >= 0 && axis <= indices->shape.size()))
      << "axis must be -1 or between 0 and " << indices->shape.size();
  CHECK_GT(depth, 0) << "Depth must be positive.";
  // TODO(SigureMo): Currently CINN does not support 0-D tensors, we use a 1-D tensor only has one element instead.
  CHECK(on_value->shape.size() == 1U && on_value->shape[0].as_int32() == 1U) << "On value must be a scalar.";
  CHECK(off_value->shape.size() == 1U && off_value->shape[0].as_int32() == 1U) << "Off value must be a scalar.";

  int true_axis = (axis == -1) ? indices->shape.size() : axis;

  // TODO(SigureMo): Get the value from 1-D tensor, it can be removed after CINN supports 0-D tensors.
  ir::Expr on_value_value  = on_value(Expr(0));
  ir::Expr off_value_value = off_value(Expr(0));
  ir::Expr on_value_cast   = ir::Cast::Make(common::Str2Type(dtype), on_value_value);
  ir::Expr off_value_cast  = ir::Cast::Make(common::Str2Type(dtype), off_value_value);

  std::vector<Expr> out_shape(indices->shape);
  out_shape.insert(out_shape.begin() + true_axis, Expr(depth));

  ir::Tensor res = lang::Compute(
      out_shape,
      [=](const std::vector<ir::Expr>& output_indices) {
        std::vector<ir::Expr> indices_indices;

        for (size_t i = 0; i < output_indices.size(); i++) {
          if (static_cast<int>(i) == true_axis) {
            continue;
          }
          indices_indices.push_back(output_indices[i]);
        }

        auto idx           = output_indices[true_axis];
        auto indices_value = ir::Cast::Make(idx.type(), indices(indices_indices));
        return ir::Select::Make(ir::EQ::Make(indices_value, idx), on_value_cast, off_value_cast);
      },

      common::UniqName(output_name));
  return res;
}

std::vector<framework::shape_t> InferShapeForOneHot(const std::vector<framework::shape_t>& inputs_shape,
                                                    const framework::AttrMapType& attrs) {
  CHECK_EQ(inputs_shape.size(), 3UL) << "The number of pool2d_grad's input should be 3";

  auto indices_shape   = inputs_shape[0];
  auto on_value_shape  = inputs_shape[1];
  auto off_value_shape = inputs_shape[2];

  CHECK_EQ(on_value_shape.size(), 1U) << "On value must be a scalar.";
  CHECK_EQ(off_value_shape.size(), 1U) << "Off value must be a scalar.";

  int depth;
  int axis;
  Type dtype;

  for (auto& iter : attrs) {
    if (iter.first == "depth") {
      depth = absl::get<int>(iter.second);
    } else if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
    } else if (iter.first == "dtype") {
      dtype = common::Str2Type(absl::get<std::string>(iter.second));
    }
  }

  int true_axis = (axis == -1) ? indices_shape.size() : axis;

  // TODO: more checks

  framework::shape_t out_shape(indices_shape);
  out_shape.insert(out_shape.begin() + true_axis, depth);
  return {out_shape};
}

std::vector<Type> InferDtypeForOneHot(const std::vector<Type>& inputs_type, const framework::AttrMapType& attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  Type dtype = common::Str2Type(absl::get<std::string>(attrs.at("dtype")));
  return {dtype};
}

std::shared_ptr<framework::OpStrategy> StrategyForOneHot(const framework::NodeAttr& attrs,
                                                         const std::vector<ir::Tensor>& inputs,
                                                         const std::vector<Type>& out_type,
                                                         const std::vector<std::vector<int>>& output_shapes,
                                                         const Target& target) {
  // check output shape
  CHECK(!output_shapes.empty() && !output_shapes[0].empty()) << "Output shape is empty! Please check.\n";
  auto indices_shape = inputs[0]->shape;

  int depth;
  int axis;
  std::string dtype("float32");

  for (auto& iter : attrs.attr_store) {
    if (iter.first == "depth") {
      depth = absl::get<int>(iter.second);
    } else if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
    } else if (iter.first == "dtype") {
      dtype = absl::get<std::string>(iter.second);
    }
  }

  CHECK(axis == -1 || (axis >= 0 && axis <= indices_shape.size()))
      << "axis must be -1 or between 0 and " << indices_shape.size();
  CHECK_GT(depth, 0) << "Depth must be positive.";

  framework::CINNCompute one_hot_compute([=](lang::Args args, lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input argument of one_hot compute is empty! Please check.\n";
    common::CINNValuePack value_args = args[0];
    CHECK(!value_args.empty()) << "at least one input tensor for transpose compute\n";
    Expr indices_expr   = value_args[0];
    Expr on_value_expr  = value_args[1];
    Expr off_value_expr = value_args[2];
    CHECK(indices_expr.as_tensor());
    CHECK(on_value_expr.as_tensor());
    CHECK(off_value_expr.as_tensor());

    ir::Tensor indices   = indices_expr.as_tensor_ref();
    ir::Tensor on_value  = on_value_expr.as_tensor_ref();
    ir::Tensor off_value = off_value_expr.as_tensor_ref();

    std::string tensor_name = UniqName("T_OneHot_out");
    // if (FLAGS_cinn_ir_schedule) {
    //   CHECK_EQ(input_args.size(), 2);
    //   tensor_name = input_args[1].operator std::string();
    // }

    auto out = OneHot(indices, on_value, off_value, depth, axis, dtype, tensor_name);
    CHECK(!out_type.empty()) << "Output type of Pool2dGrad is empty! Please check.\n";
    auto stages = CreateStages({indices, on_value, off_value});
    stages->InsertLazily(out);
    *ret = common::CINNValuePack{{common::CINNValue(out), common::CINNValue(stages)}};
  });

  framework::CINNSchedule one_hot_schedule([=](lang::Args args, lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input argument of one_hot schedule is empty! Please check.";
    common::CINNValuePack arg_pack = args[0];
    // TODO: FLAGS_cinn_ir_schedule
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr out = arg_pack[0];
    CHECK(out.as_tensor());

    // TODO: implements this
    // if (FLAGS_cinn_ir_schedule) {
    //   Expr padding_out = arg_pack[1];
    //   CHECK(padding_out.as_tensor());
    //   *ret = common::CINNValuePack{{common::CINNValue(out), common::CINNValue(padding_out)}};
    // } else {
    poly::StageMap stages = arg_pack[arg_pack.size() - 1];
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes[0], target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes[0], target);
    }
    *ret = common::CINNValuePack{{common::CINNValue(out), common::CINNValue(stages)}};
    // }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(one_hot_compute, one_hot_schedule, "strategy.one_hot.x86", 1);
  return strategy;
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(one_hot_ops) {
  CINN_REGISTER_OP(one_hot)
      .describe(
          "This operator compute a one-hot tensor where the locations repsented by indices take value on_value, "
          "otherwise take value off_value.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForOneHot)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForOneHot))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForOneHot))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  return true;
}