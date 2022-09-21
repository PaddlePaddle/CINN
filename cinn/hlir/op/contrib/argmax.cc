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

#include "cinn/hlir/op/contrib/argmax.h"

#include <iostream>
#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/contrib/sort.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_schedule.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using framework::shape_t;
using ir::Tensor;

Tensor Argmax(const Tensor &in_tensor,
              const common::Target &target,
              poly::StageMap stages,
              const int &axis,
              const bool &keep_dims,
              const std::string &name) {
  auto shape = in_tensor->shape;
  auto ndim  = shape.size();
  CHECK_GT(ndim, 0) << "tensor's dim must be more than 0";

  int pos_axis = axis;
  if (axis < 0) {
    pos_axis = static_cast<int>(ndim) + axis;
  }
  CHECK_LT(pos_axis, ndim) << "Axis must be less than tensor's dim";
  CHECK_GE(pos_axis, 0) << "Axis must be more than 0";

  std::vector<Expr> output_shape;
  for (int i = 0; i < shape.size(); ++i) {
    CHECK(shape[i].is_constant()) << "Input tensor's shape should be constant value.";
    if (axis == i) {
      if (keep_dims) {
        output_shape.push_back(Expr(1));
      }
    } else {
      output_shape.push_back(shape[i]);
    }
  }
  if (output_shape.empty()) {
    output_shape.push_back(Expr(1));
  }

  auto sort_index = ArgSort(in_tensor, target, stages, pos_axis, false, name + "_index");
  auto res        = Compute(
      output_shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> eval_indices(indices);
        if (!keep_dims) {
          eval_indices.insert(eval_indices.begin() + pos_axis, Expr(0));
        } else {
          eval_indices[pos_axis] = Expr(0);
        }
        return sort_index(eval_indices);
      },
      name);
  stages->InsertLazily(sort_index);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForArgmax(const framework::NodeAttr &attrs,
                                                         const std::vector<Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  int axis;
  bool keep_dims = false;

  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  } else {
    LOG(FATAL) << "reduce dimension is not set!";
  }
  if (attrs.attr_store.count("keep_dim")) {
    keep_dims = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }

  framework::CINNCompute argmax_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of argmax compute is empty! Please check.";
    common::CINNValuePack arg_packs = args[0];
    std::string tensor_name         = UniqName("Argmax_out");
    CHECK_EQ(arg_packs.size(), 1U) << "There should be 1 input args for argmax compute";
    Expr in_expr = arg_packs[0];
    CHECK(in_expr.as_tensor());
    Tensor in_tensor = in_expr.as_tensor_ref();
    auto stages      = CreateStages({in_tensor});
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }
    auto out_tensor = Argmax(in_tensor, target, stages, axis, keep_dims, tensor_name);

    stages->InsertLazily(out_tensor);
    std::vector<CINNValue> cinn_values{CINNValue(out_tensor), CINNValue(stages)};
    *ret = common::CINNValuePack{cinn_values};
  });

  framework::CINNSchedule argmax_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of argmax schedule is empty! Please check.";
    common::CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr out = arg_pack[0];
    CHECK(out.as_tensor());

    // When develop FLAGS_cinn_ir_schedule=true case, we should run unit test with
    // FLAGS_cinn_ir_schedule=1
    if (FLAGS_cinn_ir_schedule) {
      *ret = common::CINNValuePack{{common::CINNValue(out)}};
    } else {
      poly::StageMap stages = arg_pack[arg_pack.size() - 1];
      *ret                  = common::CINNValuePack{{common::CINNValue(out), common::CINNValue(stages)}};
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(argmax_compute, argmax_schedule, "strategy.argmax.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForArgmax(const std::vector<shape_t> &inputs_shape,
                                         const framework::AttrMapType &attrs) {
  CHECK(inputs_shape.size() == 1UL);
  auto ndim = inputs_shape[0].size();
  CHECK_GT(ndim, 0) << "tensor's dim must be more than 0";
  int axis;
  bool keep_dim;

  CHECK(attrs.find("axis") != attrs.end());
  axis = absl::get<int>(attrs.at("axis"));
  if (axis < 0) {
    axis = static_cast<int>(ndim) + axis;
  }
  CHECK_LT(axis, ndim) << "Axis must be less than tensor's dim";
  CHECK_GE(axis, 0) << "Axis must be more than 0";

  CHECK(attrs.find("keep_dim") != attrs.end());
  keep_dim = absl::get<bool>(attrs.at("keep_dim"));

  std::vector<int> out_shapes;
  for (size_t i = 0; i < ndim; ++i) {
    if (axis == i) {
      if (keep_dim) {
        out_shapes.push_back(1);
      }
    } else {
      out_shapes.push_back(inputs_shape[0][i]);
    }
  }

  if (keep_dim) {
    CHECK_EQ(ndim, out_shapes.size());
  } else {
    CHECK_EQ(ndim - 1, out_shapes.size());
  }
  if (out_shapes.empty()) {
    out_shapes.push_back(1);
  }

  return {out_shapes};
}

std::vector<Type> InferDtypeForArgmax(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {Int(32)};
}

std::vector<std::vector<std::string>> InferLayoutForArgmax(const std::vector<framework::shape_t> &input_shapes,
                                                           const std::vector<std::string> &input_layouts,
                                                           const framework::NodeAttr &attrs,
                                                           const Target &target) {
  CHECK_EQ(input_shapes.size(), 1U) << "The input's shape size is not 1! Please check again.";
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layout size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(argmax_ops) {
  CINN_REGISTER_OP(argmax)
      .describe("This operator implements the op argmax.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForArgmax)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForArgmax))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForArgmax))
      .set_support_level(4);

  return true;
}
