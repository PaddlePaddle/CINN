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

#include "cinn/hlir/op/contrib/gather.h"

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
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
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

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor Gather(const ir::Tensor &A, const ir::Tensor &B, const int &axis, const std::string &name) {
  CHECK_EQ(A->shape.size(), B->shape.size());
  auto res = Compute(
      B->shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> A_indices(indices);
        A_indices[axis] = B(indices);
        return lang::Identity(A(A_indices));
      },
      name);
  return res;
}
ir::Tensor GatherNd(const ir::Tensor &A, const ir::Tensor &B, const std::vector<int> &axes, const std::string &name) {
  std::vector<Expr> out_shape = B->shape;
  out_shape.pop_back();
  auto res = Compute(
      out_shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> A_indices(indices.begin(), indices.begin() + A->shape.size());
        std::vector<Expr> B_indices(indices);
        for (int i = 0; i < axes.size(); ++i) {
          B_indices.push_back(Expr(i));
          A_indices[axes[i]] = B(B_indices);
          B_indices.pop_back();
        }
        return lang::Identity(A(A_indices));
      },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForGather(const framework::NodeAttr &attrs,
                                                         const std::vector<ir::Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("axis")) << "find no attr of axis";
  int axis = absl::get<int>(attr_store.at("axis"));
  std::string op_name("gather");

  framework::CINNCompute gather_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of " << op_name << " compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 2U) << "2 input tensors for " << op_name << " compute\n";
    Expr A = a[0];
    Expr B = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages   = CreateStages({tensor_A, tensor_B});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ") << ", B shape: " << utils::Join(tensor_B->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    ir::Tensor out = Gather(tensor_A, tensor_B, axis, UniqName("Gather_out"));
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.gather.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForGatherNd(const framework::NodeAttr &attrs,
                                                           const std::vector<ir::Tensor> &inputs,
                                                           const std::vector<Type> &out_type,
                                                           const std::vector<std::vector<int>> &output_shapes,
                                                           const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("axes")) << "find no attr of axes";
  std::vector<int> axes = absl::get<std::vector<int>>(attr_store.at("axes"));
  std::string op_name("gather_nd");

  framework::CINNCompute gather_nd_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of " << op_name << " compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 2U) << "2 input tensors for " << op_name << " compute\n";
    Expr A = a[0];
    Expr B = a[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages   = CreateStages({tensor_A, tensor_B});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ") << ", B shape: " << utils::Join(tensor_B->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    ir::Tensor out = GatherNd(tensor_A, tensor_B, axes, UniqName("GatherNd_out"));
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_nd_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.gather_nd.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForGather(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK_EQ(inputs_shape[0].size(), inputs_shape[1].size()) << "The inputs' dims should be equal.";
  std::vector<std::vector<int>> res{inputs_shape[1]};
  return res;
}

std::vector<std::vector<int>> InferShapeForGatherNd(const std::vector<std::vector<int>> &inputs_shape,
                                                    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<int> output_shape(inputs_shape[1].begin(), inputs_shape[1].end() - 1);
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForGather(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(gather_ops) {
  CINN_REGISTER_OP(gather)
      .describe("Gather.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForGather)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForGather))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForGather))
      .set_support_level(4);
  CINN_REGISTER_OP(gather_nd)
      .describe("GatherNd.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForGatherNd)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForGatherNd))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForGather))
      .set_support_level(4);

  return true;
}
