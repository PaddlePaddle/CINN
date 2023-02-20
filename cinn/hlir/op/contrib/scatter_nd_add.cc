// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "scatter_nd_add.h"

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
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor ScatterNdAdd(const ir::Tensor& x, const ir::Tensor& index, const ir::Tensor& updates) {
  CHECK_GT(index->shape.size(), 0);
  CHECK_LE(index->shape.back().as_int32(), x->shape.size());  // does shape type relavant to build platform?

  auto updates_expect_shape =
      std::vector<ir::Expr>(index->shape.begin(), index->shape.begin() + index->shape.size() - 1);
  updates_expect_shape.insert(
      updates_expect_shape.end(), x->shape.begin() + index->shape.back().as_int32(), x->shape.end());
  CHECK_EQ(updates_expect_shape.size(), updates->shape.size());
  for (size_t i = 0; i < updates->shape.size(); i++) {
    CHECK_EQ(updates_expect_shape[i], updates->shape[i]);
  }

  auto output_shape = x->shape;

  return lang::Compute(
      output_shape,
      [&](const std::vector<ir::Expr>& indices) { return common::make_const(3.14f); },
      common::UniqName("scatter_nd_add_out"));
}

std::shared_ptr<framework::OpStrategy> StrategyForScatterNdAdd(const framework::NodeAttr& attrs,
                                                               const std::vector<ir::Tensor>& inputs,
                                                               const std::vector<Type>& out_type,
                                                               const std::vector<std::vector<int>>& output_shapes,
                                                               const Target& target) {
  std::string op_name("scatter_nd_add");

  framework::CINNCompute scatter_nd_add_compute([=](lang::Args args, lang::RetValue* ret) {
    CINNValuePack pack_args = args[0];
    Expr x                  = pack_args[0];
    Expr index              = pack_args[1];
    Expr updates            = pack_args[2];

    auto tensor_x       = x.as_tensor_ref();
    auto tensor_index   = index.as_tensor_ref();
    auto tensor_updates = updates.as_tensor_ref();

    auto stages    = CreateStages({tensor_x, tensor_index, tensor_updates});
    ir::Tensor out = ScatterNdAdd(tensor_x, tensor_index, tensor_updates);
    stages->InsertLazily(out);

    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));

    CHECK(!out_type.empty()) << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      scatter_nd_add_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.scatter_nd_add", 1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForScatterNdAdd(const std::vector<std::vector<int>>& inputs_shape,
                                                          const framework::AttrMapType& attrs) {
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForScatterNdAdd(const std::vector<Type>& inputs_type, const framework::AttrMapType& attrs) {
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(scatter_nd_add_ops) {
  CINN_REGISTER_OP(scatter_nd_add)
      .describe("ScatterNdAdd Operator.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForScatterNdAdd)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForScatterNdAdd))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForScatterNdAdd))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible);
  return true;
}