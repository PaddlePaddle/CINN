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

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "cinn/common/cas.h"
#include "cinn/common/cinn_value.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/macros.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/packed_func.h"
#include "cinn/poly/stage.h"
#include "glog/logging.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

std::shared_ptr<framework::OpStrategy> StrategyForTriangularSolve(const framework::NodeAttr &attrs,
                                                                  const std::vector<ir::Tensor> &inputs,
                                                                  const std::vector<Type> &out_type,
                                                                  const std::vector<std::vector<int>> &output_shapes,
                                                                  const Target &target) {
  framework::CINNCompute triangular_solve_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of triangular_solve is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U) << "Two input tensors are required for the computation of triangular_solve.";
    Expr a_expr             = pack_args[0];
    Expr b_expr             = pack_args[1];
    ir::Tensor a            = a_expr.as_tensor_ref();
    ir::Tensor b            = b_expr.as_tensor_ref();
    std::string tensor_name = "triangular_solve_out";
    auto out                = pe::Identity(b, tensor_name).front();
    auto stages             = CreateStages({out});
    std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
    *ret = CINNValuePack{res};
  });
  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      triangular_solve_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.triangular_solve.x86", 1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForTriangularSolve(const std::vector<framework::shape_t> &inputs_shape,
                                                             const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  framework::shape_t a_shape = inputs_shape[0];
  framework::shape_t b_shape = inputs_shape[1];
  int a_shape_size           = a_shape.size();
  int b_shape_size           = b_shape.size();
  CHECK_GE(a_shape_size, 2U) << "The input a shape size should >= 2! Please check again.";
  CHECK_GE(b_shape_size, 2U) << "The input b shape size should >= 2! Please check again.";

  int left_side = -1;
  for (auto &iter : attrs) {
    if (iter.first == "left_side") {
      left_side = absl::get<bool>(iter.second);
      break;
    }
  }

  CHECK_EQ(a_shape[a_shape_size - 2], a_shape[a_shape_size - 1])
      << "The last two dimensions of the input a must be the same!";
  if (left_side) {
    CHECK_EQ(a_shape[a_shape_size - 2], b_shape[b_shape_size - 2])
        << "The last-but-one dimension of the two vectors must be consistent.";
  } else {
    CHECK_EQ(a_shape[a_shape_size - 1], b_shape[b_shape_size - 1])
        << "The last dimension of the two vectors must be consistent.";
  }

  return {b_shape};
}

std::vector<Type> InferDtypeForTriangularSolve(const std::vector<Type> &inputs_type,
                                               const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 2U) << "The input's shape size should be 2! Please check again.";
  CHECK(inputs_type[0].is_float(32) || inputs_type[0].is_float(64))
      << "The input's dtype should be float32 or float64! Please check again.";
  CHECK(inputs_type[1].is_float(32) || inputs_type[1].is_float(64))
      << "The input's dtype should be float32 or float64! Please check again.";
  return std::vector<Type>{inputs_type[1]};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(triangular_solve_ops) {
  CINN_REGISTER_OP(triangular_solve)
      .describe("TriangularSolve")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForTriangularSolve)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForTriangularSolve))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForTriangularSolve))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
