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

#include "cinn/hlir/op/contrib/example.h"

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
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;
using framework::shape_t;

ir::Tensor Example(
    const ir::Tensor &A, const ir::Tensor &B, bool div_x, const Target &target, const std::string &name) {
  CHECK_EQ(A->shape.size(), B->shape.size());
  std::string extern_func = "cinn_";
  if (target == common::DefaultHostTarget()) {
    extern_func += "host_";
  } else if (target == common::DefaultNVGPUTarget()) {
    extern_func += "nvgpu_";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  extern_func += "example";

  if (A->type().is_float(32)) {
    extern_func += "_fp32";
  } else if (A->type().is_int(32)) {
    extern_func += "_int32";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  // just for example, it doesn't make any sense !!!
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        auto x = ir::Select::Make(Expr(div_x), A(indices), B(indices));

        auto y = A(indices) + B(indices);
        return lang::CallExtern(extern_func, {x, y});
      },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForExample(const framework::NodeAttr &attrs,
                                                          const std::vector<ir::Tensor> &inputs,
                                                          const std::vector<Type> &out_type,
                                                          const std::vector<std::vector<int>> &output_shapes,
                                                          const Target &target) {
  std::string op_name("example");

  // check Attribute
  CHECK(attrs.attr_store.count("div_x")) << "No Found Attribute [div_x] in Example Op! Please check.";
  auto div_x = absl::get<bool>(attrs.attr_store.at("div_x"));

  framework::CINNCompute example_compute([=](lang::Args args, lang::RetValue *ret) {
    // check input number
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U) << "2 input tensor for " << op_name << " compute";

    // get tensor name
    std::string tensor_name = UniqName(op_name + "_Out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 3U);
      tensor_name = pack_args[2].operator std::string();
    }

    // get input tensor
    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();

    Expr B_expr = pack_args[1];
    CHECK(B_expr.as_tensor());
    ir::Tensor B = B_expr.as_tensor_ref();

    // get op compute expression
    auto out = Example(A, B, div_x, target, tensor_name);

    // set output stage info
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      example_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.example.x86", 1);
  return strategy;
}

std::vector<shape_t> InferShapeForExample(const std::vector<shape_t> &inputs_shape,
                                          const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForExample(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 2U) << "The input's shape size should be 2! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(example_ops) {
  CINN_REGISTER_OP(example)
      .describe("Example.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForExample)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForExample))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForExample))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
