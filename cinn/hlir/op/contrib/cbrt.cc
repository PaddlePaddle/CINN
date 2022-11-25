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

#include "cinn/hlir/op/contrib/cbrt.h"

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

ir::Tensor Cbrt(const ir::Tensor &input, const Target &target, const std::string &output_name) {
  std::string extern_func = "cinn_";
  if (target == common::DefaultHostTarget()) {
    extern_func += "host_";
  } else if (target == common::DefaultNVGPUTarget()) {
    extern_func += "nvgpu_";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  extern_func += "cbrt";

  if (input->type().is_float(32)) {
    extern_func += "_fp32";
  } else if (input->type().is_float(64)) {
    extern_func += "_fp64";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  auto res = Compute(
      input->shape,
      [=](const std::vector<Expr> &indices) {
        Expr x = input(indices);
        return lang::CallExtern(extern_func, {x});
      },
      output_name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForCbrt(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  std::string op_name("cbrt");

  framework::CINNCompute cbrt_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U) << "1 input tensor for " << op_name << " compute";

    std::string tensor_name = UniqName(op_name + "_Out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      tensor_name = pack_args[1].operator std::string();
    }

    Expr input_expr = pack_args[0];
    CHECK(input_expr.as_tensor());
    ir::Tensor input = input_expr.as_tensor_ref();

    auto out = Cbrt(input, target, tensor_name);

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cbrt_compute, framework::GetInjectiveScheduleFunc(output_shapes, target), "strategy.cbrt.x86", 1);
  return strategy;
}

std::vector<shape_t> InferShapeForCbrt(const std::vector<shape_t> &inputs_shape, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForCbrt(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 1U) << "The input's type size should be 1! Please check again.";
  CHECK(inputs_type[0].is_float()) << "The input's type should be float! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(cbrt_ops) {
  CINN_REGISTER_OP(cbrt)
      .describe("Compute cube root.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForCbrt)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForCbrt))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForCbrt))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}