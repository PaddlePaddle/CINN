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

#include "cinn/hlir/op/contrib/arange.h"

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
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
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

using common::CINNValuePack;

std::vector<ir::Tensor> Arange(
    const float start, const float stop, const float step, const Type &dtype, const std::string &output_name) {
  int num_elem   = static_cast<int>(std::ceil((stop - start) / step));
  ir::Tensor res = lang::Compute(
      {Expr(num_elem)},
      [=](const std::vector<ir::Expr> &indices) {
        return ir::Cast::Make(dtype, start + step * cinn::common::cast(indices[0], common::Float(32)));
      },
      common::UniqName(output_name));
  return {res};
}

std::vector<std::vector<int>> InferShapeForArange(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::AttrMapType &attrs) {
  float start = 0.0F;
  float stop  = 0.0F;
  float step  = 1.0F;
  CHECK(attrs.find("stop") != attrs.end()) << "Please set the stop parameter of arange.";

  if (attrs.find("start") != attrs.end()) {
    start = absl::get<float>(attrs.at("start"));
  }
  if (attrs.find("stop") != attrs.end()) {
    stop = absl::get<float>(attrs.at("stop"));
  }
  if (attrs.find("step") != attrs.end()) {
    step = absl::get<float>(attrs.at("step"));
  }

  CHECK_NE(step, 0) << "The value of step cann't be 0!";

  int num_elem = static_cast<int>(std::ceil((stop - start) / step));
  CHECK_GT(num_elem, 0) << "Invalid arange parameters, start = " << start << ", stop = " << stop << ", step = " << step
                        << ", cause num_elem = " << num_elem << " which is negative.";

  std::vector<std::vector<int>> res = {{num_elem}};
  return res;
}

std::vector<Type> InferDtypeForArange(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  std::string dtype = "float32";
  if (attrs.find("dtype") != attrs.end()) {
    dtype = absl::get<std::string>(attrs.at("dtype"));
  }
  std::vector<Type> res{common::Str2Type(dtype)};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForArange(const framework::NodeAttr &attrs,
                                                         const std::vector<ir::Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  std::string dtype = "float32";
  float start       = 0.0F;
  float stop        = 0.0F;
  float step        = 1.0F;

  for (auto &iter : attrs.attr_store) {
    if (iter.first == "dtype") {
      dtype = absl::get<std::string>(iter.second);
    } else if (iter.first == "start") {
      start = absl::get<float>(iter.second);
    } else if (iter.first == "stop") {
      stop = absl::get<float>(iter.second);
    } else if (iter.first == "step") {
      step = absl::get<float>(iter.second);
    }
  }

  framework::CINNCompute arange_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of arange compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];

    std::string tensor_name = common::UniqName("T_Arange_out");

    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 1U);
      tensor_name = pack_args[0].operator std::string();
    }

    std::vector<ir::Tensor> out = Arange(start, stop, step, common::Str2Type(dtype), tensor_name);
    CHECK(out.size() == 1U) << "The size of Arange's output should be 1";

    std::vector<common::CINNValue> res;
    auto stages = CreateStages({});
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(common::CINNValue(t));
    }

    res.push_back(common::CINNValue(stages));
    *ret = common::CINNValuePack{res};
  });

  framework::CINNSchedule arange_schedule([=](lang::Args args, lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of arange_schedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      std::vector<Expr> vec_ast;
      for (int i = 0; i < arg_pack.size(); i++) {
        if (arg_pack[i].is_expr()) {
          Expr temp = arg_pack[i];
          vec_ast.emplace_back(temp);
        }
      }
      CHECK(!vec_ast.empty());
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      ir_sch.MergeExprs();
      long prod_size = std::accumulate(output_shapes[0].begin(), output_shapes[0].end(), 1, std::multiplies<int>());
      if (prod_size > 1) {
        if (target.arch == Target::Arch::NVGPU) {
          pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
        } else if (target.arch == Target::Arch::X86) {
          pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
        }
      }
      std::vector<common::CINNValue> res{common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = common::CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input argument of arange_schedule is empty! Please check.\n";
      CINNValuePack arg_pack = args[0];
      Expr out               = arg_pack[0];
      CHECK(out.as_tensor());
      *ret = arg_pack;
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(arange_compute, arange_schedule, "strategy.arange.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(arange_ops) {
  CINN_REGISTER_OP(arange)
      .describe("Returns evenly spaced values within a given interval.")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForArange)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForArange))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForArange))
      .set_support_level(4);

  return true;
}
