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

#include "cinn/hlir/op/contrib/flip.h"

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
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/hlir/pe/transform.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

ir::Tensor Flip(const ir::Tensor &in_tensor,
                             const std::vector<int>& axis,
                             const std::string& output_name) {
 for (auto& val : axis) {
    CHECK(val >= 0 && val < static_cast<int>(in_tensor->shape.size())) << "axis should be [0,n_dim)";
  }
    std::vector<Expr> shape = in_tensor->shape;
  return {Compute(
      in_tensor->shape,
      
      [=](const std::vector<Expr> &indice) {
        // ir::Tensor out_tensor(in_tensor);
        // auto e = out_tensor(indice);
        // return ir::Max::Make(ir::Min::Make(e, ir::Cast::Make(e->type(), Expr(max_val))), 
        //                                    ir::Cast::Make(e->type(), Expr(min_val)));
        std::vector<Expr> indexs(indice.begin(), indice.end());
        for (auto idx : axis) {
          indexs[idx] = shape[idx] - Expr(1) - indexs[idx];
        }
        return in_tensor(indexs);
      },
      output_name)};
}

std::vector<std::vector<std::string>> InferLayoutForFlip(const std::vector<framework::shape_t> &input_shapes,
                                                         const std::vector<std::string> &input_layouts,
                                                         const framework::NodeAttr &attrs,
                                                         const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

std::vector<shape_t> InferShapeForFlip(const std::vector<shape_t> &inputs_shape, const framework::AttrMapType &attrs) {
//   CHECK_EQ(inputs_shape.size(), 1UL);
//   std::vector<shape_t> res{inputs_shape[0]};
//   return res;

    CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  if (attrs.find("axis") != attrs.end()) {
    auto axis = absl::get<std::vector<int>>(attrs.at("axis"));
    CHECK(!axis.empty()) << "axis is empty! Please check setting.\n";
    for (auto &e : axis) {
      if (e >= static_cast<int>(inputs_shape[0].size()) || e < -1 * static_cast<int>(inputs_shape[0].size())) {
        LOG(FATAL) << "axis is not in [-n_dim, n_dim), Please check.";
      }
      if (e < 0) {
        e += inputs_shape[0].size();
      }
    }
  } else {
    LOG(FATAL) << "axis is not be set! Please check.";
  }
  return res;
  
}

std::vector<Type> InferDtypeForFlip(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForFlip(const framework::NodeAttr &attrs,
                                            const std::vector<ir::Tensor> &inputs,
                                            const std::vector<Type> &out_type,
                                            const std::vector<std::vector<int>> &output_shapes,
                                            const Target &target) {
    // check output shape
  CHECK(!output_shapes.empty() && !output_shapes[0].empty()) << "Output shape is empty! Please check.\n";
    // get axis[0, n_dim)
  std::vector<int> axis;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    CHECK(!axis.empty()) << "axis is empty! Please check setting.\n";
    for (auto &e : axis) {
      if (e >= static_cast<int>(output_shapes[0].size()) || e < -1 * static_cast<int>(output_shapes[0].size())) {
        LOG(FATAL) << "axis is not in [0, n_dim), Please check.";
      }
      if (e < 0) {
        e += output_shapes[0].size();
      }
    }
  } else {
    LOG(FATAL) << "axis is not be set! Please check.";
  }

  std::string op_name("flip");

  framework::CINNCompute flip_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "at least one input tensor for flip compute\n";
    CINNValuePack input_args = args[0];
    Expr A = input_args[0];
    CHECK(A.as_tensor());
    std::string tensor_name = UniqName(op_name + "_Out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      tensor_name = pack_args[1].operator std::string();
    }
    
    ir::Tensor AA = A.as_tensor_ref();
    auto out     = Flip(AA, axis, tensor_name);
    auto stages  = CreateStages({AA,out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  framework::CINNSchedule flip_schedule([=](lang::Args args, lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
      CINNValuePack arg_pack = args[0];
      Expr ast_expr          = arg_pack[0];
      std::vector<Expr> vec_ast{ast_expr};
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      if (target.arch == Target::Arch::NVGPU) {
        pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target);
      }
      std::vector<CINNValue> res;
      res.push_back(arg_pack[0]);
      *ret = CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
      CINNValuePack arg_pack = args[0];
      CHECK_EQ(arg_pack.size(), 2UL);
      Expr Out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(Out.as_tensor());
      if (target.arch == Target::Arch::NVGPU) {
        pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.front(), target);
      }
      *ret = arg_pack;
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of flip op is empty! Please check.";
  strategy->AddImpl(flip_compute, flip_schedule, "strategy.flip.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(flip_ops) {
  CINN_REGISTER_OP(flip)
      .describe("Flip the input tensors.")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForFlip)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForFlip))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForFlip))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForFlip))
#endif
      .set_support_level(4);

  return true;
}