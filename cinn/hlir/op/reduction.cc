// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/pe/reduction.h"

#include <iostream>
#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using pe::ReduceMax;
using pe::ReduceMin;
using pe::ReduceProd;
using pe::ReduceSum;
using PeFunc = std::function<ir::Tensor(const ir::Tensor &, const std::vector<int> &, bool, Expr, const std::string &)>;

std::shared_ptr<OpStrategy> StrategyForBNReduceMerge(const framework::NodeAttr &attrs,
                                                     const std::vector<ir::Tensor> &inputs,
                                                     const std::vector<Type> &out_type,
                                                     const std::vector<std::vector<int>> &output_shapes,
                                                     const Target &target) {
  framework::CINNCompute bn_reduce_merge_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of bn_reduce_merge compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for bn_reduce_merge compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    auto x        = A.as_tensor_ref();
    auto x_square = pe::Multiply(x, x, UniqName("bn_reduce_merge_x_square"));

    auto out0 = pe::ReduceSum(x, {0, 2}, false, Expr(0.0f), UniqName("bn_reduce_merge_out0"));
    auto out1 = pe::ReduceSum(x_square, {0, 2}, false, Expr(0.0f), UniqName("bn_reduce_merge_out1"));

    auto stages = CreateStages({x_square, out0, out1});
    stages[x_square]->ComputeInline();

    *ret = CINNValuePack{{CINNValue(out0), CINNValue(out1), CINNValue(stages)}};
  });

  framework::CINNSchedule bn_reduce_merge_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of bn_reduce_merge schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    if (target.arch == Target::Arch::NVGPU) {
      Expr out0             = arg_pack[0];
      Expr out1             = arg_pack[1];
      poly::StageMap stages = arg_pack.back();
      CHECK(out0.as_tensor());
      CHECK(out1.as_tensor());
      pe::CudaScheduleReduce(stages, out0.as_tensor_ref(), target);
      pe::CudaScheduleReduce(stages, out1.as_tensor_ref(), target);
      stages[out0.as_tensor_ref()]->SimpleComputeAt(stages[out1.as_tensor_ref()], 3);
    } else if (target.arch == Target::Arch::X86) {
      Expr out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(out.as_tensor());
      pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes.front(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of bn_reduce_merge op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(bn_reduce_merge_compute, bn_reduce_merge_schedule, "strategy.relu.x86", 1);
  } else {
    LOG(FATAL) << "bn_reduce_merge op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForBNGradReduceMerge(const framework::NodeAttr &attrs,
                                                         const std::vector<ir::Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  framework::CINNCompute bn_grad_reduce_merge_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of bn_grad_reduce_merge compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "at least one input tensor for bn_grad_reduce_merge compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    Expr Mean = a[1];
    CHECK(Mean.as_tensor());
    Expr Grad = a[2];
    CHECK(Grad.as_tensor());

    auto x      = A.as_tensor_ref();
    auto x_mean = Mean.as_tensor_ref();
    auto y_grad = Grad.as_tensor_ref();

    auto x_mean_diff      = pe::Substract(x, x_mean, UniqName("bn_grad_reduce_merge_mean_diff"), Expr(1));
    auto grad_x_mean_diff = pe::Multiply(x_mean_diff, y_grad, UniqName("bn_grad_reduce_merge_grad_mean_diff"));

    auto out0 = pe::ReduceSum(y_grad, {0, 2}, false, Expr(0.0f), UniqName("bn_grad_reduce_merge_out0"));
    auto out1 = pe::ReduceSum(grad_x_mean_diff, {0, 2}, false, Expr(0.0f), UniqName("bn_grad_reduce_merge_out1"));

    auto stages = CreateStages({x_mean_diff, grad_x_mean_diff, out0, out1});
    stages[x_mean_diff]->ComputeInline();
    stages[grad_x_mean_diff]->ComputeInline();
    *ret = CINNValuePack{{CINNValue(out0), CINNValue(out1), CINNValue(stages)}};
  });

  framework::CINNSchedule bn_grad_reduce_merge_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of bn_grad_reduce_merge schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    if (target.arch == Target::Arch::NVGPU) {
      Expr out0             = arg_pack[0];
      Expr out1             = arg_pack[1];
      poly::StageMap stages = arg_pack.back();
      CHECK(out0.as_tensor());
      CHECK(out1.as_tensor());
      pe::CudaScheduleReduce(stages, out0.as_tensor_ref(), target);
      pe::CudaScheduleReduce(stages, out1.as_tensor_ref(), target);
      stages[out0.as_tensor_ref()]->SimpleComputeAt(stages[out1.as_tensor_ref()], 3);
    } else if (target.arch == Target::Arch::X86) {
      Expr out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(out.as_tensor());
      pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes.front(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of bn_grad_reduce_merge op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(bn_grad_reduce_merge_compute, bn_grad_reduce_merge_schedule, "strategy.relu.x86", 1);
  } else {
    LOG(FATAL) << "bn_grad_reduce_merge op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

std::vector<shape_t> InferShapeForBNReduce(const std::vector<shape_t> &inputs_shape,
                                           const framework::AttrMapType &attrs) {
  CHECK(inputs_shape.size() == 3UL || inputs_shape.size() == 1UL);
  std::vector<int> out_shapes{inputs_shape[0][1], inputs_shape[0][3]};
  return {out_shapes, out_shapes};
}

#define StrategyForReduction(op_name__, pe__, pe_func__)                                            \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                   \
                                                const std::vector<ir::Tensor> &inputs,              \
                                                const std::vector<Type> &out_type,                  \
                                                const std::vector<std::vector<int>> &output_shapes, \
                                                const Target &target) {                             \
    return StrategyForReduce(attrs, inputs, out_type, output_shapes, target, #op_name__, pe__);     \
  }

std::shared_ptr<OpStrategy> StrategyForReduce(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const std::vector<std::vector<int>> &output_shapes,
                                              const Target &target,
                                              const std::string &op_name,
                                              const PeFunc &pe_func) {
  std::vector<int> dim;
  bool keep_dim = false;
  if (attrs.attr_store.count("dim")) {
    dim = absl::get<std::vector<int>>(attrs.attr_store.at("dim"));
    std::sort(dim.begin(), dim.end());
  }
  if (attrs.attr_store.count("keep_dim")) {
    keep_dim = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }
  framework::CINNCompute reduction_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 1U) << "1 input tensor for " << op_name << " compute";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();

    // if do reduce on last axis and reduce axis size > 1.
    // two step to do reduce: 1.[n,c,h,w] -> [c,w]; 2.[c,w] -> [c]
    if (dim.back() == inputs[0]->shape.size() - 1 && dim.size() > 1 && target == common::DefaultNVGPUTarget()) {
      // reduce without last dimension
      std::vector<int> dims_without_last(dim.begin(), --dim.end());
      auto out0 = pe_func(A, dims_without_last, keep_dim, Expr(), UniqName(op_name + "_out0"));
      // reduce on last dimension
      std::vector<int> dims_last(1, out0->shape.size() - 1);
      auto out1   = pe_func(out0, dims_last, keep_dim, Expr(), UniqName(op_name + "_out1"));
      auto stages = CreateStages({A, out0, out1});
      *ret        = CINNValuePack{{CINNValue(out1), CINNValue(out0), CINNValue(stages)}};
    } else {
      auto out    = pe_func(A, dim, keep_dim, Expr(), UniqName(op_name + "_out"));
      auto stages = CreateStages({A, out});
      *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
    }
  });

  framework::CINNSchedule reduction_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK(arg_pack.size() == 2UL || arg_pack.size() == 3UL);
    if (target.arch == Target::Arch::NVGPU) {
      poly::StageMap stages = arg_pack.back();
      Expr out0             = arg_pack.size() == 2 ? arg_pack[0] : arg_pack[1];
      pe::CudaScheduleReduce(stages, out0.as_tensor_ref(), target);

      if (arg_pack.size() == 3) {
        Expr out1 = arg_pack[0];
        pe::CudaScheduleReduce(stages, out1.as_tensor_ref(), target);
      }
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reduction_compute, reduction_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForReduction(const std::vector<shape_t> &inputs_shape,
                                            const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<int> dim;
  bool keep_dim = false;
  if (attrs.find("dim") != attrs.end()) {
    dim = absl::get<std::vector<int>>(attrs.at("dim"));
  }

  if (attrs.find("keep_dim") != attrs.end()) {
    keep_dim = absl::get<bool>(attrs.at("keep_dim"));
  }
  CHECK(!dim.empty()) << "should have reduce dim, please check!";
  CHECK_LE(dim.size(), inputs_shape[0].size()) << "reduce dim should no more than the input size";
  std::vector<int> out_shapes, out_shapes_internal;
  auto ndim = inputs_shape[0].size();
  if (keep_dim) {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(dim.begin(), dim.end(), i) != dim.end()) {
        out_shapes.push_back(1);
      } else {
        out_shapes.push_back(inputs_shape[0][i]);
      }
    }

    if (std::find(dim.begin(), dim.end(), inputs_shape[0].size() - 1) != dim.end()) {
      out_shapes_internal        = out_shapes;
      out_shapes_internal.back() = inputs_shape[0].back();
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(dim.begin(), dim.end(), i) == dim.end()) {
        out_shapes.push_back(inputs_shape[0][i]);
      }
    }

    if (std::find(dim.begin(), dim.end(), inputs_shape[0].size() - 1) != dim.end()) {
      out_shapes_internal = out_shapes;
      out_shapes_internal.push_back(inputs_shape[0].back());
    }
  }
  if (out_shapes.empty()) {
    out_shapes.push_back(1);
  }

  if (out_shapes_internal.empty()) {
    out_shapes_internal.push_back(1);
  }
  return {out_shapes, out_shapes_internal};
}

std::vector<Type> InferDtypeForReduction(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForReduction(const std::vector<framework::shape_t> &input_shapes,
                                                              const std::vector<std::string> &input_layouts,
                                                              const framework::NodeAttr &attrs,
                                                              const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> new_input_layouts = input_layouts;
  if (input_shapes[0].size() > 4) {
    // alter input layout back
    new_input_layouts[0] = "NCHW";
    VLOG(3) << "alter input layout from " << input_layouts[0] << " to " << new_input_layouts[0];
  }
  new_input_layouts.push_back("");

  return {{"", ""}, new_input_layouts};
}

StrategyForReduction(reduce_sum, ReduceSum, PeFunc);
StrategyForReduction(reduce_prod, ReduceProd, PeFunc);
StrategyForReduction(reduce_max, ReduceMax, PeFunc);
StrategyForReduction(reduce_min, ReduceMin, PeFunc);

#undef StrategyForReduction

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reduce_ops) {
#define CINN_REGISTER_REDUCTION(op__, op_stragegy__)                                                                  \
  CINN_REGISTER_OP(op__)                                                                                              \
      .describe(#op__ " function")                                                                                    \
      .set_num_inputs(1)                                                                                              \
      .set_num_outputs(2)                                                                                             \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)  \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForReduction))                                 \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForReduction))                                 \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForReduction))                               \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kCommReduce) \
      .set_support_level(4);

  CINN_REGISTER_REDUCTION(reduce_sum, ReduceSum);
  CINN_REGISTER_REDUCTION(reduce_prod, ReduceProd);
  CINN_REGISTER_REDUCTION(reduce_max, ReduceMax);
  CINN_REGISTER_REDUCTION(reduce_min, ReduceMin);

#undef CINN_REGISTER_REDUCTION

  CINN_REGISTER_OP(bn_reduce_merge)
      .describe("This operator implements the optimization of bn reduce")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBNReduceMerge)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBNReduce))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForReduction))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForReduction))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  CINN_REGISTER_OP(bn_grad_reduce_merge)
      .describe("This operator implements the optimization of bn grad reduce")
      .set_num_inputs(3)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBNGradReduceMerge)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBNReduce))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForReduction))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForReduction))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque)
      .set_support_level(4);

  return true;
}
