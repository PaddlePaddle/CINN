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
#include "cinn/hlir/pe/transform.h"
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
using ReduceFunc =
    std::function<ir::Tensor(const ir::Tensor &, const std::vector<int> &, const bool, Expr, const std::string &)>;
using BlockReduceInternalFunc = std::function<std::vector<ir::Tensor>(
    const ir::Tensor &, const std::vector<int> &, const bool, const std::string &)>;
using BlockReduceFunc         = std::function<std::vector<ir::Tensor>(
    const ir::Tensor &, const std::vector<int> &, const int, const bool, const std::string &)>;
using BlockShuffleFunc        = std::function<std::vector<ir::Tensor>(
    const ir::Tensor &, const std::vector<int> &, const bool, const std::string &)>;

#define StrategyForReduction(                                                                                 \
    op_name_, reduce_op_, reduce_func_, block_reduce_internal_func_, block_reduce_func_, block_shuffle_func_) \
  std::shared_ptr<OpStrategy> StrategyFor##reduce_op_(const framework::NodeAttr &attrs,                       \
                                                      const std::vector<ir::Tensor> &inputs,                  \
                                                      const std::vector<Type> &out_type,                      \
                                                      const std::vector<std::vector<int>> &output_shapes,     \
                                                      const Target &target) {                                 \
    return StrategyForReduce(attrs,                                                                           \
                             inputs,                                                                          \
                             out_type,                                                                        \
                             output_shapes,                                                                   \
                             target,                                                                          \
                             #op_name_,                                                                       \
                             reduce_func_,                                                                    \
                             block_reduce_internal_func_,                                                     \
                             block_reduce_func_,                                                              \
                             block_shuffle_func_);                                                            \
  }

std::shared_ptr<OpStrategy> StrategyForReduce(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const std::vector<std::vector<int>> &output_shapes,
                                              const Target &target,
                                              const std::string &op_name,
                                              const ReduceFunc &reduce_func,
                                              const BlockReduceInternalFunc &block_reduce_internal_func,
                                              const BlockReduceFunc &block_reduce_func,
                                              const BlockShuffleFunc &block_shuffle_func) {
  std::vector<int> dim;
  bool keep_dim = false;
  if (attrs.attr_store.count("dim")) {
    dim = absl::get<std::vector<int>>(attrs.attr_store.at("dim"));
    if (dim.empty()) {
      for (int i = 0; i < inputs[0]->shape.size(); ++i) {
        dim.push_back(i);
      }
    }
    std::sort(dim.begin(), dim.end());
    // check dim
    CHECK_LE(dim.size(), inputs[0]->shape.size());
    CHECK_LT(dim.back(), inputs[0]->shape.size());
    for (int idx = 1; idx < dim.size(); ++idx) {
      CHECK_NE(dim[idx - 1], dim[idx]);
    }
  } else {
    LOG(FATAL) << "reduce dimension is not set!";
  }

  if (attrs.attr_store.count("keep_dim")) {
    keep_dim = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }

  auto WithoutLastDimInReduce = [](const std::vector<ir::Expr> &inshape, const std::vector<int> &axes) {
    // if last axis is in reduce.
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx].as_int32();
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  };

  // compute reduce args
  int succesive_dim_idx     = dim.back();
  bool reduce_dim_succesive = true;
  int last_succesive_dim    = inputs[0]->shape[dim.back()].as_int32();
  for (int idx = dim.size() - 2; idx >= 0; --idx) {
    if (dim[idx] != dim[idx + 1] - 1) {
      succesive_dim_idx    = idx + 1;
      reduce_dim_succesive = false;
      break;
    } else {
      if (last_succesive_dim * inputs[0]->shape[dim[idx]].as_int32() > 1024) {
        succesive_dim_idx    = idx + 1;
        reduce_dim_succesive = false;
        break;
      }
      last_succesive_dim *= inputs[0]->shape[dim[idx]].as_int32();
    }
  }

  framework::CINNCompute reduction_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 1U) << "1 input tensor for " << op_name << " compute";
    Expr x_expr = a[0];
    CHECK(x_expr.as_tensor());
    ir::Tensor x = x_expr.as_tensor_ref();
    if (target == common::DefaultNVGPUTarget() && !WithoutLastDimInReduce(inputs[0]->shape, dim)) {
      // the reduce dimension is succesive
      if (reduce_dim_succesive) {
        if (last_succesive_dim <= 1024) {
          VLOG(3) << "Do BlockReduceInternal Compute!";
          // if the succesive reduce dimension size <= 1024
          auto res = block_reduce_internal_func(x, dim, keep_dim, UniqName(op_name + "_out"));
          CHECK_EQ(res.size(), 2);
          auto stages = CreateStages(res);
          *ret        = CINNValuePack{{CINNValue(res[0]), CINNValue(res[1]), CINNValue(stages)}};
        } else {
          VLOG(3) << "Do BlockReduce Compute!";
          /*
          auto res    = pe::TwoStepBlockReduceInternal(x, dim, keep_dim, UniqName(op_name + "_out"));
          auto stages = CreateStages(res);
          std::vector<CINNValue> cinn_values;
          for (auto &t : res) {
            cinn_values.emplace_back(t);
          }
          cinn_values.emplace_back(stages);
          *ret = CINNValuePack{cinn_values};
          */
        }
      } else /* the reduce dimension is not succesive */ {
        VLOG(3) << "Do Reduce And BlockReduceInternal Compute!";
        // compute the parallel reduce dimension size
        int last_succesive_dim_tmp = last_succesive_dim;
        std::vector<int> first_reduce_axes(dim.begin(), dim.begin() + succesive_dim_idx);
        std::vector<int> second_reduce_axes(dim.begin() + succesive_dim_idx, dim.end());
        if (!keep_dim) {
          for (auto &axis : second_reduce_axes) {
            axis -= first_reduce_axes.size();
          }
        }
        // TODO(sunli) : support last dimension size over 1024
        CHECK_LE(last_succesive_dim_tmp, 1024) << "last dimension size is over 1024";
        // first: do reduce without last dimension
        auto out = reduce_func(x, first_reduce_axes, keep_dim, Expr(), UniqName(op_name + "_out"));
        // second: do reduce on last dimension
        auto res = block_reduce_internal_func(out, second_reduce_axes, keep_dim, UniqName(op_name + "_out"));
        CHECK_EQ(res.size(), 2);
        auto stages = CreateStages({res[0], res[1], out});
        *ret        = CINNValuePack{{CINNValue(res[0]), CINNValue(res[1]), CINNValue(out), CINNValue(stages)}};
      }
    } else {
      VLOG(3) << "Do ReduceSum Compute!";
      auto out    = block_shuffle_func(x, dim, keep_dim, UniqName(op_name + "_out"));
      auto stages = CreateStages(out);
      if (out.size() == 1) {
        *ret = CINNValuePack{{CINNValue(out[0]), CINNValue(stages)}};
      } else {
        *ret = CINNValuePack{{CINNValue(out[0]), CINNValue(out[1]), CINNValue(out[2]), CINNValue(stages)}};
      }
    }
  });

  framework::CINNSchedule reduction_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK(arg_pack.size() == 2UL || arg_pack.size() == 3UL || arg_pack.size() == 4UL);
    if (target.arch == Target::Arch::NVGPU) {
      if (!WithoutLastDimInReduce(inputs[0]->shape, dim)) {
        if (reduce_dim_succesive) {
          CHECK_EQ(arg_pack.size(), 3UL);
          Expr out              = arg_pack[0];
          Expr tmp_out          = arg_pack[1];
          poly::StageMap stages = arg_pack.back();

          VLOG(3) << "Do CudaScheduleBlockReduceInternal Schedule!";
          pe::CudaScheduleBlockReduceInternal(
              stages, tmp_out.as_tensor_ref(), out.as_tensor_ref(), common::DefaultNVGPUTarget());
        } else {
          CHECK_EQ(arg_pack.size(), 4UL);
          Expr out              = arg_pack[0];
          Expr tmp_out          = arg_pack[1];
          Expr reduce_tmp_out   = arg_pack[2];
          poly::StageMap stages = arg_pack.back();

          VLOG(3) << "Do CudaScheduleBlockReduce Schedule!";
          pe::CudaScheduleBlockReduce(stages,
                                      reduce_tmp_out.as_tensor_ref(),
                                      tmp_out.as_tensor_ref(),
                                      out.as_tensor_ref(),
                                      common::DefaultNVGPUTarget());
        }
      } else {
        if (arg_pack.size(), 2) {
          Expr reduce_out       = arg_pack[0];
          poly::StageMap stages = arg_pack.back();
          VLOG(3) << "Do CudaScheduleReduce Schedule!";
          pe::CudaScheduleReduce(stages, reduce_out.as_tensor_ref(), inputs[0]->shape.size() - dim.back() - 1, target);
        } else {
          CHECK_EQ(arg_pack.size(), 4) << "args is not equal 4!";
          Expr reduce_reshape   = arg_pack[2];
          Expr reduce_internal  = arg_pack[2];
          Expr reduce_out       = arg_pack[0];
          poly::StageMap stages = arg_pack.back();
          VLOG(3) << "Do CudaScheduleShuffleReduce Schedule!";
          pe::CudaScheduleShuffleReduce(stages,
                                        reduce_reshape.as_tensor_ref(),
                                        reduce_internal.as_tensor_ref(),
                                        reduce_out.as_tensor_ref(),
                                        target);
        }
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
  CHECK(inputs_shape.size() == 1UL || inputs_shape.size() == 3UL);
  std::vector<int> dim;
  bool keep_dim = false;
  if (attrs.find("dim") != attrs.end()) {
    dim = absl::get<std::vector<int>>(attrs.at("dim"));
  }

  if (attrs.find("keep_dim") != attrs.end()) {
    keep_dim = absl::get<bool>(attrs.at("keep_dim"));
  }
  std::vector<int> out_shapes;
  if (!dim.empty()) {
    CHECK_LE(dim.size(), inputs_shape[0].size()) << "reduce dim should no more than the input size";
    auto ndim = inputs_shape[0].size();
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(dim.begin(), dim.end(), i) != dim.end()) {
        if (keep_dim) {
          out_shapes.push_back(1);
        }
      } else {
        out_shapes.push_back(inputs_shape[0][i]);
      }
    }
  }

  if (out_shapes.empty()) {
    out_shapes.push_back(1);
  }

  return {out_shapes};
}

std::vector<Type> InferDtypeForReduction(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
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

  return {{""}, new_input_layouts};
}

std::vector<shape_t> InferShapeForBnOptimize(const std::vector<shape_t> &inputs_shape,
                                             const framework::AttrMapType &attrs) {
  auto shapes = InferShapeForReduction(inputs_shape, attrs);
  CHECK_GE(shapes.size(), 1) << "shapes's size less than 1, please check!";
  return {shapes[0], shapes[0]};
}

std::vector<Type> InferDtypeForBnOptimize(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {inputs_type[0], inputs_type[0]};
}

std::vector<std::vector<std::string>> InferLayoutForBnOptimize(const std::vector<framework::shape_t> &input_shapes,
                                                               const std::vector<std::string> &input_layouts,
                                                               const framework::NodeAttr &attrs,
                                                               const Target &target) {
  return {{"", ""}, {"", ""}};
}

StrategyForReduction(
    reduce_sum, ReduceSum, pe::ReduceSum, pe::BlockReduceSumInternal, pe::BlockReduceSum, pe::BlockShuffleReduceSum);
StrategyForReduction(reduce_prod,
                     ReduceProd,
                     pe::ReduceProd,
                     pe::BlockReduceProdInternal,
                     pe::BlockReduceProd,
                     pe::BlockShuffleReduceProd);
StrategyForReduction(
    reduce_max, ReduceMax, pe::ReduceMax, pe::BlockReduceMaxInternal, pe::BlockReduceMax, pe::BlockShuffleReduceMax);
StrategyForReduction(
    reduce_min, ReduceMin, pe::ReduceMin, pe::BlockReduceMinInternal, pe::BlockReduceMin, pe::BlockShuffleReduceMin);

#undef StrategyForReduction

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reduce_ops) {
#define CINN_REGISTER_REDUCTION(op__, op_stragegy__)                                                                  \
  CINN_REGISTER_OP(op__)                                                                                              \
      .describe(#op__ " function")                                                                                    \
      .set_num_inputs(1)                                                                                              \
      .set_num_outputs(1)                                                                                             \
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

  return true;
}
