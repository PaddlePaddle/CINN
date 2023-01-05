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

#include "absl/types/optional.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
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
using PeFunc = std::function<std::vector<ir::Tensor>(const ir::Tensor &A, const std::string &out_name)>;

#define STRATEGY_FOR_UNARY(op_name__, pe__)                                                         \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                   \
                                                const std::vector<ir::Tensor> &inputs,              \
                                                const std::vector<Type> &out_type,                  \
                                                const std::vector<std::vector<int>> &output_shapes, \
                                                const Target &target) {                             \
    return StrategyForUnary(attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__);  \
  }

std::shared_ptr<OpStrategy> StrategyForUnary(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const std::vector<std::vector<int>> &output_shapes,
                                             const Target &target,
                                             const std::string &op_name,
                                             const PeFunc &pe_func) {
  framework::CINNCompute unary_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 2U);
    CHECK(pack_args[1].is_string());
    std::string tensor_name = pack_args[1].operator std::string();

    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out     = pe_func(A, tensor_name);
    auto stages  = CreateStages({A});
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      unary_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForUnary(const std::vector<shape_t> &inputs_shape, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForUnary(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<Type> InferDtypeForUnaryBool(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {Bool()};
}

std::vector<std::vector<std::string>> InferLayoutForUnary(const std::vector<framework::shape_t> &input_shapes,
                                                          const std::vector<std::string> &input_layouts,
                                                          const framework::NodeAttr &attrs,
                                                          const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

STRATEGY_FOR_UNARY(exp, Exp);
STRATEGY_FOR_UNARY(erf, Erf);
STRATEGY_FOR_UNARY(sqrt, Sqrt);
STRATEGY_FOR_UNARY(log, Log);
STRATEGY_FOR_UNARY(floor, Floor);
STRATEGY_FOR_UNARY(ceil, Ceil);
STRATEGY_FOR_UNARY(round, Round);
STRATEGY_FOR_UNARY(tanh, Tanh);
STRATEGY_FOR_UNARY(log2, Log2);
STRATEGY_FOR_UNARY(log10, Log10);
STRATEGY_FOR_UNARY(trunc, Trunc);
STRATEGY_FOR_UNARY(cos, Cos);
STRATEGY_FOR_UNARY(cosh, Cosh);
STRATEGY_FOR_UNARY(tan, Tan);
STRATEGY_FOR_UNARY(sin, Sin);
STRATEGY_FOR_UNARY(sinh, Sinh);
STRATEGY_FOR_UNARY(acos, Acos);
STRATEGY_FOR_UNARY(acosh, Acosh);
STRATEGY_FOR_UNARY(asin, Asin);
STRATEGY_FOR_UNARY(asinh, Asinh);
STRATEGY_FOR_UNARY(atan, Atan);
STRATEGY_FOR_UNARY(atanh, Atanh);

STRATEGY_FOR_UNARY(isnan, IsNan);
STRATEGY_FOR_UNARY(isfinite, IsFinite);
STRATEGY_FOR_UNARY(isinf, IsInf);
STRATEGY_FOR_UNARY(bitwise_not, BitwiseNot);

STRATEGY_FOR_UNARY(negative, Negative);
STRATEGY_FOR_UNARY(identity, Identity);
STRATEGY_FOR_UNARY(logical_not, LogicalNot);
STRATEGY_FOR_UNARY(sign, Sign);
STRATEGY_FOR_UNARY(abs, Abs);
STRATEGY_FOR_UNARY(rsqrt, Rsqrt);
STRATEGY_FOR_UNARY(sigmoid, Sigmoid);

#undef STRATEGY_FOR_UNARY

std::shared_ptr<OpStrategy> StrategyForScale(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const std::vector<std::vector<int>> &output_shapes,
                                             const Target &target) {
  float scale           = 1.f;
  float bias            = 0.f;
  bool bias_after_scale = true;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "scale") {
      scale = absl::get<float>(iter.second);
    } else if (iter.first == "bias") {
      bias = absl::get<float>(iter.second);
    } else if (iter.first == "bias_after_scale") {
      bias_after_scale = absl::get<bool>(iter.second);
    }
  }
  framework::CINNCompute scale_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of scale compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "The input tensors of scale compute is empty! Please check.";
    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor out;
    std::string tensor_name = UniqName("Scale_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }

    if (bias_after_scale) {
      out = Compute(
          A->shape,
          [=](const std::vector<Expr> &indice) {
            return ir::Cast::Make(A->type(), Expr(scale)) * A(indice) + ir::Cast::Make(A->type(), Expr(bias));
          },
          tensor_name);
    } else {
      out = Compute(
          A->shape,
          [=](const std::vector<Expr> &indice) {
            return ir::Cast::Make(A->type(), Expr(scale)) * (A(indice) + ir::Cast::Make(A->type(), Expr(bias)));
          },
          tensor_name);
    }
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.scale.x86", 1);

  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForSqueeze(const framework::NodeAttr &attrs,
                                                          const std::vector<ir::Tensor> &inputs,
                                                          const std::vector<Type> &out_type,
                                                          const std::vector<std::vector<int>> &output_shapes,
                                                          const Target &target) {
  const std::vector<int> &axes =
      attrs.attr_store.count("axes") ? absl::get<std::vector<int>>(attrs.attr_store.at("axes")) : std::vector<int>{};

  framework::CINNCompute squeeze_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Squeeze compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U) << "at least 1 input tensors for Squeeze compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto stages   = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    std::string tensor_name = UniqName("Squeeze_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      tensor_name = pack_args[1].operator std::string();
    }

    ir::Tensor out = pe::Squeeze(tensor_A, axes, tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Squeeze is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(squeeze_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.squeeze.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForSqueeze(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U);
  const std::vector<int> &axes =
      attrs.count("axes") ? absl::get<std::vector<int>>(attrs.at("axes")) : std::vector<int>{};
  VLOG(4) << "The [axis] value used in Squeeze: " << cinn::utils::Join(axes, ",");

  const auto &posi_axes = GetPositiveAxes(axes, inputs_shape[0].size());
  std::vector<int> output_shape;
  if (posi_axes.size()) {
    for (int idx = 0; idx < inputs_shape[0].size(); ++idx) {
      // if can't find idx in axis
      if (std::find(posi_axes.begin(), posi_axes.end(), idx) == posi_axes.end()) {
        output_shape.push_back(inputs_shape[0][idx]);
      } else {
        CHECK_EQ(inputs_shape[0][idx], 1);
      }
    }
  } else {
    for (int idx = 0; idx < inputs_shape[0].size(); ++idx) {
      if (inputs_shape[0][idx] != 1) {
        output_shape.push_back(inputs_shape[0][idx]);
      }
    }
  }

  VLOG(4) << "The output calculated in Squeeze: " << cinn::utils::Join(output_shape, ", ");

  return {output_shape};
}

std::shared_ptr<OpStrategy> StrategyForExpandDims(const framework::NodeAttr &attrs,
                                                  const std::vector<ir::Tensor> &inputs,
                                                  const std::vector<Type> &out_type,
                                                  const std::vector<std::vector<int>> &output_shapes,
                                                  const Target &target) {
  const std::vector<int> &axes =
      attrs.attr_store.count("axes") ? absl::get<std::vector<int>>(attrs.attr_store.at("axes")) : std::vector<int>{};

  framework::CINNCompute expand_dims_compute{[=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input args are empty! Please check again.";
    CINNValuePack input_args = args[0];
    int input_size           = input_args.size();
    CHECK_GE(input_size, 1U) << "Require 1 input tensors for expand_dims compute.";
    Expr x = input_args[0];
    CHECK(x.as_tensor());

    std::string tensor_name = UniqName("expand_dims_output");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(input_args.size(), 2U);
      CHECK(input_args[1].is_string());
      tensor_name = input_args[1].operator std::string();
    }

    auto out    = pe::ExpandDims(x.as_tensor_ref(), axes, output_shapes[0], tensor_name);
    auto stages = CreateStages({x.as_tensor_ref()});
    stages->InsertLazily(out);
    std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
    *ret = CINNValuePack{res};
  }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      expand_dims_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.expand_dims.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForExpandDims(const std::vector<std::vector<int>> &inputs_shape,
                                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";

  CHECK_EQ(inputs_shape.size(), 1U);
  const std::vector<int> &axes =
      attrs.count("axes") ? absl::get<std::vector<int>>(attrs.at("axes")) : std::vector<int>{};
  VLOG(4) << "The [axes] value used in ExpandDims: " << cinn::utils::Join(axes, ",");

  std::vector<int> out_shape(inputs_shape[0].size() + axes.size(), 1);
  const auto &posi_axes = GetPositiveAxes(axes, out_shape.size());

  int shape_pos = 0, axes_pos = 0;
  for (int i = 0; i < out_shape.size(); ++i) {
    if (axes_pos < posi_axes.size() && posi_axes[axes_pos] == i) {
      out_shape[i] = 1;
      ++axes_pos;
    } else if (shape_pos < inputs_shape[0].size()) {
      out_shape[i] = inputs_shape[0][shape_pos];
      ++shape_pos;
    }
  }

  VLOG(4) << "The output calculated in ExpandDims: " << cinn::utils::Join(out_shape, ", ");
  return {out_shape};
}

std::shared_ptr<OpStrategy> StrategyForReshape(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const std::vector<std::vector<int>> &output_shapes,
                                               const Target &target) {
  framework::CINNCompute reshape_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Reshape compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U) << "at least 1 input tensors for Reshape compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto attr_store = attrs.attr_store;
    CHECK(attr_store.count("shape")) << "find no attr of shape";
    std::vector<int> new_shape = absl::get<std::vector<int>>(attr_store.at("shape"));
    auto tensor_A              = A.as_tensor_ref();
    auto stages                = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    std::string tensor_name = UniqName("Reshape_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }

    ir::Tensor out = pe::Reshape(tensor_A, output_shapes[0], tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Reshape is empty! Please check.\n";
    res.push_back(CINNValue(stages));

    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reshape_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.reshape.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForReshape(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<int> output_shape;
  for (auto &iter : attrs) {
    if (iter.first == "shape") {
      output_shape = absl::get<std::vector<int>>(iter.second);
      break;
    }
  }
  int tensor_size = 1;
  for (auto i : inputs_shape[0]) {
    tensor_size *= i;
  }
  CHECK(!output_shape.empty()) << "infer_shape for reshape turns out to be empty. Please check\n";
  int flag_index = -1;
  for (int i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > 0) {
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: " << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == 0) {
      CHECK_LT(i, inputs_shape[0].size())
          << "In op reshape, when attribute shape[i] == 0, shape[i] = input_shape[i]. But now the size of input_shape "
             "<= i, which is incompatible. Please check!";
      output_shape[i] = inputs_shape[0][i];
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: " << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == -1 && flag_index == -1) {
      flag_index = i;
    } else if (output_shape[i] == -1) {
      LOG(FATAL) << "More than one -1 in output_shape of op reshape.";
    } else {
      LOG(FATAL) << "Unsupported output_shape " << output_shape[i];
    }
  }
  if (flag_index >= 0) output_shape[flag_index] = tensor_size;
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForCast(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  framework::CINNCompute cast_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Cast compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U) << "at least 1 input tensors for Cast compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto stages   = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    std::string tensor_name = UniqName("Cast_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      tensor_name = pack_args[1].operator std::string();
    }
    ir::Tensor out = pe::Cast(tensor_A, out_type[0], tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Cast is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.reshape.x86", 1);
  return strategy;
}

std::vector<Type> InferDtypeForCast(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(attrs.count("dtype"));
  return {common::Str2Type(absl::get<std::string>(attrs.at("dtype")))};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(unary_ops) {
#define CINN_REGISTER_UNARY(op__, op_stragegy__)                                                                       \
  CINN_REGISTER_OP(op__)                                                                                               \
      .describe(#op__ " function")                                                                                     \
      .set_num_inputs(1)                                                                                               \
      .set_num_outputs(1)                                                                                              \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)   \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForUnary))                                      \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnary))                                      \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))                                    \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
      .set_support_level(4);

  CINN_REGISTER_UNARY(exp, Exp);
  CINN_REGISTER_UNARY(erf, Erf);
  CINN_REGISTER_UNARY(sqrt, Sqrt);
  CINN_REGISTER_UNARY(log, Log);
  CINN_REGISTER_UNARY(floor, Floor);
  CINN_REGISTER_UNARY(ceil, Ceil);
  CINN_REGISTER_UNARY(round, Round);
  CINN_REGISTER_UNARY(tanh, Tanh);
  CINN_REGISTER_UNARY(log2, Log2);
  CINN_REGISTER_UNARY(log10, Log10);
  CINN_REGISTER_UNARY(trunc, Trunc);
  CINN_REGISTER_UNARY(cos, Cos);
  CINN_REGISTER_UNARY(cosh, Cosh);
  CINN_REGISTER_UNARY(tan, Tan);
  CINN_REGISTER_UNARY(sin, Sin);
  CINN_REGISTER_UNARY(sinh, Sinh);
  CINN_REGISTER_UNARY(acos, Acos);
  CINN_REGISTER_UNARY(acosh, Acosh);
  CINN_REGISTER_UNARY(asin, Asin);
  CINN_REGISTER_UNARY(asinh, Asinh);
  CINN_REGISTER_UNARY(atan, Atan);
  CINN_REGISTER_UNARY(atanh, Atanh);
  CINN_REGISTER_UNARY(bitwise_not, BitwiseNot)

  CINN_REGISTER_UNARY(negative, Negative)
  CINN_REGISTER_UNARY(identity, Identity)
  CINN_REGISTER_UNARY(logical_not, LogicalNot)
  CINN_REGISTER_UNARY(sign, Sign)
  CINN_REGISTER_UNARY(abs, Abs)
  CINN_REGISTER_UNARY(rsqrt, Rsqrt)
  CINN_REGISTER_UNARY(sigmoid, Sigmoid)

#undef CINN_REGISTER_UNARY

#define CINN_REGISTER_COMPARE(op__, op_stragegy__)                                                                     \
  CINN_REGISTER_OP(op__)                                                                                               \
      .describe(#op__ " function")                                                                                     \
      .set_num_inputs(1)                                                                                               \
      .set_num_outputs(1)                                                                                              \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)   \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForUnary))                                      \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnaryBool))                                  \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))                                    \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
      .set_support_level(4);

  CINN_REGISTER_COMPARE(isnan, IsNan)
  CINN_REGISTER_COMPARE(isfinite, IsFinite)
  CINN_REGISTER_COMPARE(isinf, IsInf)

#undef CINN_REGISTER_COMPARE

  CINN_REGISTER_OP(scale)
      .describe("Putting scale and bias to the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForScale)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForUnary))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnary))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(sum)
      .describe("Sum the input tensors.")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSum)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSum))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSum))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(squeeze)
      .describe("The operator is used to squeeze input tensor's dims")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSqueeze)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSqueeze))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnary))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(expand_dims)
      .describe("This operator is used to expand input tensor's dims.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForExpandDims)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForExpandDims))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnary))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(reshape)
      .describe("This operator is used to reshape input tensor X.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForReshape)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForReshape))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForUnary))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(cast)
      .describe("This operator is used to cast input tensor's type to target.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForCast)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForUnary))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForCast))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForUnary))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
