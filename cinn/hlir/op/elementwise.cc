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

#include "cinn/hlir/pe/elementwise.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
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

#define StrategyForUnary(op_name__, pe__)                                                                \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                        \
                                                const std::vector<ir::Tensor> &inputs,                   \
                                                const std::vector<Type> &out_type,                       \
                                                const std::vector<std::vector<int>> &output_shapes,      \
                                                const Target &target) {                                  \
    return StrategyForElementwise(attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__); \
  }

std::shared_ptr<OpStrategy> StrategyForElementwise(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target,
                                                   const std::string &op_name,
                                                   const PeFunc &pe_func) {
  framework::CINNCompute unary_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK_EQ(a.size(), 1U) << "1 input tensor for " << op_name << " compute";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out     = pe_func(A, UniqName(op_name + "_Out"));
    auto stages  = CreateStages({A});
    std::vector<CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule unary_schedule([=](lang::Args args, lang::RetValue *ret) {
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
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(unary_compute, unary_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForElementwise(const std::vector<shape_t> &inputs_shape,
                                              const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwise(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForElementwise(const std::vector<framework::shape_t> &input_shapes,
                                                                const std::vector<std::string> &input_layouts,
                                                                const framework::NodeAttr &attrs,
                                                                const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

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
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensors of scale compute is empty! Please check.";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor out;
    if (bias_after_scale) {
      out = Compute(
          A->shape, [=](const std::vector<Expr> &indice) { return scale * A(indice) + bias; }, UniqName("Scale_out"));
    } else {
      out = Compute(
          A->shape, [=](const std::vector<Expr> &indice) { return scale * (A(indice) + bias); }, UniqName("Scale_out"));
    }
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule scale_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of scale schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL) << "The input tensor's size of scale schedule is " << arg_pack.size()
                                   << "and it should be equal to 2! Please check.";
    Expr Out              = arg_pack[0];
    poly::StageMap stages = arg_pack[1];
    CHECK(Out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, scale_schedule, "strategy.scale.x86", 1);

  return strategy;
}

Expr GetScalarExpr(const framework::NodeAttr::attr_t &attr) {
  Expr scalar;
  struct Visitor {
    Expr &scalar_;
    explicit Visitor(Expr &scalar) : scalar_(scalar) {}
    void operator()(float v) { scalar_ = Expr(v); }
    void operator()(int v) { scalar_ = Expr(v); }
    void operator()(bool v) { scalar_ = Expr(v); }
    void operator()(const std::string &v) { scalar_ = Expr(v); }
    void operator()(const std::vector<int> &) { LOG(FATAL) << "wrong type std::vector<int>"; }
    void operator()(const std::vector<float> &) { LOG(FATAL) << "wrong type std::vector<float>"; }
    void operator()(const std::vector<bool> &) { LOG(FATAL) << "wrong type std::vector<bool>"; }
    void operator()(const std::vector<std::string> &) { LOG(FATAL) << "wrong type std::vector<std::string>"; }
  };
  absl::visit(Visitor{scalar}, attr);
  return scalar;
}

std::shared_ptr<OpStrategy> StrategyForConstScalar(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target) {
  framework::CINNCompute const_scalar_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of const_float compute is empty! Please check.";
    auto scalar = GetScalarExpr(attrs.attr_store.at("value"));
    auto out    = lang::Compute(
        {Expr(1)}, [=](const std::vector<Expr> &indice) { return scalar; }, UniqName("const_scalar_Out"));
    CHECK(out.defined()) << "can't create const scalar with the given type " << out_type[0];
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  framework::CINNSchedule const_scalar_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of create_const_float schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr Out              = arg_pack[0];
    poly::StageMap stages = arg_pack.back();
    CHECK(Out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(const_scalar_compute, const_scalar_schedule, "strategy.const_scalar.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForConstScalar(const std::vector<shape_t> &inputs_shape,
                                              const framework::AttrMapType &attrs) {
  return {{1}};
}

std::vector<Type> InferDtypeForConstScalar(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(attrs.count("value"));
  auto scalar   = GetScalarExpr(attrs.at("value"));
  auto out_type = scalar->type();
  VLOG(3) << "scalar type: " << out_type;
  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForConstScalar(const std::vector<framework::shape_t> &input_shapes,
                                                                const std::vector<std::string> &input_layouts,
                                                                const framework::NodeAttr &attrs,
                                                                const Target &target) {
  return {{"C"}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForSum(const framework::NodeAttr &attrs,
                                           const std::vector<ir::Tensor> &inputs,
                                           const std::vector<Type> &out_type,
                                           const std::vector<std::vector<int>> &output_shapes,
                                           const Target &target) {
  LOG(FATAL) << "The operator will be decomposed into several primitive operators. Please Use Decomposer Program Pass.";
}

std::vector<shape_t> InferShapeForSum(const std::vector<shape_t> &inputs_shape, const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty()) << "At least 1 input tensor for sum operator.";
  auto shape = inputs_shape[0];
  for (size_t i = 1; i < inputs_shape.size(); ++i) {
    if (inputs_shape[i] != shape) {
      LOG(FATAL) << "The input shapes must be the same. But received: the i-th(" << i << ") input shape is "
                 << utils::Join(inputs_shape[i], ",") << " and the first input shape is " << utils::Join(shape, ",");
    }
  }
  std::vector<shape_t> out_shape{shape};

  return out_shape;
}

std::vector<Type> InferDtypeForSum(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "At least 1 input tensor for sum operator.";
  auto type = inputs_type[0];
  for (size_t i = 1; i < inputs_type.size(); ++i) {
    if (inputs_type[i] != type) {
      LOG(FATAL) << "The input types must be the same. But received: the i-th(" << i << ") input type is "
                 << inputs_type[i] << " and the first input type is " << type;
    }
  }
  std::vector<Type> res{type};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForFillConstant(const framework::NodeAttr &attrs,
                                                    const std::vector<ir::Tensor> &inputs,
                                                    const std::vector<Type> &out_type,
                                                    const std::vector<std::vector<int>> &output_shapes,
                                                    const Target &target) {
  framework::CINNCompute fill_constant_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of fill_constant compute is empty! Please check.";
    bool force_cpu = false;
    CHECK(attrs.attr_store.count("shape"));
    auto shape = absl::get<std::vector<int>>(attrs.attr_store.at("shape"));
    CHECK(attrs.attr_store.count("value"));
    auto value = GetScalarExpr(attrs.attr_store.at("value"));
    CHECK(attrs.attr_store.count("force_cpu"));
    force_cpu = absl::get<bool>(attrs.attr_store.at("force_cpu"));

    if (force_cpu) CINN_NOT_IMPLEMENTED

    CHECK(!shape.empty()) << "shape attr is empty!";
    auto shape_exprs = ToCinnExprs(shape);
    auto out         = lang::Compute(
        shape_exprs, [=](const std::vector<Expr> &indice) { return value; }, UniqName("fill_constant_Out"));
    CHECK(out.defined()) << "can't create fill_constant with the given type " << out_type[0];
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  framework::CINNSchedule fill_constant_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of create_const_float schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr Out              = arg_pack[0];
    poly::StageMap stages = arg_pack.back();
    CHECK(Out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(fill_constant_compute, fill_constant_schedule, "strategy.fill_constant.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForFillConstant(const std::vector<shape_t> &inputs_shape,
                                               const framework::AttrMapType &attrs) {
  CHECK(attrs.count("shape"));
  auto shape = absl::get<std::vector<int>>(attrs.at("shape"));
  CHECK(!shape.empty()) << "shape attr is empty!";
  return {shape};
}

std::vector<Type> InferDtypeForFillConstant(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(attrs.count("value"));
  auto scalar   = GetScalarExpr(attrs.at("value"));
  auto out_type = scalar->type();
  VLOG(3) << "scalar type: " << out_type;
  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForFillConstant(const std::vector<framework::shape_t> &input_shapes,
                                                                 const std::vector<std::string> &input_layouts,
                                                                 const framework::NodeAttr &attrs,
                                                                 const Target &target) {
  return {{""}, input_layouts};
}

StrategyForUnary(exp, Exp);
StrategyForUnary(erf, Erf);
StrategyForUnary(sqrt, Sqrt);
StrategyForUnary(log, Log);
StrategyForUnary(floor, Floor);
StrategyForUnary(ceil, Ceil);
StrategyForUnary(round, Round);
StrategyForUnary(tanh, Tanh);
StrategyForUnary(log2, Log2);
StrategyForUnary(log10, Log10);
StrategyForUnary(trunc, Trunc);
StrategyForUnary(cos, Cos);
StrategyForUnary(cosh, Cosh);
StrategyForUnary(tan, Tan);
StrategyForUnary(sin, Sin);
StrategyForUnary(sinh, Sinh);
StrategyForUnary(acos, Acos);
StrategyForUnary(acosh, Acosh);
StrategyForUnary(asin, Asin);
StrategyForUnary(asinh, Asinh);
StrategyForUnary(atan, Atan);
StrategyForUnary(atanh, Atanh);

StrategyForUnary(isnan, IsNan);
StrategyForUnary(isfinite, IsFinite);
StrategyForUnary(isinf, IsInf);
StrategyForUnary(bitwise_not, BitwiseNot);

StrategyForUnary(negative, Negative);
StrategyForUnary(identity, Identity);
StrategyForUnary(logical_not, LogicalNot);
StrategyForUnary(sign, Sign);
StrategyForUnary(abs, Abs);
StrategyForUnary(rsqrt, Rsqrt);
StrategyForUnary(sigmoid, Sigmoid);

#undef StrategyForUnary

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise_ops) {
#define CINN_REGISTER_UNARY(op__, op_stragegy__)                                                                     \
  CINN_REGISTER_OP(op__)                                                                                             \
      .describe(#op__ " function")                                                                                   \
      .set_num_inputs(1)                                                                                             \
      .set_num_outputs(1)                                                                                            \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__) \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))                              \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))                              \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))                            \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)  \
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

  CINN_REGISTER_UNARY(isnan, IsNan)
  CINN_REGISTER_UNARY(isfinite, IsFinite)
  CINN_REGISTER_UNARY(isinf, IsInf)
  CINN_REGISTER_UNARY(bitwise_not, BitwiseNot)

  CINN_REGISTER_UNARY(negative, Negative)
  CINN_REGISTER_UNARY(identity, Identity)
  CINN_REGISTER_UNARY(logical_not, LogicalNot)
  CINN_REGISTER_UNARY(sign, Sign)
  CINN_REGISTER_UNARY(abs, Abs)
  CINN_REGISTER_UNARY(rsqrt, Rsqrt)
  CINN_REGISTER_UNARY(sigmoid, Sigmoid)

#undef CINN_REGISTER_UNARY

  CINN_REGISTER_OP(scale)
      .describe("Putting scale and bias to the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForScale)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)
      .set_support_level(4);

  CINN_REGISTER_OP(const_scalar)
      .describe("create const scalar with the given value")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForConstScalar)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForConstScalar))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForConstScalar))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForConstScalar))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)
      .set_support_level(4);

  CINN_REGISTER_OP(sum)
      .describe("Sum the input tensors.")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSum)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSum))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSum))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise);

  CINN_REGISTER_OP(fill_constant)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForFillConstant)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForFillConstant))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForFillConstant))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForFillConstant))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)
      .set_support_level(4);

  CINN_REGISTER_OP(get_shape)
      .describe("get the shape of the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForElementwise));

  return true;
}
