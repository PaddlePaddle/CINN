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
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {

Expr GetScalarExpr(const framework::NodeAttr::attr_t &attr) {
  Expr scalar;
  struct Visitor {
    Expr &scalar_;
    explicit Visitor(Expr &scalar) : scalar_(scalar) {}
    void operator()(float v) { scalar_ = Expr(v); }
    void operator()(double v) { scalar_ = Expr(v); }
    void operator()(int32_t v) { scalar_ = Expr(v); }
    void operator()(int64_t v) { scalar_ = Expr(v); }
    void operator()(bool v) { scalar_ = Expr(v); }
    void operator()(const std::string &v) { scalar_ = Expr(v); }
    void operator()(const std::vector<int> &) { LOG(FATAL) << "wrong type std::vector<int>"; }
    void operator()(const std::vector<int64_t> &) { LOG(FATAL) << "wrong type std::vector<int64_t>"; }
    void operator()(const std::vector<float> &) { LOG(FATAL) << "wrong type std::vector<float>"; }
    void operator()(const std::vector<double> &) { LOG(FATAL) << "wrong type std::vector<double>"; }
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
    auto scalar             = GetScalarExpr(attrs.attr_store.at("value"));
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 1U);
    CHECK(pack_args[0].is_string());
    std::string tensor_name = pack_args[0].operator std::string();
    auto out                = lang::Compute(
        {Expr(1)}, [=](const std::vector<Expr> &indice) { return scalar; }, tensor_name);
    CHECK(out.defined()) << "can't create const scalar with the given type " << out_type[0];
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      const_scalar_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.const_scalar.x86", 1);

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

std::shared_ptr<OpStrategy> StrategyForFillConstant(const framework::NodeAttr &attrs,
                                                    const std::vector<ir::Tensor> &inputs,
                                                    const std::vector<Type> &out_type,
                                                    const std::vector<std::vector<int>> &output_shapes,
                                                    const Target &target) {
  framework::CINNCompute fill_constant_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of fill_constant compute is empty! Please check.";
    auto shape     = GetAttr<std::vector<int>>(attrs.attr_store, "shape", {});
    auto value     = GetScalarExpr(attrs.attr_store.at("value"));
    bool force_cpu = GetAttr<bool>(attrs.attr_store, "force_cpu", false);

#ifdef CINN_WITH_CUDA
    if (force_cpu && target != common::DefaultHostTarget()) {
      LOG(WARNING) << "[force_cpu] not supported in CINN! The output will placed on device.";
    }
#endif

    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 1U);
    CHECK(arg_pack[0].is_string());
    std::string tensor_name = arg_pack[0].operator std::string();

    CHECK(!shape.empty()) << "shape attr is empty!";
    auto shape_exprs = ToCinnExprs(shape);
    auto out         = lang::Compute(
        shape_exprs, [=](const std::vector<Expr> &indice) { return ir::Cast::Make(out_type[0], value); }, tensor_name);
    CHECK(out.defined()) << "can't create fill_constant with the given type " << out_type[0];
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      fill_constant_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.fill_constant.x86", 1);

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
  common::Type out_type;
  if (attrs.find("dtype") != attrs.end()) {
    // attribute [dtype] are given
    auto dtype_str = absl::get<std::string>(attrs.at("dtype"));
    out_type       = common::Str2Type(dtype_str);
    VLOG(3) << "FillConstant output dtype (from [dtype]): " << dtype_str;
  } else {
    // attribute [dtype] no given, infered by value's type
    CHECK(attrs.count("value"));
    auto scalar = GetScalarExpr(attrs.at("value"));
    out_type    = scalar->type();
    VLOG(3) << "FillConstant scalar type (from [vaule]): " << common::Type2Str(out_type);
  }
  return {out_type};
}

std::shared_ptr<framework::OpStrategy> StrategyForArange(const framework::NodeAttr &attrs,
                                                         const std::vector<ir::Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("start"));
  CHECK(attr_store.count("stop"));
  CHECK(attr_store.count("step"));
  CHECK(attr_store.count("dtype"));

  auto start = absl::get<float>(attr_store.at("start"));
  auto stop  = absl::get<float>(attr_store.at("stop"));
  auto step  = absl::get<float>(attr_store.at("step"));
  auto dtype = common::Str2Type(absl::get<std::string>(attr_store.at("dtype")));

  framework::CINNCompute arange_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of arange compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 1U);
    std::string tensor_name = pack_args[0].operator std::string();

    auto out = pe::Arange(start, stop, step, dtype, tensor_name);
    std::vector<common::CINNValue> res;
    auto stages = CreateStages({out});
    res.push_back(common::CINNValue(out));
    res.push_back(common::CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(arange_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.reshape.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForArange(const std::vector<std::vector<int>> &inputs_shape,
                                                  const framework::AttrMapType &attrs) {
  CHECK(attrs.count("start"));
  CHECK(attrs.count("stop"));
  CHECK(attrs.count("step"));
  float start = absl::get<float>(attrs.at("start"));
  float stop  = absl::get<float>(attrs.at("stop"));
  float step  = absl::get<float>(attrs.at("step"));
  CHECK_NE(step, 0.0f) << "The value of step can't be 0!";

  int num = static_cast<int>(std::ceil((stop - start) / step));
  CHECK(num) << "Invalid arange parameters, start = " << start << ", stop = " << stop << ", step = " << step
             << ", cause num_elem = " << num << " which is negative.";
  return {{num}};
}

std::vector<Type> InferDtypeForArange(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(attrs.count("dtype"));
  return {common::Str2Type(absl::get<std::string>(attrs.at("dtype")))};
}

std::vector<std::vector<std::string>> InferLayoutForConstant(const std::vector<framework::shape_t> &input_shapes,
                                                             const std::vector<std::string> &input_layouts,
                                                             const framework::NodeAttr &attrs,
                                                             const Target &target) {
  return {{""}, input_layouts};
}

#define EXPAND_ATTR_TYPE(MACRO) \
  MACRO(bool)                   \
  MACRO(float)                  \
  MACRO(int)                    \
  MACRO(int64_t)                \
  MACRO(double)

std::shared_ptr<OpStrategy> StrategyForAssignValue(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target) {
  framework::CINNCompute assign_value_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of assign_value compute is empty! Please check.";
    CHECK(attrs.attr_store.count("values")) << "assign_value should set attribute [values]! Please check.";
    const auto &value = attrs.attr_store.at("values");

    CINNValuePack arg_pack  = args[0];
    std::string tensor_name = UniqName("T_assign_value_out");
    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(arg_pack.size(), 1U);
      CHECK(arg_pack[0].is_string());
      tensor_name = arg_pack[0].operator std::string();
    }

    absl::optional<ir::Tensor> out;
#define EXPAND_VALUE_TO_TENSOR(TYPE)                                                            \
  else if (absl::get_if<TYPE>(&value)) {                                                        \
    out = pe::AssignValue(std::vector<TYPE>{absl::get<TYPE>(value)}, out_type[0], tensor_name); \
  }                                                                                             \
  else if (absl::get_if<std::vector<TYPE>>(&value)) {                                           \
    out = pe::AssignValue(absl::get<std::vector<TYPE>>(value), out_type[0], tensor_name);       \
  }

    if (false) {
    }
    EXPAND_ATTR_TYPE(EXPAND_VALUE_TO_TENSOR)
    else {
      LOG(FATAL) << "Assign value not support the type " << out_type[0];
    }
#undef EXPAND_VALUE_TO_TENSOR

    CHECK(out && out.value().defined()) << "can't create assign_value with the given type " << out_type[0];

    auto stages = CreateStages({out.value()});
    *ret        = CINNValuePack{{CINNValue(Expr(out.value().get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      assign_value_compute, GetElementwiseScheduleFunc(output_shapes, target), "strategy.assign_value.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForAssignValue(const std::vector<shape_t> &inputs_shape,
                                              const framework::AttrMapType &attrs) {
  CHECK(attrs.count("values")) << "assign_value should set attribute [values]! Please check.";
  const auto &value = attrs.at("values");

  shape_t shape;
#define EXPAND_ATTR_TO_GET_SHAPE(TYPE)                              \
  else if (absl::get_if<TYPE>(&value)) {                            \
    shape.emplace_back(1);                                          \
  }                                                                 \
  else if (absl::get_if<std::vector<TYPE>>(&value)) {               \
    shape.emplace_back(absl::get<std::vector<TYPE>>(value).size()); \
  }

  if (false) {
  }
  EXPAND_ATTR_TYPE(EXPAND_ATTR_TO_GET_SHAPE)
  else {
    LOG(FATAL) << "assign_value not support the type!";
  }
#undef EXPAND_ATTR_TO_GET_SHAPE

  VLOG(3) << "The output shape of assign_value is [" << cinn::utils::Join(shape, ", ") << "]";

  return {shape};
}

std::vector<Type> InferDtypeForAssignValue(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  Type out_type;
  if (attrs.find("dtype") != attrs.end()) {
    // attribute [dtype] are given
    auto dtype_str = absl::get<std::string>(attrs.at("dtype"));
    if (!dtype_str.empty()) {
      // if the [dtype] is not empty, output as the given type
      out_type = common::Str2Type(dtype_str);
    }
  }

  // attribute [dtype] not given or is empty
  if (out_type.is_unk()) {
    // infer from [values]'s dtype
    CHECK(attrs.count("values")) << "assign_value should set attribute [values]! Please check.";
    const auto &value = attrs.at("values");

#define EXPAND_ATTR_TO_GET_DTYPE(TYPE)                \
  else if (absl::get_if<TYPE>(&value)) {              \
    out_type = common::type_of<TYPE>();               \
  }                                                   \
  else if (absl::get_if<std::vector<TYPE>>(&value)) { \
    out_type = common::type_of<TYPE>();               \
  }

    if (false) {
    }
    EXPAND_ATTR_TYPE(EXPAND_ATTR_TO_GET_DTYPE)
    else {
      LOG(FATAL) << "assign_value not support the type!";
    }
#undef EXPAND_ATTR_TO_GET_DTYPE
  }

  VLOG(3) << "The data type of assign_value is " << out_type;

  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForAssignValue(const std::vector<framework::shape_t> &input_shapes,
                                                                const std::vector<std::string> &input_layouts,
                                                                const framework::NodeAttr &attrs,
                                                                const Target &target) {
  return {{""}, input_layouts};
}

#undef EXPAND_ATTR_TYPE

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(constant_ops) {
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
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

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
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(assign_value)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForAssignValue)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForAssignValue))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForAssignValue))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForAssignValue))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);
  return true;
}
