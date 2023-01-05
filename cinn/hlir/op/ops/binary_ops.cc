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

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/layout.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

#define STRATEGY_FOR_BINARY(op_name__, pe__)                                                        \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                   \
                                                const std::vector<ir::Tensor> &inputs,              \
                                                const std::vector<Type> &out_type,                  \
                                                const std::vector<std::vector<int>> &output_shapes, \
                                                const Target &target) {                             \
    return StrategyForBinary(attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__); \
  }

std::shared_ptr<OpStrategy> StrategyForBinary(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    ir::Tensor (*pe_func)(const ir::Tensor &A, const ir::Tensor &B, const std::string &output_name, const Expr &axis)) {
  framework::CINNCompute binary_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 3U) << op_name << " 's input is not enough!";
    CHECK(pack_args[2].is_string());
    std::string tensor_name = pack_args[2].operator std::string();
    Expr A_expr             = pack_args[0];
    Expr B_expr             = pack_args[1];
    CHECK(A_expr.as_tensor());
    CHECK(B_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor B = B_expr.as_tensor_ref();
    Expr axis;
    bool trans_a;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "axis") {
        axis = Expr(absl::get<int>(iter.second));
        break;
      }
    }
    auto out    = pe_func(A, B, tensor_name, axis);
    auto stages = CreateStages({A, B, out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(binary_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy." + op_name + ".x86", 1);
  return strategy;
}

std::vector<shape_t> InferShapeForBinary(const std::vector<shape_t> &inputs_shape,
                                         const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2UL);
  std::vector<int> out_shape;

  int axis = -1;
  for (auto &iter : attrs) {
    if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
      break;
    }
  }
  VLOG(3) << "broadcast input shapes are : " << utils::Join(inputs_shape[0], ", ") << "; "
          << utils::Join(inputs_shape[1], ", ");
  pe::GetBroadcastOutShape(inputs_shape[0], inputs_shape[1], &out_shape, axis);
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  return {out_shape};
}

std::vector<Type> InferDtypeForBinary(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<Type> InferDtypeForBinaryCmp(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  return {Bool()};
}

std::vector<std::vector<std::string>> InferLayoutForBinary(const std::vector<std::vector<int>> &input_shapes,
                                                           const std::vector<std::string> &input_layouts,
                                                           const framework::NodeAttr &attrs,
                                                           const Target &target) {
  int input_size = input_layouts.size();
  CHECK(input_size == 2U || input_size == 3U) << "The input's layouts size is not 2 or 3! Please check again.";
  int axis = -1;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  std::vector<std::string> out_layouts = input_layouts;
  if (input_layouts[0].empty() && input_layouts[1].empty()) {
    return {{input_layouts[0]}, input_layouts};
  } else if (input_layouts[0].empty() || input_layouts[1].empty()) {
    int undef_idx = input_layouts[0] == "" ? 0 : 1;
    int def_idx   = 1 - undef_idx;
    CHECK_GE(input_shapes[def_idx].size(), input_shapes[undef_idx].size());
    auto ret = out_layouts[def_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  } else {
    // e.g. NCHWxc + NCHW
    ir::Layout layout0(input_layouts[0]);
    ir::Layout layout1(input_layouts[1]);
    int large_idx = layout0.ndims() >= layout1.ndims() ? 0 : 1;
    auto ret      = input_layouts[large_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  }
}

std::shared_ptr<OpStrategy> StrategyForPow(const framework::NodeAttr &attrs,
                                           const std::vector<ir::Tensor> &inputs,
                                           const std::vector<Type> &out_type,
                                           const std::vector<std::vector<int>> &output_shapes,
                                           const Target &target) {
  framework::CINNCompute pow_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pow compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 3U) << "pow 's input is not enough!";
    CHECK(pack_args[2].is_string());
    auto tensor_name = pack_args[2].operator std::string();

    Expr A_expr = pack_args[0];
    Expr B_expr = pack_args[1];
    CHECK(A_expr.as_tensor());
    CHECK(B_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor B = B_expr.as_tensor_ref();

    int axis = -1;
    if (attrs.attr_store.count("axis")) {
      axis = absl::get<int>(attrs.attr_store.at("axis"));
    }
    auto out    = pe::Pow(A, B, tensor_name, Expr(axis), target);
    auto stages = CreateStages({A, B, out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pow_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.pow.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForIsClose(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const std::vector<shape_t> &output_shapes,
                                               const Target &target) {
  float rtol = 1e-05f, atol = 1e-08f;
  bool equal_nan = false;

  if (attrs.attr_store.count("rtol")) {
    rtol = absl::get<float>(attrs.attr_store.at("rtol"));
  }
  if (attrs.attr_store.count("atol")) {
    atol = absl::get<float>(attrs.attr_store.at("atol"));
  }
  if (attrs.attr_store.count("equal_nan")) {
    equal_nan = absl::get<bool>(attrs.attr_store.at("equal_nan"));
  }

  framework::CINNCompute isclose_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of isclose compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    int input_size          = pack_args.size();

    std::string tensor_name = UniqName("IsClose_output");
    if (FLAGS_cinn_ir_schedule) {
      // the last pack argument is the output tensor name
      tensor_name = pack_args.back().operator std::string();
      --input_size;
    }
    CHECK_EQ(input_size, 2) << "The input number of isclose should be 2, but here " << input_size << "! Please check.";

    // the input tensor are in front
    Expr x_expr = pack_args[0];
    CHECK(x_expr.as_tensor());
    auto x_tensor = x_expr.as_tensor_ref();

    Expr y_expr = pack_args[1];
    CHECK(y_expr.as_tensor());
    auto y_tensor = y_expr.as_tensor_ref();

    auto out = pe::IsClose(x_tensor, y_tensor, rtol, atol, equal_nan, tensor_name);

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(isclose_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.assertisclose", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForIsClose(const std::vector<shape_t> &input_shapes,
                                          const framework::AttrMapType &attrs) {
  int input_size = input_shapes.size();
  CHECK_EQ(input_size, 2UL) << "The input number of isclose should be a multiple of 2, but here " << input_size
                            << "! Please check.";

  CHECK(input_shapes[0] == input_shapes[1])
      << "The two inputs shape of isclose should be equal, but here x:[" << cinn::utils::Join(input_shapes[0], ",")
      << "] != y:[" << cinn::utils::Join(input_shapes[1], ",") << "] ! Please check.";
  return {input_shapes[0]};
}

std::vector<Type> InferDtypeForIsClose(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  int input_size = inputs_type.size();
  CHECK_EQ(input_size, 2UL) << "The input number of isclose should be a multiple of 2, but here " << input_size
                            << "! Please check.";
  CHECK(inputs_type[0] == inputs_type[1])
      << "The two inputs dtype sof isclose should be equal, but here x:" << inputs_type[0] << " != y:" << inputs_type[1]
      << "! Please check.";

  return {Bool()};
}

std::vector<std::vector<std::string>> InferLayoutForIsClose(const std::vector<std::vector<int>> &input_shapes,
                                                            const std::vector<std::string> &input_layouts,
                                                            const framework::NodeAttr &attrs,
                                                            const Target &target) {
  return {{""}, input_layouts};
}

STRATEGY_FOR_BINARY(elementwise_add, Add);
STRATEGY_FOR_BINARY(atan2, Atan2);
STRATEGY_FOR_BINARY(elementwise_mul, Multiply);

STRATEGY_FOR_BINARY(substract, Substract);
STRATEGY_FOR_BINARY(divide, Divide);
STRATEGY_FOR_BINARY(floor_divide, FloorDivide);
STRATEGY_FOR_BINARY(mod, Mod);
STRATEGY_FOR_BINARY(remainder, Remainder);
STRATEGY_FOR_BINARY(max, Maximum);
STRATEGY_FOR_BINARY(min, Minimum);
STRATEGY_FOR_BINARY(logical_and, LogicalAnd);
STRATEGY_FOR_BINARY(logical_or, LogicalOr);
STRATEGY_FOR_BINARY(logical_xor, LogicalXOr);
STRATEGY_FOR_BINARY(greater, Greater);
STRATEGY_FOR_BINARY(less, Less);
STRATEGY_FOR_BINARY(equal, Equal);
STRATEGY_FOR_BINARY(not_equal, NotEqual);
STRATEGY_FOR_BINARY(greater_equal, GreaterEqual);
STRATEGY_FOR_BINARY(less_equal, LessEqual);

STRATEGY_FOR_BINARY(bitwise_or, BitwiseOr);
STRATEGY_FOR_BINARY(bitwise_xor, BitwiseXor);
STRATEGY_FOR_BINARY(bitwise_and, BitwiseAnd);
STRATEGY_FOR_BINARY(left_shift, LeftShift);
STRATEGY_FOR_BINARY(right_shift, RightShift);

#undef STRATEGY_FOR_BINARY

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(binary_ops) {
#define CINN_REGISTER_BINARY(op__, op_stragegy__)                                                                      \
  CINN_REGISTER_OP(op__)                                                                                               \
      .describe(#op__ " function")                                                                                     \
      .set_num_inputs(1)                                                                                               \
      .set_num_outputs(1)                                                                                              \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)   \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBinary))                                     \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBinary))                                     \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForBinary))                                   \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
      .set_support_level(4);

#define CINN_REGISTER_BINARY_CMP(op__, op_stragegy__)                                                                  \
  CINN_REGISTER_OP(op__)                                                                                               \
      .describe(#op__ " function")                                                                                     \
      .set_num_inputs(1)                                                                                               \
      .set_num_outputs(1)                                                                                              \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)   \
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBinary))                                     \
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBinaryCmp))                                  \
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForBinary))                                   \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
      .set_support_level(4);

  CINN_REGISTER_BINARY(elementwise_add, Add);
  CINN_REGISTER_BINARY(atan2, Atan2);
  CINN_REGISTER_BINARY(elementwise_mul, Multiply);

  CINN_REGISTER_BINARY(substract, Substract);
  CINN_REGISTER_BINARY(divide, Divide);
  CINN_REGISTER_BINARY(floor_divide, FloorDivide);
  CINN_REGISTER_BINARY(mod, Mod);
  CINN_REGISTER_BINARY(remainder, Remainder);
  CINN_REGISTER_BINARY(max, Maximum);
  CINN_REGISTER_BINARY(min, Minimum);

  CINN_REGISTER_BINARY_CMP(logical_and, LogicalAnd);
  CINN_REGISTER_BINARY_CMP(logical_or, LogicalOr);
  CINN_REGISTER_BINARY_CMP(logical_xor, LogicalXOr);
  CINN_REGISTER_BINARY_CMP(greater, Greater);
  CINN_REGISTER_BINARY_CMP(less, Less);
  CINN_REGISTER_BINARY_CMP(equal, Equal);
  CINN_REGISTER_BINARY_CMP(not_equal, NotEqual);
  CINN_REGISTER_BINARY_CMP(greater_equal, GreaterEqual);
  CINN_REGISTER_BINARY_CMP(less_equal, LessEqual);

  CINN_REGISTER_BINARY(bitwise_or, BitwiseOr);
  CINN_REGISTER_BINARY(bitwise_xor, BitwiseXor);
  CINN_REGISTER_BINARY(bitwise_and, BitwiseAnd);
  CINN_REGISTER_BINARY(left_shift, LeftShift);
  CINN_REGISTER_BINARY(right_shift, RightShift);
#undef CINN_REGISTER_BINARY

  CINN_REGISTER_OP(pow)
      .describe("pow op")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForPow)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForBinary))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForBinary))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForBinary))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(isclose)
      .describe("This operator checks if all x and y satisfy the condition: |x - y| <= atol + rtol * |y|")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForIsClose)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForIsClose))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForIsClose))
      .set_attr("inferlayout", MakeOpFunction(cinn::hlir::op::InferLayoutForIsClose))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
