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

#include "cinn/frontend/cinn_builder.h"

#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

#define UNARY_OP_DEF(func_name__, op_type__) \
  Variable CinnBuilder::func_name__(const Variable& operand) { return UnaryOp(#op_type__, operand); }
UNARY_OP_DEF(Exp, exp)
UNARY_OP_DEF(Erf, erf)
UNARY_OP_DEF(Sqrt, sqrt)
UNARY_OP_DEF(Rsqrt, rsqrt)
UNARY_OP_DEF(Log, log)
UNARY_OP_DEF(Log2, log2)
UNARY_OP_DEF(Log10, log10)
UNARY_OP_DEF(Floor, floor)
UNARY_OP_DEF(Ceil, ceil)
UNARY_OP_DEF(Round, round)
UNARY_OP_DEF(Trunc, trunc)
UNARY_OP_DEF(Sin, sin)
UNARY_OP_DEF(Cos, cos)
UNARY_OP_DEF(Tan, tan)
UNARY_OP_DEF(Sinh, sinh)
UNARY_OP_DEF(Cosh, cosh)
UNARY_OP_DEF(Tanh, tanh)
UNARY_OP_DEF(Asin, asin)
UNARY_OP_DEF(Acos, acos)
UNARY_OP_DEF(Atan, atan)
UNARY_OP_DEF(Asinh, asinh)
UNARY_OP_DEF(Acosh, acosh)
UNARY_OP_DEF(Atanh, atanh)
UNARY_OP_DEF(IsNan, isnan)
UNARY_OP_DEF(IsFinite, isfinite)
UNARY_OP_DEF(IsInf, isinf)
UNARY_OP_DEF(LogicalNot, logical_not)
UNARY_OP_DEF(BitwiseNot, bitwise_not)
UNARY_OP_DEF(Negative, negative)
UNARY_OP_DEF(Sign, sign)
UNARY_OP_DEF(Abs, abs)
UNARY_OP_DEF(Identity, identity)
#undef UNARY_OP_DEF

#define BINARY_OP_DEF(func_name__, op_type__) \
  Variable CinnBuilder::func_name__(const Variable& lhs, const Variable& rhs) { return BinaryOp(#op_type__, lhs, rhs); }
BINARY_OP_DEF(Dot, matmul)
BINARY_OP_DEF(Add, elementwise_add)
BINARY_OP_DEF(Sub, substract)
BINARY_OP_DEF(Mul, elementwise_mul)
BINARY_OP_DEF(Div, divide)
BINARY_OP_DEF(FloorDiv, floor_divide)
BINARY_OP_DEF(Mod, mod)
BINARY_OP_DEF(FloorMod, floor_mod)
BINARY_OP_DEF(Max, max)
BINARY_OP_DEF(Min, min)
BINARY_OP_DEF(Power, power)
BINARY_OP_DEF(LogicalAnd, logical_and)
BINARY_OP_DEF(LogicalOr, logical_or)
BINARY_OP_DEF(LogicalXor, logical_xor)
BINARY_OP_DEF(BitwiseAnd, bitwise_and)
BINARY_OP_DEF(BitwiseOr, bitwise_or)
BINARY_OP_DEF(BitwiseXor, bitwise_xor)
BINARY_OP_DEF(LeftShift, left_shift)
BINARY_OP_DEF(RightShift, right_shift)
#undef BINARY_OP_DEF

Variable CinnBuilder::Conv(const Variable& lhs,
                           const Variable& rhs,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& dilations,
                           int groups,
                           const std::string& conv_type,
                           const std::string& data_format,
                           const std::string& padding_algorithm,
                           const std::vector<int>& output_shape) {
  Instruction instr("conv2d");
  instr.SetInputs({lhs, rhs});
  instr.SetAttr("stride", strides);
  instr.SetAttr("padding", paddings);
  instr.SetAttr("dilation", dilations);
  instr.SetAttr("groups", groups);
  instr.SetAttr("conv_type", conv_type);
  instr.SetAttr("data_format", data_format);
  instr.SetAttr("padding_algorithm", padding_algorithm);
  instr.SetAttr("output_shape", output_shape);

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable CinnBuilder::Compare(const Variable& lhs, const Variable& rhs, ComparisonKind kind) {
  switch (kind) {
    case ComparisonKind::kEq:
      return BinaryOp("equal", lhs, rhs);
    case ComparisonKind::kNe:
      return BinaryOp("not_equal", lhs, rhs);
    case ComparisonKind::kGe:
      return BinaryOp("greater_equal", lhs, rhs);
    case ComparisonKind::kGt:
      return BinaryOp("greater", lhs, rhs);
    case ComparisonKind::kLe:
      return BinaryOp("less_equal", lhs, rhs);
    case ComparisonKind::kLt:
      return BinaryOp("less", lhs, rhs);
    default:
      LOG(FATAL) << "unknown comparison kind";
  }
}

const std::vector<Variable>& CinnBuilder::CustomInstr(const std::string& type,
                                                      const std::vector<Variable>& inputs,
                                                      const AttributeMap& attrs) {
  Instruction instr(type);
  instr.SetInputs(inputs);
  for (auto& kv : attrs) {
    instr.SetAttr(kv.first, kv.second);
  }

  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

std::vector<Variable> CinnBuilder::BnMeanVariance(const Variable& x) {
  Instruction instr("bn_mean_variance", {x});
  // optimize bn forward reduce computation, set reduce dimension(NCHW suppport only, to be deprecated).
  instr.SetAttr("dim", std::vector<int>{0, 2, 3});
  instr.SetAttr("keep_dim", false);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

std::vector<Variable> CinnBuilder::BnGradBiasScale(const Variable& x, const Variable& x_mean, const Variable& y_grad) {
  Instruction instr("bn_grad_bias_scale", {x, x_mean, y_grad});
  // optimize bn backward reduce computation, set reduce dimension(NCHW suppport only, to be deprecated).
  instr.SetAttr("dim", std::vector<int>{0, 2, 3});
  instr.SetAttr("keep_dim", false);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

}  // namespace frontend
}  // namespace cinn
