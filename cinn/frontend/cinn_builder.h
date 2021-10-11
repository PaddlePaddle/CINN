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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cinn/common/type.h"
#include "cinn/frontend/base_builder.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

enum class ComparisonKind : std::int8_t {
  kUnk = -1,
  kEq,
  kNe,
  kGe,
  kGt,
  kLe,
  kLt,
};

enum class ReduceKind : std::int8_t {
  kUnk = -1,
  kSum,
  kProd,
  kMax,
  kMin,
};

class CinnBuilder : public BaseBuilder {
 public:
  using BaseBuilder::BaseBuilder;

  /**
   * @brief Create scalar with the specific value and type.
   * @param value The scalar value to be set.
   * @param name The name of output variable.
   * @return The result variable.
   */
  template <typename T>
  Variable ConstScalar(T value, const std::string& name) {
    Instruction instr("const_scalar");
    instr.SetInputs({});
    instr.SetAttr("value", value);
    InferShape(instr);
    AppendInstruction(instr);
    auto out = instr.GetOutput(0);
    out.set_id(name);
    auto out_type = type_of<T>();
    CHECK(out_type.is_float() || out_type.is_int() || out_type.is_bool()) << "no supported type: " << out_type;
    out->type = out_type;
    return out;
  }

#define UNARY_OP_DECL(func_name__) Variable func_name__(const Variable& operand);
  UNARY_OP_DECL(Exp)
  UNARY_OP_DECL(Erf)
  UNARY_OP_DECL(Sqrt)
  UNARY_OP_DECL(Rsqrt)
  UNARY_OP_DECL(Log)
  UNARY_OP_DECL(Log2)
  UNARY_OP_DECL(Log10)
  UNARY_OP_DECL(Floor)
  UNARY_OP_DECL(Ceil)
  UNARY_OP_DECL(Round)
  UNARY_OP_DECL(Trunc)
  UNARY_OP_DECL(Sin)
  UNARY_OP_DECL(Cos)
  UNARY_OP_DECL(Tan)
  UNARY_OP_DECL(Sinh)
  UNARY_OP_DECL(Cosh)
  UNARY_OP_DECL(Tanh)
  UNARY_OP_DECL(Asin)
  UNARY_OP_DECL(Acos)
  UNARY_OP_DECL(Atan)
  UNARY_OP_DECL(Asinh)
  UNARY_OP_DECL(Acosh)
  UNARY_OP_DECL(Atanh)
  UNARY_OP_DECL(IsNan)
  UNARY_OP_DECL(IsFinite)
  UNARY_OP_DECL(IsInf)
  UNARY_OP_DECL(LogicalNot)
  UNARY_OP_DECL(BitwiseNot)
  UNARY_OP_DECL(Negative)
  UNARY_OP_DECL(Sign)
  UNARY_OP_DECL(Abs)
  UNARY_OP_DECL(Identity)
#undef UNARY_OP_DECL

#define BINARY_OP_DECL(func_name__) Variable func_name__(const Variable& lhs, const Variable& rhs);
  BINARY_OP_DECL(Dot)
  BINARY_OP_DECL(Add)
  BINARY_OP_DECL(Sub)
  BINARY_OP_DECL(Mul)
  BINARY_OP_DECL(Div)
  BINARY_OP_DECL(FloorDiv)
  BINARY_OP_DECL(Mod)
  BINARY_OP_DECL(FloorMod)
  BINARY_OP_DECL(Max)
  BINARY_OP_DECL(Min)
  BINARY_OP_DECL(Power)
  BINARY_OP_DECL(LogicalAnd)
  BINARY_OP_DECL(LogicalOr)
  BINARY_OP_DECL(LogicalXor)
  BINARY_OP_DECL(BitwiseAnd)
  BINARY_OP_DECL(BitwiseOr)
  BINARY_OP_DECL(BitwiseXor)
  BINARY_OP_DECL(LeftShift)
  BINARY_OP_DECL(RightShift)
#undef BINARY_OP_DECL

  Variable Concat(const Variable& lhs, const Variable& rhs, int axis = 0);

  Variable Conv(const Variable& lhs,
                const Variable& rhs,
                const std::vector<int>& strides      = {1, 1},
                const std::vector<int>& paddings     = {0, 0},
                const std::vector<int>& dilations    = {1, 1},
                int groups                           = 1,
                const std::string& data_format       = "NCHW",
                const std::string& padding_algorithm = "EXPLICIT");

  Variable Compare(const Variable& lhs, const Variable& rhs, ComparisonKind kind);

  /**
   * @brief Reduce array elements over the given dims.
   *
   * @param operand The input variable.
   * @param dim The dims along which a sum is performed. If dim is empty, the operation will sum over all elements
   * of the input array. If the dim has negative value, it should count from the last dim to the first.
   * @param keep_dim If it is set true, the axes which are reduced are left in the result as dimensions with size one.
   * With this option, the result will broadcast correctly against the input array.
   *
   * @return The result variable.
   */
  Variable Reduce(const Variable& operand, ReduceKind kind, const std::vector<int>& dim, bool keep_dim = false);

  Variable BroadcastTo(const Variable& operand,
                       const std::vector<int>& out_shape,
                       const std::vector<int>& broadcast_axes);

  Variable Reshape(const Variable& operand, const std::vector<int>& shape);

  Variable Slice(const Variable& operand,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts = {},
                 const std::vector<int>& ends   = {});

  Variable Select(const Variable& condition, const Variable& true_value, const Variable& false_value);

  Variable Reverse(const Variable& operand, const std::vector<int>& axis);

 private:
  Variable UnaryOp(const std::string& op_type, const Variable& operand);

  Variable BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs);
};

}  // namespace frontend
}  // namespace cinn
