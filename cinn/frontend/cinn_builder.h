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

// clang-format off
#define UNARY_OP_FOREACH(macro__)         \
    macro__(Exp)                          \
    macro__(Erf)                          \
    macro__(Sqrt)                         \
    macro__(Rsqrt)                        \
    macro__(Log)                          \
    macro__(Log2)                         \
    macro__(Log10)                        \
    macro__(Floor)                        \
    macro__(Ceil)                         \
    macro__(Round)                        \
    macro__(Trunc)                        \
    macro__(Sin)                          \
    macro__(Cos)                          \
    macro__(Tan)                          \
    macro__(Sinh)                         \
    macro__(Cosh)                         \
    macro__(Tanh)                         \
    macro__(Asin)                         \
    macro__(Acos)                         \
    macro__(Atan)                         \
    macro__(Asinh)                        \
    macro__(Acosh)                        \
    macro__(Atanh)                        \
    macro__(IsNan)                        \
    macro__(IsFinite)                     \
    macro__(IsInf)                        \
    macro__(LogicalNot)                   \
    macro__(BitwiseNot)                   \
    macro__(Negative)                     \
    macro__(Sign)                         \
    macro__(Abs)                          \
    macro__(Identity)

#define BINARY_OP_FOREACH(macro__)        \
    macro__(Dot)                          \
    macro__(Add)                          \
    macro__(Sub)                          \
    macro__(Mul)                          \
    macro__(Div)                          \
    macro__(FloorDiv)                     \
    macro__(Mod)                          \
    macro__(FloorMod)                     \
    macro__(Max)                          \
    macro__(Min)                          \
    macro__(Power)                        \
    macro__(LogicalAnd)                   \
    macro__(LogicalOr)                    \
    macro__(LogicalXor)                   \
    macro__(BitwiseAnd)                   \
    macro__(BitwiseOr)                    \
    macro__(BitwiseXor)                   \
    macro__(LeftShift)                    \
    macro__(RightShift)
// clang-format on
namespace cinn {
namespace frontend {

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
    instr.SetAttr<T>("value", value);
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
  UNARY_OP_FOREACH(UNARY_OP_DECL)
#undef UNARY_OP_DECL

#define BINARY_OP_DECL(func_name__) Variable func_name__(const Variable& lhs, const Variable& rhs);
  BINARY_OP_FOREACH(BINARY_OP_DECL)
#undef BINARY_OP_DECL

  Variable Conv(const Variable& lhs,
                const Variable& rhs,
                const std::vector<int>& strides      = {1, 1},
                const std::vector<int>& paddings     = {0, 0},
                const std::vector<int>& dilations    = {1, 1},
                int groups                           = 1,
                const std::string& conv_type         = "forward",
                const std::string& data_format       = "NCHW",
                const std::string& padding_algorithm = "EXPLICIT",
                const std::vector<int>& output_shape = {});

  Variable Compare(const Variable& lhs, const Variable& rhs, ComparisonKind kind);

  std::vector<Variable> BnMeanVariance(const Variable& x);

  std::vector<Variable> BnGradBiasScale(const Variable& x, const Variable& x_mean, const Variable& y_grad);
};

}  // namespace frontend
}  // namespace cinn
