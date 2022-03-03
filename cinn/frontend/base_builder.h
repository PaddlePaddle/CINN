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

#include <string>
#include <vector>

#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"

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

// WARNING: In BaseBuilder, you should only place the meta op, which are also the common op between NetBuilder and
// CinnBuilder! That means, you should not place any non-meta op, or the op only belong to NetBuilder or
// CinnBuilder in BaseBuilder!
class BaseBuilder {
 public:
  explicit BaseBuilder(const std::string& name);

  Program Build();

  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");
  Placeholder CreateInput(const Variable& input);

  // name of this builder
  const std::string& name() { return name_; }

  // the number of instructions
  const size_t size() { return instrs_.size(); }

  virtual ~BaseBuilder() {}

  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  Variable Concat(const std::vector<Variable>& input_vars, int axis = 0);

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

  Variable Transpose(const Variable& operand, const std::vector<int>& axis);

  Variable Slice(const Variable& operand,
                 const std::vector<int>& axes,
                 const std::vector<int>& starts        = {},
                 const std::vector<int>& ends          = {},
                 const std::vector<int>& infer_flags   = {},
                 const std::vector<int>& decrease_axis = {});

  /**
   * This API reverses the Variable x along the given axis.
   * Example 1: x = [[0, 1], [2, 3], [4, 5]], axis = [0]
   *            output = [[4, 5], [2, 3], [0, 1]]
   * Example 2: x = [[0, 1], [2, 3], [4, 5]], axis = [0, 1]
   *            output = [[5, 4], [3, 2], [1, 0]]
   */
  Variable Reverse(const Variable& operand, const std::vector<int>& axis);

  Variable Select(const Variable& condition, const Variable& true_value, const Variable& false_value);
  std::vector<Variable> Split(const Variable& operand, const std::vector<int>& num_or_sections, int axis = 0);

  Variable IndexSelect(const Variable& x, const Variable& index, int axis = 0);

 protected:
  void InferShape(Instruction instr) const;

  Variable UnaryOp(const std::string& op_type, const Variable& operand);

  Variable BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs);

  std::string name_;
  std::vector<Instruction> instrs_;
  std::vector<Variable> inputs_;

  CINN_DISALLOW_COPY_AND_ASSIGN(BaseBuilder);
};

}  // namespace frontend
}  // namespace cinn
