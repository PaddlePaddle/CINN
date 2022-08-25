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

#include "cinn/frontend/base_builder.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {

using common::Context;
using common::Type;
using hlir::framework::Operator;
using utils::AttributeMap;
using utils::ShapeType;

BaseBuilder::BaseBuilder(const std::string& name) : name_(name) {}

Program BaseBuilder::Build(bool in_reverse) {
  std::vector<Instruction> instrs;
  if (in_reverse) {
    instrs.reserve(instrs_.size());
    for (auto it = instrs_.rbegin(); it != instrs_.rend(); it++) {
      instrs.emplace_back(*it);
    }
  } else {
    instrs = std::move(instrs_);
  }

  Program program{std::move(instrs), std::move(inputs_)};
  program.Validate();
  return program;
}

Placeholder BaseBuilder::CreateInput(const Type& type, const std::vector<int>& shape, const std::string& id_hint) {
  if (!id_hint.empty()) {
    CheckVarNameValid(id_hint);
  }
  std::string id = id_hint.empty() ? Context::Global().NewName("placeholder") : id_hint;

  inputs_.emplace_back(id);
  auto& var  = inputs_.back();
  var->type  = type;
  var->shape = shape;
  return Placeholder(var);
}

Placeholder BaseBuilder::CreateInput(const Variable& var) {
  CHECK(!var->shape.empty()) << "The input's shape is not set yet";
  CHECK(!var->type.is_unk()) << "The input's type is not set yet";
  inputs_.push_back(var);
  return Placeholder(var);
}

void BaseBuilder::InferShape(Instruction instr) const {
  using ShapeFunc           = std::function<std::vector<ShapeType>(const std::vector<ShapeType>&, const AttributeMap&)>;
  using TypeFunc            = std::function<std::vector<Type>(const std::vector<Type>&, const AttributeMap&)>;
  const auto& op_infershape = Operator::GetAttrs<ShapeFunc>("infershape");
  const auto& op_inferdtype = Operator::GetAttrs<TypeFunc>("inferdtype");

  size_t size = instr->inputs.size();
  std::vector<ShapeType> in_shapes(size);
  std::vector<Type> in_types(size);
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_shapes.begin(), [](const Variable& var) { return var->shape; });
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_types.begin(), [](const Variable& var) { return var->type; });

  auto key        = Operator::Get(instr->op_type);
  auto out_shapes = op_infershape[key](in_shapes, instr->attrs);
  auto out_types  = op_inferdtype[key](in_types, instr->attrs);

  auto& outs            = instr->outputs;
  size_t origin_out_num = outs.size();
  outs.resize(out_shapes.size());
  for (size_t i = origin_out_num; i < outs.size(); i++) {
    outs[i] = Variable();
  }
  for (size_t i = 0; i < outs.size(); i++) {
    outs[i]->shape = out_shapes[i];
    outs[i]->type  = out_types[i];
  }
}

std::vector<Variable> BaseBuilder::Split(const Variable& operand, const std::vector<int>& num_or_sections, int axis) {
  Instruction instr("split", {operand});
  instr.SetAttr("num_or_sections", num_or_sections);
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable BaseBuilder::Concat(const std::vector<Variable>& input_vars, int axis) {
  Instruction instr("concat", input_vars);
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Reduce(const Variable& operand, ReduceKind kind, const std::vector<int>& dim, bool keep_dim) {
  auto reduce_func = [&](const std::string& op_type) {
    Instruction instr(op_type, {operand});
    instr.SetAttr("dim", dim);
    instr.SetAttr("keep_dim", keep_dim);
    InferShape(instr);
    AppendInstruction(instr);
    return instr.GetOutput(0);
  };

  switch (kind) {
    case ReduceKind::kSum:
      return reduce_func("reduce_sum");
    case ReduceKind::kProd:
      return reduce_func("reduce_prod");
    case ReduceKind::kMax:
      return reduce_func("reduce_max");
    case ReduceKind::kMin:
      return reduce_func("reduce_min");
    case ReduceKind::kAll:
      return reduce_func("reduce_all");
    case ReduceKind::kAny:
      return reduce_func("reduce_any");
    default:
      LOG(FATAL) << "unknown reduction kind";
  }
}

Variable BaseBuilder::FillConstant(
    const std::vector<int>& shape, float value, const std::string& name, const std::string& dtype, bool force_cpu) {
  Instruction instr("fill_constant");
  instr.SetInputs({});
  instr.SetAttr("shape", shape);
  instr.SetAttr("value", value);
  instr.SetAttr("dtype", dtype);
  instr.SetAttr("force_cpu", force_cpu);

  InferShape(instr);
  AppendInstruction(instr);
  auto out = instr.GetOutput(0);
  out.set_id(name);
  return out;
}

Variable BaseBuilder::BroadcastTo(const Variable& operand, const std::vector<int>& out_shape) {
  auto x_shape_size = operand->shape.size();
  auto y_shape_size = out_shape.size();
  CHECK_GT(x_shape_size, 0) << "Cannot broadcast a empty operand " << operand->id << " to "
                            << cinn::utils::Join(out_shape, ",");
  CHECK_LE(x_shape_size, y_shape_size) << "The broadcast_p's input shape dimension should less than the output's, "
                                       << "but here (" << x_shape_size << " > " << y_shape_size << ").";

  VLOG(4) << "Try broadcast " << operand->id << " from shape (" << cinn::utils::Join(operand->shape, ",")
          << ") to shape (" << cinn::utils::Join(out_shape, ",") << ")";

  std::vector<int> broadcast_axes(x_shape_size, 0);
  if (x_shape_size > 1) {
    for (int i = 1; i <= x_shape_size; ++i) {
      CHECK((out_shape[y_shape_size - i] == operand->shape[x_shape_size - i]) ||
            (operand->shape[x_shape_size - i] == 1))
          << "We cannot broadcast from shape (" << cinn::utils::Join(operand->shape, ",") << ") to shape ("
          << cinn::utils::Join(out_shape, ",") << ")";
      broadcast_axes[x_shape_size - i] = y_shape_size - i;
    }
  } else {
    int axis     = -1;
    auto x_shape = operand->shape.at(0);
    if (x_shape == 1) {
      // Can broadcast directly, default axis 0
      axis = 0;
    } else {
      // The broadcast axes is the index of the shape in out_shape when the input dimension is 1
      for (int i = 0; i < y_shape_size; ++i) {
        if (out_shape[i] == x_shape) {
          axis = i;
          break;
        }
      }
      CHECK_NE(axis, -1) << "When we broadcast a 1-dimension shape, the number should contained in the out_shape. "
                         << "We cannot broadcast from shape (" << cinn::utils::Join(operand->shape, ",")
                         << ") to shape (" << cinn::utils::Join(out_shape, ",") << ")";
    }
    broadcast_axes[0] = axis;
  }

  return BroadcastTo(operand, out_shape, broadcast_axes);
}

Variable BaseBuilder::BroadcastTo(const Variable& operand,
                                  const std::vector<int>& out_shape,
                                  const std::vector<int>& broadcast_axes) {
  Instruction instr("broadcast_to", {operand});
  instr.SetAttr("out_shape", out_shape);
  instr.SetAttr("broadcast_axes", broadcast_axes);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Reshape(const Variable& operand, const std::vector<int>& shape) {
  Instruction instr("reshape", {operand});
  instr.SetAttr("shape", shape);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Transpose(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("transpose", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Slice(const Variable& operand,
                            const std::vector<int>& axes,
                            const std::vector<int>& starts,
                            const std::vector<int>& ends,
                            const std::vector<int>& infer_flags,
                            const std::vector<int>& strides) {
  Instruction instr("slice", {operand});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  instr.SetAttr("infer_flags", infer_flags);
  instr.SetAttr("strides", strides);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::SliceAssign(const Variable& input,
                                  const Variable& assign,
                                  const std::vector<int>& axes,
                                  const std::vector<int>& starts,
                                  const std::vector<int>& ends,
                                  const std::vector<int>& strides) {
  Instruction instr("slice_assign", {input, assign});
  instr.SetAttr("axes", axes);
  instr.SetAttr("starts", starts);
  instr.SetAttr("ends", ends);
  instr.SetAttr("strides", strides);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Reverse(const Variable& operand, const std::vector<int>& axis) {
  Instruction instr("reverse", {operand});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::Select(const Variable& condition, const Variable& true_value, const Variable& false_value) {
  Instruction instr("select", {condition, true_value, false_value});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::IndexSelect(const Variable& operand, const Variable& index, int axis) {
  Instruction instr("index_select", {operand, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::ScatterAssign(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  Instruction instr("scatter_assign", {operand, updates, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::ScatterAdd(const Variable& operand, const Variable& updates, const Variable& index, int axis) {
  Instruction instr("scatter_add", {operand, updates, index});
  instr.SetAttr("axis", axis);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::UnaryOp(const std::string& op_type, const Variable& operand) {
  Instruction instr(op_type, {operand});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable BaseBuilder::BinaryOp(const std::string& op_type, const Variable& lhs, const Variable& rhs) {
  Instruction instr(op_type, {lhs, rhs});
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn
