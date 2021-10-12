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
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace frontend {

using common::Context;
using common::Type;
using hlir::framework::AttrMapType;
using hlir::framework::Operator;
using hlir::framework::shape_t;

BaseBuilder::BaseBuilder(const std::string& name) : name_(name) {}

Program BaseBuilder::Build() {
  Program program{std::move(instrs_), std::move(inputs_)};
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

void BaseBuilder::SetInputs(const std::vector<Variable>& inputs) {
  CHECK(inputs_.empty()) << "The original inputs is not empty, which doesn't support to be set.";
  CHECK(!inputs.empty()) << "At least one input is needed for building a program!";
  for (int i = 0; i < inputs.size(); i++) {
    CHECK(!inputs[i]->shape.empty()) << "Found " << i << "-th input's shape is not set yet";
    CHECK(!inputs[i]->type.is_unk()) << "Found " << i << "-th input's type is not set yet";
    inputs_.push_back(inputs[i]);
  }
}

void BaseBuilder::InferShape(Instruction instr) const {
  using shape_func_t        = std::function<std::vector<shape_t>(const std::vector<shape_t>&, const AttrMapType&)>;
  using type_func_t         = std::function<std::vector<Type>(const std::vector<Type>&, const AttrMapType&)>;
  const auto& op_infershape = Operator::GetAttrs<shape_func_t>("infershape");
  const auto& op_inferdtype = Operator::GetAttrs<type_func_t>("inferdtype");

  size_t size = instr->inputs.size();
  std::vector<shape_t> in_shapes(size);
  std::vector<Type> in_types(size);
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_shapes.begin(), [](const Variable& var) { return var->shape; });
  std::transform(
      instr->inputs.begin(), instr->inputs.end(), in_types.begin(), [](const Variable& var) { return var->type; });

  auto key        = Operator::Get(instr->op_type);
  auto out_shapes = op_infershape[key](in_shapes, instr->attrs);
  auto out_types  = op_inferdtype[key](in_types, instr->attrs);

  auto& outs = instr->outputs;
  for (size_t i = 0; i < outs.size(); i++) {
    outs[i]->shape = out_shapes[i];
    outs[i]->type  = out_types[i];
  }
}

}  // namespace frontend
}  // namespace cinn
