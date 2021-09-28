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
#include "cinn/hlir/op/use_ops.h"

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

void BaseBuilder::InferShape(Instruction instr) const {
  using shape_func_t        = std::function<std::vector<shape_t>(const std::vector<shape_t>&, AttrMapType&)>;
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
