#include "cinn/frontend/base_builder.h"

#include <string>
#include <utility>

#include "cinn/common/common.h"
#include "cinn/common/context.h"

namespace cinn {
namespace frontend {

Program BaseBuilder::Build() {
  Program program{std::move(instrs_), std::move(inputs_)};
  program.Validate();
  return program;
}

Placeholder BaseBuilder::CreateInput(const common::Type& type,
                                     const std::vector<int>& shape,
                                     const std::string& id_hint) {
  if (!id_hint.empty()) {
    CheckVarNameValid(id_hint);
  }
  std::string id = id_hint.empty() ? common::Context::Global().NewName("placeholder") : id_hint;

  inputs_.emplace_back(id);
  auto& var  = inputs_.back();
  var->type  = type;
  var->shape = shape;
  return Placeholder(var);
}

}  // namespace frontend
}  // namespace cinn
