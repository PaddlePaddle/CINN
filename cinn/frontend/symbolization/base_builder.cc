#include "cinn/frontend/symbolization/base_builder.h"

#include <utility>

namespace cinn {
namespace frontend {
namespace symbolization {
Program BaseBuilder::Build() { return Program{std::move(instrs_), std::move(inputs_)}; }

Placeholder BaseBuilder::CreateInput(const common::Type& type,
                                     const std::vector<int>& shape,
                                     const std::string& id_hint) {
  inputs_.emplace_back(id_hint);
  auto& var  = inputs_.back();
  var->type  = type;
  var->shape = shape;
  return Placeholder(var);
}
}  // namespace symbolization
}  // namespace frontend
}  // namespace cinn
