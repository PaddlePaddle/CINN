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

#include "cinn/frontend/op_mapper_registry.h"

#include "cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn {
namespace frontend {

void OpMapperContext::AddVar(const std::string& origin_name, const Variable& var, bool replace) const {
  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);
  CHECK(replace || !var_map_->count(name)) << "Duplicate variable [" << name << "] found";
  (*var_map_)[name] = var;
  VLOG(4) << "Add variable [" << name << "] with shape " << cinn::utils::Join(var->shape, ",");
}

void OpMapperContext::AddVarModelToProgram(const std::string& name, const std::string& id) const {
  (*var_model_to_program_map_)[name] = id;
  VLOG(4) << "Paddle name [" << name << "] map to program id " << id;
}

Variable OpMapperContext::GetVar(const std::string& origin_name) const {
  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);

  auto it = var_map_->find(name);
  if (it != var_map_->end()) return it->second;

  auto* var = scope_.FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    Variable var;
    var.set_id(name);
    var->shape = tensor->shape().data();
    var->type  = tensor->type();
    AddVar(name, var);
    return var;
  }

  LOG(FATAL) << "No var called [" << name << "] exists";
  return Variable();
}

void OpMapperContext::AddVarInfo(const std::string& name, const VarInfo& info) {
  CHECK(!var_info_map_.count(name)) << "Duplicate variable info [" << name << "] found";
  var_info_map_[name] = info;
}

const OpMapperContext::VarInfo& OpMapperContext::GetVarInfo(const std::string& name) const {
  CHECK(var_info_map_.count(name)) << "No variable info called [" << name << "] exists";
  return var_info_map_.at(name);
}

}  // namespace frontend
}  // namespace cinn
