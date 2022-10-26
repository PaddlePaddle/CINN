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
  CHECK(replace || !var_map_->count(origin_name))
      << "Duplicate variable [" << origin_name << "] found, whose id is " << var_map_->at(origin_name)->id;
  (*var_map_)[origin_name] = var;
  VLOG(4) << "Add variable [" << origin_name << "] with shape " << cinn::utils::Join(var->shape, ",");
}

void OpMapperContext::AddVarModelToProgram(const std::string& name, const std::string& id) const {
  CHECK(!id.empty()) << "Paddle name [" << name << "]'s program id is empty ! Please check.";
  (*var_model_to_program_map_)[name] = id;
  VLOG(4) << "Paddle name [" << name << "] map to program id " << id;
}

void OpMapperContext::AddFetchVarName(const std::string& name) const { fetch_var_names_->insert(name); }

Variable OpMapperContext::GetVar(const std::string& origin_name) const {
  auto it = var_map_->find(origin_name);
  if (it != var_map_->end()) return it->second;

  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);

  auto* var = scope_.FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    Variable local_var;
    local_var.set_id(name);
    local_var->shape = tensor->shape().data();
    local_var->type  = tensor->type();
    AddVar(origin_name, local_var);
    return local_var;
  }

  LOG(FATAL) << "No var called [" << origin_name << "] exists";
  return Variable();
}

void OpMapperContext::AddFeedInfo(const std::string& name, const FeedInfo& info) {
  CHECK(!feed_info_map_.count(name)) << "Duplicate variable info [" << name << "] found";
  feed_info_map_[name] = info;
}

const OpMapperContext::FeedInfo& OpMapperContext::GetFeedInfo(const std::string& name) const {
  CHECK(feed_info_map_.count(name)) << "No variable info called [" << name << "] exists";
  return feed_info_map_.at(name);
}

}  // namespace frontend
}  // namespace cinn
