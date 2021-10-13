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

#include <memory>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

// Transform paddle model to CINN NetBuilder object.
// The paddle model is readed from __model__ file in model_dir, the PaddleModelToNetBuilder
// will run each op's kernel registered in OpMapper, each kernel will add instruction in
// NetBuilder, after running all op of model, it will return the complete NetBuilder object.
// Note that if anyone op not registered, the program will failed and aborted.
class PaddleModelToNetBuilder {
 public:
  explicit PaddleModelToNetBuilder(hlir::framework::Scope* scope, const common::Target& target)
      : scope_(scope), target_(target) {
    CHECK(scope_);
  }

  // RunOp accept OpDesc and global run context then run it's kernel registered in OpMapper.
  static void RunOp(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx);

  // operator() accept the modle's directory, and return the NetBuilder object.
  std::unique_ptr<NetBuilder> operator()(const std::string& model_dir, bool is_combined = false);

  // return the internal variable map
  const absl::flat_hash_map<std::string, Variable>& var_map() const { return var_map_; }

  // return the map from the variable name in paddle model to cinn program.
  const absl::flat_hash_map<std::string, std::string>& var_model_to_program_map() { return var_model_to_program_map_; }

 private:
  absl::flat_hash_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  absl::flat_hash_map<std::string, std::string> var_model_to_program_map_;
  hlir::framework::Scope* scope_{};
  const common::Target& target_;
};

}  // namespace frontend
}  // namespace cinn
