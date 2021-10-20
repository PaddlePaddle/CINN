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
#include "cinn/frontend/paddle/cpp/block_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

// Transform paddle model to CINN fronted::Program object.
// The paddle model is readed from __model__ file in model_dir, the PaddleModelConvertor
// will run each op's kernel registered in OpMapper, each kernel will add instruction in
// NetBuilder, after running all op of model, it will invoke its Build function and
// finally return the complete fronted::Program object.
// Note that if anyone op not registered, the program will failed and aborted.
class PaddleModelConvertor {
 public:
  explicit PaddleModelConvertor(hlir::framework::Scope* scope, const common::Target& target)
      : scope_(scope), target_(target) {
    CHECK(scope_);
  }

  // prepare feed variable before run CINN op
  void PrepareRun(const paddle::cpp::BlockDesc& block_desc, OpMapperContext* ctx);

  // RunOp accept OpDesc and global run context then run it's kernel registered in OpMapper.
  static void RunOp(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx);

  // operator() accept the modle's directory, and return the fronted::Program object.
  Program operator()(const std::string& model_dir, bool is_combined = false);

  // return the internal variable map
  const auto& var_map() const { return var_map_; }

  // return the map from the variable name in paddle model to cinn program.
  const auto& var_model_to_program_map() const { return var_model_to_program_map_; }

 private:
  std::unordered_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string> var_model_to_program_map_;
  hlir::framework::Scope* scope_{};
  const common::Target& target_;
};

}  // namespace frontend
}  // namespace cinn
