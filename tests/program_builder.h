// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/common/type.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/syntax.h"
#include "cinn/utils/type_defs.h"

namespace cinn {
namespace tests {

struct VariableInfo {
  std::string id;
  std::vector<int> shape;
  common::Type type;
  VariableInfo(std::string name, std::vector<int> shape, common::Type dtype = common::Float(32))
      : id(name), shape(shape), type(dtype) {}
};

class ProgramBuilder {
 public:
  ProgramBuilder(const std::string& name) : builder_(name) {}

  virtual frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                                  const utils::AttributeMap& attrs) = 0;

  const std::vector<frontend::Variable>& GetOutputs() const { return outputs_; }

 protected:
  void AddOutput(frontend::Variable var) { outputs_.emplace_back(var); }

  frontend::NetBuilder builder_;
  common::Type dtype_;
  std::vector<frontend::Variable> outputs_;
};

class OpBuilder final : public ProgramBuilder {
 public:
  OpBuilder(const std::string& op_name);

  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) override;

 private:
  std::string op_name_;
};

class PaddleModelBuilder final : public ProgramBuilder {
 public:
  PaddleModelBuilder(const std::string& model_path, const common::Target& target);

  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) override;

 private:
  std::string model_path_;
  common::Target target_;
};

}  // namespace tests
}  // namespace cinn
