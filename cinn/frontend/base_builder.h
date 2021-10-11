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

#include <string>
#include <vector>

#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace frontend {

class BaseBuilder {
 public:
  explicit BaseBuilder(const std::string& name);

  Program Build();

  Placeholder CreateInput(const common::Type& type, const std::vector<int>& shape, const std::string& id_hint = "");

  // name of this builder
  const std::string& name() { return name_; }

  virtual ~BaseBuilder() {}

 protected:
  void AppendInstruction(const Instruction& instr) { instrs_.push_back(instr); }

  void InferShape(Instruction instr) const;

  std::string name_;
  std::vector<Instruction> instrs_;
  std::vector<Variable> inputs_;
};

}  // namespace frontend
}  // namespace cinn
