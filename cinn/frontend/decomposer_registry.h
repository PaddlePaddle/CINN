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

#include <functional>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

class Decomposer;

class DecomposerContext {
 public:
  explicit DecomposerContext(Program* prog) : program(prog) {}

  Program* program{nullptr};
};

class InstrDecomposerRegistry : public Registry<Decomposer> {
 public:
  static InstrDecomposerRegistry* Global() {
    static InstrDecomposerRegistry x;
    return &x;
  }

  inline const Decomposer* Get(const std::string& op_name, const common::Target& target) {
    const Decomposer* decomposer = Find(op_name, target);
    CHECK(decomposer) << "Decomposer for [" << op_name << ", " << target << "] is not registered";
    return decomposer;
  }

  inline const Decomposer* Find(const std::string& name, const common::Target& target) {
    return Registry<Decomposer>::Find(name + "_" + target.arch_str());
  }

  inline Decomposer& __REGISTER__(const std::string& name, const common::Target& target) {
    return Registry<Decomposer>::__REGISTER__(name + "_" + target.arch_str());
  }

 private:
  InstrDecomposerRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(InstrDecomposerRegistry);
};

class Decomposer {
 public:
  using DecomposerKernel = std::function<void(const Instruction& instr, const DecomposerContext&)>;

  Decomposer& set_body(const DecomposerKernel& kernel) {
    kernel_ = kernel;
    return *this;
  }

  void Run(const Instruction& instr, const DecomposerContext& context) { kernel_(instr, context); }

  std::string name;

 private:
  DecomposerKernel kernel_;
};

#define CINN_DECOMPOSER_REGISTER(name, target)                                                \
  static ::cinn::frontend::Decomposer& CINN_STR_CONCAT(__make_Decomposer_name, __COUNTER__) = \
      ::cinn::frontend::InstrDecomposerRegistry::Global()->__REGISTER__(#name, target)

}  // namespace frontend
}  // namespace cinn
