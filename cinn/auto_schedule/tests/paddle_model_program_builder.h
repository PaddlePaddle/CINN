// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/auto_schedule/tests/program_case_builder.h"
#include "cinn/frontend/net_builder.h"

namespace cinn {
namespace auto_schedule {

class PaddleModelProgramBuilder : public ProgramCaseBuilder {
 public:
  PaddleModelProgramBuilder(const std::string& model_path,
                            const std::vector<std::string>& input_names,
                            const std::vector<std::vector<int>>& input_shapes)
      : model_path_(model_path), input_names_(input_names), input_shapes_(input_shapes) {}

  frontend::Program operator()() override {
    CHECK(!input_names_.empty());
    CHECK_EQ(input_names_.size(), input_shapes_.size());

    auto scope = std::make_shared<hlir::framework::Scope>();
    std::unordered_map<std::string, std::vector<int>> input_to_shape;
    for (int idx = 0; idx < input_names_.size(); ++idx) {
      input_to_shape[input_names_[idx]] = input_shapes_[idx];
    }
    auto loadedProgram = cinn::frontend::LoadPaddleProgram(model_path_, scope.get(), input_to_shape, true, target_);
    auto& program      = std::get<0>(loadedProgram);
    auto& varmap       = std::get<1>(loadedProgram);
    VLOG(3) << "loaded program: " << *program;
    CHECK(!varmap.empty());

    std::vector<frontend::Variable> input_vars;
    std::transform(input_names_.begin(), input_names_.end(), std::back_inserter(input_vars), [&](const std::string& x) {
      return varmap.at(x);
    });

    for (int i = 0; i < input_vars.size(); i++) {
      input_vars[i]->shape = input_shapes_[i];
    }

    program->SetInputs(input_vars);
    program->Validate();

    return *program;
  }

 private:
  const std::string model_path_;
  std::vector<std::string> input_names_;
  std::vector<std::vector<int>> input_shapes_;
#ifdef CINN_WITH_CUDA
  Target target_ = common::DefaultNVGPUTarget();
#else
  Target target_ = common::DefaultHostTarget();
#endif
};

}  // namespace auto_schedule
}  // namespace cinn
