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

#include <random>

#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

static Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

static void SetRandData(hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

class PassTest {
 public:
  PassTest() { target_ = GetTarget(); }

  int ApplyProgramPass(NetBuilder& builder,
                       const std::vector<std::string>& program_passes,
                       const std::vector<std::string>& output_names) {
    program_        = builder.Build();
    int before_size = program_.size();
    LOG(INFO) << program_;
    CHECK(IsValid()) << "The origin program is not valid.";

    std::unordered_set<std::string> fetch_var_ids(output_names.begin(), output_names.end());
    ProgramPass::Apply(&program_, fetch_var_ids, target_, program_passes);
    int after_size = program_.size();
    LOG(INFO) << program_;
    CHECK(IsValid()) << "The transformed program is not valid.";

    return before_size - after_size;
  }

  void Execute(const std::vector<std::string>& input_names, const std::vector<std::string>& output_names) {
    auto graph = std::make_shared<hlir::framework::Graph>(program_, target_);
    hlir::framework::ApplyPass(graph.get(), "OpFusion");

    scope_ = hlir::framework::BuildScope(target_, graph);
    hlir::framework::GraphCompiler gc(target_, scope_, graph);

    hlir::framework::GraphCompiler::CompileOptions options;
    options.with_instantiate_variables = true;
    std::unordered_set<std::string> fetch_var_ids(output_names.begin(), output_names.end());
    auto result          = gc.Build(options, std::move(fetch_var_ids));
    auto runtime_program = std::move(result.runtime_program);

    for (auto& name : input_names) {
      SetInputTensor(name);
    }

    runtime_program->Execute();
  }

  void SetInputTensor(const std::string& name) {
    scope_->Var<hlir::framework::Tensor>(name);
    auto tensor = scope_->GetTensor(name);
    SetRandData(tensor, target_);
  }

 protected:
  bool IsValid() {
    std::unordered_set<std::string> inputs;
    for (auto& var : program_.GetInputs()) {
      inputs.insert(var->id);
    }

    std::unordered_set<std::string> outputs;
    for (int i = 0; i < program_.size(); ++i) {
      const auto& instr = program_[i];
      for (auto& var : instr->outputs) {
        outputs.insert(var->id);
      }
    }

    bool valid = true;
    for (int i = 0; i < program_.size(); ++i) {
      const auto& instr = program_[i];
      // The inputs should be feeded, or other instructions' output.
      for (auto& var : instr->inputs) {
        if (!inputs.count(var->id) && !outputs.count(var->id)) {
          LOG(INFO) << "The input " << var->id << " of " << i << "-th instrution (" << instr
                    << ") is not the output of any other instructions.";
          valid = false;
        }
      }
    }

    return valid;
  }

  Target target_;
  Program program_;
  std::shared_ptr<hlir::framework::Scope> scope_;
};

}  // namespace cinn::frontend
