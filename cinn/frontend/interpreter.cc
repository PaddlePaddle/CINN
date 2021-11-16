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

#include "cinn/frontend/interpreter.h"

#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

struct Interpreter::Impl {
  Impl(const std::vector<std::string>& input_names, const std::vector<hlir::framework::shape_t>& input_shapes)
      : scope_(std::make_shared<hlir::framework::Scope>()), input_names_(input_names), input_shapes_(input_shapes) {}

  /**
   * Build the model.
   * @param input_names The name of input variables.
   * @param input_shapes The input shapes.
   */
  void Build(const std::vector<std::string>& input_names,
             const std::vector<hlir::framework::shape_t>& input_shapes,
             const Target& target);

 private:
  friend class Interpreter;

  std::vector<std::string> input_names_;
  std::vector<hlir::framework::shape_t> input_shapes_;

  std::shared_ptr<hlir::framework::Scope> scope_;
  std::unique_ptr<frontend::Program> program_;
  std::unique_ptr<hlir::framework::GraphCompiler> graph_compiler_;

  absl::flat_hash_map<std::string, Variable> var_map_;
  absl::flat_hash_map<std::string, std::string> var_map_paddle_to_cinn_;
  absl::flat_hash_map<std::string, std::string> var_map_cinn_to_paddle_;

  std::unique_ptr<hlir::framework::Program> runtime_program_;
  std::unique_ptr<hlir::framework::Program> prerun_program_;
};

void Interpreter::LoadPaddleModel(const std::string& model_dir, const Target& target, bool params_combined) {
  auto programTuple               = LoadPaddleProgram(model_dir, impl_->scope_.get(), params_combined, target);
  auto& program                   = std::get<0>(programTuple);
  auto& var_map                   = std::get<1>(programTuple);
  auto& var_map_paddle_to_program = std::get<2>(programTuple);
  impl_->program_.reset(program.release());
  impl_->var_map_                = var_map;
  impl_->var_map_paddle_to_cinn_ = var_map_paddle_to_program;

  impl_->Build(impl_->input_names_, impl_->input_shapes_, target);
}

void Interpreter::Run() { impl_->runtime_program_->Execute(); }

hlir::framework::Tensor Interpreter::GetTensor(const std::string& name) {
  if (impl_->scope_->FindVar(name)) return impl_->scope_->GetTensor(name);

  auto it = impl_->var_map_paddle_to_cinn_.find(name);
  if (it == impl_->var_map_paddle_to_cinn_.end()) {
    LOG(FATAL) << "No variable called [" << name
               << "] found in executor\nThe existing vars: " << utils::Join(impl_->scope_->var_names(), ", ");
  }
  return impl_->scope_->GetTensor(it->second);
}

void Interpreter::Impl::Build(const std::vector<std::string>& input_names,
                              const std::vector<hlir::framework::shape_t>& input_shapes,
                              const Target& target) {
  CHECK(!input_names.empty());
  CHECK(!var_map_.empty());
  CHECK_EQ(input_names.size(), input_shapes.size());

  std::vector<Variable> input_vars;
  std::transform(input_names.begin(), input_names.end(), std::back_inserter(input_vars), [&](const std::string& x) {
    return var_map_.at(x);
  });

  for (int i = 0; i < input_vars.size(); i++) input_vars[i]->shape = input_shapes[i];

  program_->SetInputs({input_vars});
  program_->Validate();

  VLOG(1) << "Program:\n" << *program_;

  auto graph = std::make_shared<hlir::framework::Graph>(*program_, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
#ifndef CINN_WITH_CUDA
  if (target.arch == Target::Arch::X86) {
    hlir::framework::ApplyPass(graph.get(), "AlterLayout");
  }
#endif
  hlir::framework::ApplyPass(graph.get(), "ConstPropagate");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  // Target target = common::DefaultHostTarget();
  scope_ = hlir::framework::BuildScope(target, graph, scope_);
  graph_compiler_.reset(new hlir::framework::GraphCompiler(target, scope_, graph));
  runtime_program_ = graph_compiler_->Build();
  runtime_program_->PreRun();
}

std::shared_ptr<hlir::framework::Scope> Interpreter::scope() {
  CHECK(impl_->scope_);
  return impl_->scope_;
}

Interpreter::Interpreter(const std::vector<std::string>& input_names,
                         const std::vector<hlir::framework::shape_t>& input_shapes)
    : impl_(new Impl(input_names, input_shapes)) {}

}  // namespace cinn::frontend

cinn::frontend::Interpreter::~Interpreter() {}
