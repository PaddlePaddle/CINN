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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/compiler.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * The Program is the runtime instance for running a computation.
 */
class Program {
 public:
  /**
   * Constructor.
   * @param scope The scope containing all the runtime variables.
   * @param instrs The instructions belonging to this program.
   */
  Program(const std::shared_ptr<Scope>& scope, std::vector<std::unique_ptr<Instruction>>&& instrs) : scope_(scope) {
    for (auto& ins : instrs) {
      if (ins->pre_run) {
        prerun_instrs_.push_back(std::move(ins));
      } else {
        instrs_.push_back(std::move(ins));
      }
    }
  }

  void PreRun(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr) {
    for (auto& ins : prerun_instrs_) {
      ins->Run(name2podargs);
    }
  }
  void Export(const std::vector<std::string>& persistent_vars, const std::string& filename);
  /**
   * Execute the program -- that is running all the instructions inside it.
   */
  void Execute(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr) {
    for (auto& ins : instrs_) {
      ins->Run(name2podargs);
    }
#ifdef CINN_WITH_CUDA
    if (instrs_[0]->target_.arch == Target::Arch::NVGPU) {
      CUDA_CALL(cudaDeviceSynchronize());
    }
#endif
  }

  void ExecuteTest(int repeat_) {
    cinn::utils::Timer timer1;
    for (int i = 0; i < 100; i++) {
      for (auto& ins : instrs_) {
        ins->Run();
      }
    }
    timer1.Start();
    for (int i = 0; i < repeat_; i++) {
      for (auto& ins : instrs_) {
        ins->Run();
      }
    }
#ifdef CINN_WITH_CUDA
    if (instrs_[0]->target_.arch == Target::Arch::NVGPU) {
      CUDA_CALL(cudaDeviceSynchronize());
    }
#endif
    double test_op_time = timer1.Stop() / repeat_;
    LOG(INFO) << "Repeat times: [" << repeat_ << "], average op time: [" << test_op_time << "] ms";
  }
  /**
   * Get the number of instructions.
   */
  size_t size() const { return instrs_.size(); }

 private:
  // We need to hold scope to assure tensors alive used in instructions.
  std::shared_ptr<Scope> scope_;
  // prerun instructions
  std::vector<std::unique_ptr<Instruction>> prerun_instrs_;
  // only runtime instructions
  std::vector<std::unique_ptr<Instruction>> instrs_;
};

/**
 * GraphCompiler compiles a graph and generate the runtime Program.
 */
class GraphCompiler final {
 public:
  GraphCompiler(Target target, const std::shared_ptr<Scope>& scope, const std::shared_ptr<Graph>& graph)
      : target_(std::move(target)), scope_(scope), graph_(graph), m_builder_(UniqName("module"), target) {}

  struct CompilationResult {
    std::unique_ptr<Program> runtime_program;
  };

  struct CompileOptions {
    std::string attached_code       = "";
    bool with_instantiate_variables = false;
  };

  // Compile with a packing option and result, to be extended easily.
  CompilationResult Build(const CompileOptions& options);
  void ExportObject(const std::string& path) { compiler_->ExportObject(path); }

  std::unique_ptr<Program> Build(const std::string& code = "");

  std::string GenSourceCode();

  void PrintFunc();

  const std::shared_ptr<Scope>& GetScope() const { return scope_; }

 private:
  std::vector<ir::LoweredFunc> GetOpFunc(const std::vector<Node*>& nodes);

  std::vector<ir::LoweredFunc> GetOpFunc(const Node* node);

  std::string GenOpFuncName(const Node* node) const { return "fn_" + node->id(); }

  // TODO(haozech) add implementation
  std::vector<std::string> OpGetInputNames(const Node* node) const;
  // TODO(haozech) add implementation
  std::vector<std::string> OpGetOutputNames(const Node* node) const;

  std::vector<std::unique_ptr<Instruction>> BuildInstructions();

 private:
  void ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_func);
  Target target_;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Scope> scope_;
  // mapping a function's name to its input artuments' names
  std::map<std::string, std::vector<std::string>> function2input_args_;
  // mapping a function's name to its output artuments' names
  std::map<std::string, std::vector<std::string>> function2output_args_;

  std::unique_ptr<backends::Compiler> compiler_;

  ir::Module::Builder m_builder_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);
};

std::shared_ptr<Scope> BuildScope(Target target,
                                  const std::shared_ptr<Graph>& graph,
                                  std::shared_ptr<Scope> scope = nullptr);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
