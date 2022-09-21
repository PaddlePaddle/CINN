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

#include <vector>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/ir/lowered_func.h"
#ifdef CINN_WITH_CUDA
#include "cinn/runtime/cuda/cuda_module.h"
#endif
namespace cinn {
namespace hlir {
namespace framework {

class ParallelCompiler {
 public:
  struct CompileOptions {
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  };

 public:
  explicit ParallelCompiler(std::shared_ptr<Scope>& scope,
                            std::shared_ptr<Graph>& graph,
                            CompileOptions& option,
                            const common::Target& target)
      : scope_(scope), graph_(graph), optition_(option), target_(target) {}
  ~ParallelCompiler() {}
  std::vector<std::unique_ptr<Instruction>> operator()();

 private:
  void SplitTask();
  void LaunchTask();
  std::vector<std::unique_ptr<Instruction>> MergeResult();

 public:
  struct Task {
   public:
    Task(std::shared_ptr<Scope>& s,
         std::shared_ptr<Graph>& g,
         std::vector<std::shared_ptr<Graph::Group>> gg,
         std::vector<std::vector<ir::LoweredFunc>>& f,
         Target t)
        : scope(s), graph(g), groups(gg), lowered_funcs(f), target(t) {}
    void Lowering();
    void CodegenAndJit();
    void BuildInstruction();

   public:
    Target target;
    std::shared_ptr<Scope> scope;
    std::shared_ptr<Graph> graph;
    std::vector<std::shared_ptr<Graph::Group>> groups;
    std::vector<std::unique_ptr<Instruction>> instructions;
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;

   public:
    std::unique_ptr<backends::ExecutionEngine> engine;
#ifdef CINN_WITH_CUDA
    std::unique_ptr<runtime::cuda::CUDAModule> cumodule;
#endif
  };
  std::vector<Task> tasks_;

 private:
  common::Target target_;
  CompileOptions optition_;
  std::shared_ptr<Scope> scope_;
  std::shared_ptr<Graph> graph_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
