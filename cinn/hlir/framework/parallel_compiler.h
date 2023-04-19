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

#include <mutex>
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
                            const CompileOptions& option,
                            const common::Target& target)
      : scope_(scope), graph_(graph), option_(option), target_(target) {}
  ~ParallelCompiler() {}
  std::vector<std::unique_ptr<Instruction>> operator()();

 private:
  void SplitTask();
  void LaunchTask();
  std::vector<std::unique_ptr<Instruction>> MergeResult();

 public:
  struct Task {
   public:
    Task(ParallelCompiler* p,
         std::shared_ptr<Scope>& s,
         std::shared_ptr<Graph>& g,
         const CompileOptions& cp,
         const Target& t)
        : compiler(p),
          scope(s),
          graph(g),
          options(cp),
          target(t),
          task_mtx_(std::make_unique<std::mutex>()),
          instructions(g->fusion_groups.size()) {}
    void Run();
    std::vector<ir::LoweredFunc> Lowering(OpLowerer* lower, std::shared_ptr<Graph::Group>& group, int idx);
    void CodegenAndJit(const std::vector<ir::LoweredFunc>& func, int idx);
    std::unique_ptr<Instruction> BuildInstruction(std::shared_ptr<Graph::Group>& group);

   public:
    const Target target;
    ParallelCompiler* compiler;
    std::shared_ptr<Scope> scope;
    std::shared_ptr<Graph> graph;
    const CompileOptions& options;

    std::unique_ptr<std::mutex> task_mtx_;
    std::vector<int> gidx;
    std::vector<std::unique_ptr<Instruction>> instructions;

   public:
    std::unique_ptr<backends::ExecutionEngine> engine;
#ifdef CINN_WITH_CUDA
    std::unique_ptr<runtime::cuda::CUDAModule> cumodule;
#endif
  };
  std::vector<Task> tasks_;
  int GetGroupIdx();

 private:
  int index{0};
  std::mutex mtx_;

  const common::Target target_;
  const CompileOptions& option_;
  std::shared_ptr<Scope> scope_;
  std::shared_ptr<Graph> graph_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
