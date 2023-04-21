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
  explicit ParallelCompiler(const std::shared_ptr<Scope>& scope,
                            const std::shared_ptr<Graph>& graph,
                            const CompileOptions& option,
                            const common::Target& target)
      : scope_(scope), graph_(graph), option_(option), target_(target), index(0) {}
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
         const std::shared_ptr<Scope>& s,
         const std::shared_ptr<Graph>& g,
         const CompileOptions& cp,
         const Target& t)
        : compiler(p), scope(s), graph(g), options(cp), target(t) {}
    void Run();
    ir::LoweredFunc Lowering(std::shared_ptr<Graph::Group>& group, int idx);
    std::unique_ptr<backends::ExecutionEngine> CodegenAndJit(const std::vector<ir::LoweredFunc>& func, int idx);
    std::unique_ptr<Instruction> BuildInstruction(const std::shared_ptr<Graph::Group>& group,
                                                  backends::ExecutionEngine* engine);

   public:
    const Target& target;
    ParallelCompiler* compiler;
    const std::shared_ptr<Scope>& scope;
    const std::shared_ptr<Graph>& graph;
    const CompileOptions& options;

    std::vector<int> gidx;
    std::vector<std::unique_ptr<Instruction>> instructions;

   public:
    // restruct ExecutionEngine to avoid useless construct and deconstruct
    std::vector<std::unique_ptr<backends::ExecutionEngine>> engines;
#ifdef CINN_WITH_CUDA
    // restruct CUDAModule to avoid useless construct and deconstruct
    std::vector<std::unique_ptr<runtime::cuda::CUDAModule>> cumodules;
#endif
  };
  std::vector<std::unique_ptr<Task>> tasks_;
  int GetGroupIdx();

 private:
  std::atomic_int32_t index{0};

  const common::Target& target_;
  const CompileOptions& option_;
  const std::shared_ptr<Scope>& scope_;
  const std::shared_ptr<Graph>& graph_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
