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

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_lowering.h"

namespace cinn {
namespace hlir {
namespace framework {

class ParallelCompiler {
 public:
  struct CompileOptions {
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  };

 public:
  ParallelCompile(std::shared_ptr<Graph>& graph, CompileOptions& option, const common::Target& target);
  ~ParallelCompile();
  std::vector<Instruction> operator()();

 private:
  void SplitTask();
  void LaunchTask();
  std::vector<Instruction> MergeResult();
  struct Task {
   public:
    Task(std::vector<std::shared_ptr<Group>>& g, std::vector<std::vector<ir::LoweredFunc>> f)
        : groups(g), lowered_funcs(f) {}
    void Lowering();
    void CodegenAndJit();
    void BuildInstruction();

   public:
    ir::Module ir_module;
    std::vector<std::shared_ptr<Group>> groups;
    std::vector<std::unique_ptr<Instruction>> instructions;
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;

   public:
    std::unique_ptr<ExecutionEngine> engine;
#ifdef CINN_WITH_CUDA
    std::unique_ptr<runtime::cuda::CUDAModule> cumodule;
#endif
  };
  std::vector<Task> tasks_;

 private:
  Target target_;
  CompileOptions optition_;
  std::shared_ptr<Graph> graph_;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
