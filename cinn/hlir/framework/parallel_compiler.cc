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

#include "cinn/hlir/framework/parallel_compiler.h"

#include <thread>

namespace cinn {
namespace hlir {
namespace framework {

std::vector<Instruction> ParallelCompiler::operator()() {
  // Task Spilt
  TaskSpilit();
  // merge instruction
  return MergeResult();
}

void ParallelCompiler::SplitTask() {}

void RunTask(ParallelCompiler::Task& task) {}

void ParallelCompiler::LaunchTask() {
  // start sub-task.
  std::vector<std::thread> threads;
  for (int idx = 1; idx < tasks_.size(); ++idx) {
    threads.emplace_back(RunTask, tasks_[idx]);
  }

  RunTask(task[0]);
  // syncthreads.
  for (auto task : threads) {
    task.join();
  }
}

std::vector<Instruction> ParallelCompiler::MergeResult() {
  std::vector<Instruction> res;
  for (auto& task : tasks_) {
    res.insert(res.end(), task.instructions.begin(), task.instructions.end());
  }
  return res;
}

void ParallelCompiler::Task::Lower() {
  if (lowered_funcs.size()) {
    return;
  }
  // do op lowering
  ir::Module::Builder builder;
  for (auto& func : lowered_funcs) {
    builder.AddFunction(func);
  }

  builder.Build();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
