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

void ParallelCompiler::SplitTask() {
  //
}

void RunTask(ParallelCompiler::Task& task) {
  task.Lowering();
  task.CodegenAndJit();
  task.BuildInstruction();
}

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

void ParallelCompiler::Task::Lowering() {
  if (!lowered_funcs.size()) {
  }
  // do op lowering
  ir::Module::Builder builder;
  for (auto& func : lowered_funcs) {
    builder.AddFunction(func);
  }

  ir_module = builder.Build();
}

void ParallelCompiler::Task::CodegenAndJit() {
  if (target_ == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = SplitCudaAndHostModule(module);
    ir_module           = std::get<0>(splited_module);
    auto dmodule        = td::get<1>(splited_module);

    CodeGenCUDA_Dev codegen(target_);
    auto cuda_c = codegen.Compile(dmodule);

    using runtime::cuda::CUDAModule;
    backends::NVRTC_Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty());

    // load cumodule
    cumodule.reset(new CUDAModule(ptx, CUDAModule::Kind::PTX));
    // register kernel
    RuntimeSymbols symbols;
    for (auto& fn : dmodule.functions()) {
      auto cufunc = cumodule->GetFunction(0, fn->name);
      CHECK(cufunc);
      symbols.RegisterVar(fn->name + "_ptr_", reinterpret_cast<void*>(cufunc));
    }
    engine_ = ExecutionEngine::Create(ExecutionOptions(), std::move(symbols));
    engine_->Link<CodeGenCUDA_Host>(ir_module);
#endif
  } else {
    engine_ = ExecutionEngine::Create(ExecutionOptions());
    engine_->Link<CodeGenX86>(module);
  }
}

void ParallelCompiler::Task::BuildInstruction() {
  for (auto& group : groups) {
    auto instr  = std::unique_ptr<Instruction>(new Instruction(
        target_, scope_.get(), fusion_group->input_names, fusion_group->output_names, fusion_group->GetFuncName()));
    auto fn_ptr = engine_->Lookup(fusion_group->GetFuncName());
    CHECK(fn_ptr) << "Can't find jit function : " << fusion_group->GetFuncName();
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), fusion_group->GetFuncName());

    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
