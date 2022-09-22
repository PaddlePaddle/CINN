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

#include <algorithm>
#include <thread>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/llvm/codegen_x86.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/common/context.h"
#include "cinn/ir/module.h"

DECLARE_int32(cinn_parallel_compile_size);

namespace cinn {
namespace hlir {
namespace framework {

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::operator()() {
  // Task Spilt
  SplitTask();
  // merge instruction
  return MergeResult();
}

void ParallelCompiler::SplitTask() {
  CHECK_EQ(graph_->fusion_groups.size(), optition_.lowered_funcs.size());
  // split task
  int num_per_task = std::max((graph_->fusion_groups.size() - 1) / FLAGS_cinn_parallel_compile_size + 1, 16UL);

  for (int idx = 0; idx < graph_->fusion_groups.size(); idx += num_per_task) {
    int start          = idx;
    int end            = std::min(idx + num_per_task, static_cast<int>(graph_->fusion_groups.size()));
    auto groups        = std::vector<std::shared_ptr<Graph::Group>>(graph_->fusion_groups.begin() + start,
                                                             graph_->fusion_groups.begin() + end);
    auto lowered_funcs = optition_.lowered_funcs.size() > 0
                             ? std::vector<std::vector<ir::LoweredFunc>>(optition_.lowered_funcs.begin() + start,
                                                                         optition_.lowered_funcs.begin() + end)
                             : optition_.lowered_funcs;
    tasks_.emplace_back(scope_, graph_, groups, lowered_funcs, target_);
  }
}

void RunTask(ParallelCompiler::Task* task) {
  task->Lowering();
  task->CodegenAndJit();
  task->BuildInstruction();
}

void ParallelCompiler::LaunchTask() {
  // start sub-task.
  std::vector<std::thread> threads;
  for (int idx = 1; idx < tasks_.size(); ++idx) {
    threads.emplace_back(RunTask, &tasks_[idx]);
  }

  RunTask(&tasks_[0]);
  // syncthreads.
  for (auto& worker : threads) {
    worker.join();
  }
}

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::MergeResult() {
  std::vector<std::unique_ptr<Instruction>> res;
  for (auto& task : tasks_) {
    for (auto& instr : task.instructions) {
      res.push_back(std::move(instr));
    }
  }
  return std::move(res);
}

void ParallelCompiler::Task::Lowering() {
  if (!lowered_funcs.size()) {
    auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
    auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

    OpLowerer op_lowerer(dtype_dict, shape_dict, target);
    for (auto& group : groups) {
      VLOG(3) << "group_id is : " << group->group_id << ", and its number is : " << group->nodes.size();

      lowered_funcs.emplace_back(std::move(op_lowerer.Lower(group)));
      CHECK_EQ(lowered_funcs.back().size(), 1) << "Lowerd Function Is Not Equal 1!";
      VLOG(3) << lowered_funcs.back()[0];
    }
  }
}

void ParallelCompiler::Task::CodegenAndJit() {
  // build module
  ir::Module::Builder builder(common::UniqName("module"), target);
  for (auto& func : lowered_funcs) {
    CHECK_EQ(func.size(), 1);
    builder.AddFunction(func[0]);
  }
  auto ir_module = builder.Build();
  // codegen compile
  if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = backends::SplitCudaAndHostModule(ir_module);
    auto hmodule        = std::get<0>(splited_module);
    auto dmodule        = std::get<1>(splited_module);

    backends::CodeGenCUDA_Dev codegen(target);
    auto cuda_c = codegen.Compile(dmodule);

    using runtime::cuda::CUDAModule;
    backends::NVRTC_Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty());

    // load cumodule
    cumodule.reset(new CUDAModule(ptx, CUDAModule::Kind::PTX));
    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : dmodule.functions()) {
      auto cufunc = cumodule->GetFunction(0, fn->name);
      CHECK(cufunc);
      symbols.RegisterVar(fn->name + "_ptr_", reinterpret_cast<void*>(cufunc));
    }
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions(), std::move(symbols));
    engine->Link<backends::CodeGenCUDA_Host>(hmodule);
#endif
  } else {
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions());
    engine->Link<backends::CodeGenX86>(ir_module);
  }
}

void ParallelCompiler::Task::BuildInstruction() {
  // get func args from lowered func.
  auto get_func_args = [](ir::LoweredFunc func, ir::Argument::IO io) {
    std::vector<std::string> args;
    std::transform(func->args.begin(), func->args.end(), std::back_inserter(args), [io](ir::Argument arg) {
      if (io == ir::Argument::IO::kInput && arg.is_input()) {
        return arg.name();
      } else if (io == ir::Argument::IO::kOutput && arg.is_output()) {
        return arg.name();
      } else {
        return std::string("");
      }
    });
    args.erase(std::remove(args.begin(), args.end(), ""), args.end());
    return args;
  };
  // create instruction.
  for (auto& func : lowered_funcs) {
    CHECK_EQ(func.size(), 1);
    auto instr  = std::unique_ptr<Instruction>(new Instruction(target,
                                                              scope.get(),
                                                              get_func_args(func[0], ir::Argument::IO::kInput),
                                                              get_func_args(func[0], ir::Argument::IO::kOutput),
                                                              func[0]->name));
    auto fn_ptr = engine->Lookup(func[0]->name);
    CHECK(fn_ptr) << "Can't find jit function : " << func[0]->name;
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), func[0]->name);

    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
