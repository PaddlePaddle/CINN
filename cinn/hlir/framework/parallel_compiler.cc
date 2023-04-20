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
#include <fstream>
#include <thread>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/compiler.h"
#include "cinn/backends/llvm/codegen_x86.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/backends/nvrtc/nvrtc_util.h"
#include "cinn/common/context.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/ir/module.h"

DECLARE_int32(cinn_parallel_compile_size);

namespace cinn {
namespace hlir {
namespace framework {
static constexpr int DebugLogMaxLen = 30000;

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::operator()() {
  if (!FLAGS_cinn_parallel_compile_size) {
    return std::vector<std::unique_ptr<Instruction>>();
  }
  if (graph_->fusion_groups.size() == 0) {
    hlir::framework::ApplyPasses(graph_.get(), {"BuildNonFusedGroupsPass"});
  }
  // Task Spilt
  SplitTask();
  // launch task
  LaunchTask();
  // merge instruction
  return MergeResult();
}

OpPatternKind GetOpKind(const framework::Node* node) {
  auto& op_pattern_dict = framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
  CHECK(op_pattern_dict.Find(node->op())) << "Don't find the pattern of op : " << node->id();
  auto kind = op_pattern_dict[node->op()];

  if (kind == framework::kBroadcast) {
    // As binary op was defined as broadcast, actually it should be element-wise.
    if (node->op()->name != "broadcast_to") {
      return framework::kElementWise;
    }
  }

  return kind;
}

void ParallelCompiler::SplitTask() {
  CHECK(graph_->fusion_groups.size());
  CHECK(graph_->fusion_groups.size() == option_.lowered_funcs.size() || option_.lowered_funcs.size() == 0);
  // split task
  size_t flag_parallel_num = graph_->fusion_groups.size();
  if (FLAGS_cinn_parallel_compile_size > 0) {
    flag_parallel_num = FLAGS_cinn_parallel_compile_size;
  }
  int task_num = std::min(graph_->fusion_groups.size(), flag_parallel_num);

  for (int idx = 0; idx < task_num; ++idx) {
    tasks_.emplace_back(std::make_unique<Task>(this, scope_, graph_, option_, target_));
  }
  VLOG(2) << "Split task to " << tasks_.size() << " sub-task!";
}

void RunTask(ParallelCompiler::Task* task) {
  VLOG(2) << "Stark run sub-task, Thread Id : " << std::this_thread::get_id();
  task->Run();
  VLOG(2) << "Finish run sub-task, Thread Id : " << std::this_thread::get_id();
}

void ParallelCompiler::LaunchTask() {
  // start sub-task.
  std::vector<std::thread> threads;
  for (int idx = 1; idx < tasks_.size(); ++idx) {
    threads.emplace_back(RunTask, tasks_[idx].get());
  }

  RunTask(tasks_[0].get());
  // syncthreads.
  for (auto& worker : threads) {
    worker.join();
  }
}

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::MergeResult() {
  std::vector<std::unique_ptr<Instruction>> res(graph_->fusion_groups.size());
  for (auto& task : tasks_) {
    for (int idx = 0; idx < task->gidx.size(); ++idx) {
      res[task->gidx[idx]] = std::move(task->instructions[idx]);
    }
  }
  return std::move(res);
}

void ParallelCompiler::Task::Run() {
  if (!options.lowered_funcs.empty()) {
    CHECK_EQ(options.lowered_funcs.size(), graph->fusion_groups.size());
  }

  while (true) {
    int idx = compiler->GetGroupIdx();
    if (idx < 0) {
      break;
    }

    auto& group = graph->fusion_groups[idx];

    VLOG(1) << "Start Lowering Group " << idx << " :\n"
            << "Group " << idx << " {\n"
            << graph->DebugGroupedGraph(group->CollectNodes()) << "}\n";
    auto func = Lowering(group, idx);

    VLOG(2) << "Start Codegen and JIT of Group " << idx;
    auto engine = CodegenAndJit(func, idx);

    VLOG(2) << "Start BuildInstruction of Group " << idx;
    auto instr = BuildInstruction(group, std::move(engine));

    gidx.emplace_back(idx);
    instructions.emplace_back(std::move(instr));
  }
}

std::vector<ir::LoweredFunc> ParallelCompiler::Task::Lowering(std::shared_ptr<Graph::Group>& group, int idx) {
  std::vector<ir::LoweredFunc> func;
  if (options.lowered_funcs.size() > idx) {
    func = options.lowered_funcs[idx];
  } else {
    auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
    auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

    OpLowerer op_lowerer(dtype_dict, shape_dict, target);

    func = op_lowerer.Lower(group);
  }
  CHECK_EQ(func.size(), 1) << "Lowerd Function Is Not Equal 1!";
  return func;
}

std::unique_ptr<backends::ExecutionEngine> ParallelCompiler::Task::CodegenAndJit(
    const std::vector<ir::LoweredFunc>& func, int idx) {
  // build module
  ir::Module::Builder builder(common::UniqName("module"), target);
  builder.AddFunction(func[0]);
  auto ir_module = builder.Build();

  std::unique_ptr<backends::ExecutionEngine> engine;

  // codegen compile
  if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = backends::SplitCudaAndHostModule(ir_module);
    auto hmodule        = std::get<0>(splited_module);
    auto dmodule        = std::get<1>(splited_module);

    VLOG(3) << "Host Code:\n" << hmodule;
    VLOG(3) << "Device Code:\n" << dmodule;
    backends::CodeGenCUDA_Dev codegen(target);
    auto cuda_c = codegen.Compile(dmodule);
    CHECK(!cuda_c.empty()) << "Compile CUDA C code failed from device module:\n" << dmodule;

    cinn::backends::SourceCodePrint::GetInstance()->write(cuda_c);
    graph->SaveSourceCode(idx, cuda_c);

    using runtime::cuda::CUDAModule;
    backends::nvrtc::Compiler nvrtc;
    auto ptx = nvrtc(cuda_c);
    CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n" << cuda_c;
    graph->SavePTXCode(idx, ptx);

    // load cumodule
    runtime::cuda::CUDAModule cumodule(ptx, nvrtc.compile_to_cubin() ? CUDAModule::Kind::CUBIN : CUDAModule::Kind::PTX);
    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : dmodule.functions()) {
      auto cufunc = cumodule.GetFunction(0, fn->name);
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
  return engine;
}

std::unique_ptr<Instruction> ParallelCompiler::Task::BuildInstruction(
    std::shared_ptr<Graph::Group>& group, std::unique_ptr<backends::ExecutionEngine>&& engine) {
  // create instruction.
  CHECK(group->input_names.size() > 0 || group->output_names.size() > 0);
  auto instr =
      std::make_unique<Instruction>(target, scope.get(), group->input_names, group->output_names, group->GetFuncName());

  auto fn_ptr = engine->Lookup(group->GetFuncName());
  CHECK(fn_ptr) << "Can't find jit function : " << group->GetFuncName();
  instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), group->GetFuncName());

  instr->Finalize();
  return instr;
}

int ParallelCompiler::GetGroupIdx() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (index < graph_->fusion_groups.size()) {
    return index++;
  } else {
    return -1;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
