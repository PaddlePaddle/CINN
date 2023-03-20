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

#include "tests/utils.h"

#include <glog/logging.h>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/compiler.h"
#include "cinn/frontend/optimize.h"
#include "cinn/hlir/framework/op_lowering.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace tests {

using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;

std::shared_ptr<hlir::framework::Graph> OptimizeByPass(frontend::Program& program, const common::Target& target) {
  return frontend::Optimize(&program, {}, target);
}

std::vector<ir::LoweredFunc> LowerFusionGroup(std::shared_ptr<hlir::framework::Graph> graph,
                                              std::shared_ptr<hlir::framework::Graph::Group> group,
                                              const common::Target& target,
                                              bool apply_manual_schedule) {
  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  hlir::framework::OpLowerer op_lowerer(dtype_dict, shape_dict, target);

  return apply_manual_schedule ? op_lowerer.Lower(group) : op_lowerer.LowerWithoutSchedule(group);
}

ir::IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs) {
  std::vector<Expr> bodys;
  for (auto&& func : lowered_funcs) {
    bodys.emplace_back(func->body);
  }
  return ir::IRSchedule(ir::ModuleExpr({std::move(bodys)}));
}

std::vector<ir::LoweredFunc> OptimizeBySchedule(const ir::IRSchedule& schedule,
                                                const std::vector<ir::LoweredFunc>& original_funcs,
                                                const common::Target& target) {
  auto&& updated_bodys = schedule.GetModule().GetExprs();
  CHECK_EQ(updated_bodys.size(), original_funcs.size()) << "associated exprs size not equal";

  std::vector<ir::LoweredFunc> results;
  for (int i = 0; i < original_funcs.size(); ++i) {
    ir::Expr func_body              = updated_bodys.at(i);
    const ir::LoweredFunc& ori_func = original_funcs.at(i);
    auto&& new_func                 = auto_schedule::UpdateFuncWithNewBody(target, ori_func, func_body);
    results.emplace_back(new_func);
  }

  return results;
}

ir::Module BuildIRModule(const std::vector<ir::LoweredFunc>& lowered_funcs, const common::Target& target) {
  ir::Module::Builder builder("test_bulder", target);
  for (auto&& func : lowered_funcs) {
    builder.AddFunction(func);
  }
  return builder.Build();
}

std::string GenSourceCode(const ir::Module& ir_module, const common::Target& target) {
  std::unique_ptr<backends::CodeGenC> codegen;
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    codegen = std::make_unique<backends::CodeGenCUDA_Dev>(target);
  } else {
    codegen = std::make_unique<backends::CodeGenCX86>(target, CodeGenCX86::Feature::AVX512);
  }
#else
  codegen = std::make_unique<backends::CodeGenCX86>(target, CodeGenCX86::Feature::AVX512);
#endif
  codegen->SetInlineBuiltinCodes(false);
  return codegen->Compile(ir_module, CodeGenC::OutputKind::CImpl);
}

std::vector<hlir::framework::Instruction> BuildExecution(const ir::Module& ir_module,
                                                         const common::Target& target,
                                                         hlir::framework::Scope* scope) {
  auto backend_compier = backends::Compiler::Create(target);
  backend_compier->Build(ir_module);

  std::vector<hlir::framework::Instruction> results;
  for (auto&& func : ir_module.functions()) {
    std::vector<std::string> input_args;
    std::vector<std::string> output_args;

    for (auto&& arg : func->args) {
      if (arg.is_input())
        input_args.emplace_back(arg.name());
      else
        output_args.emplace_back(arg.name());
    }
    results.emplace_back(Instruction(target, scope, input_args, output_args, func->name));

    auto func_ptr = reinterpret_cast<void (*)(void**, int32_t)>(backend_compier->Lookup(func->name));
    results.back().SetLoweredFunc(reinterpret_cast<void*>(func_ptr));
    results.back().Finalize();
  }
  return results;
}

}  // namespace tests
}  // namespace cinn
