#include "cinn/backends/llvm/simple_jit.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

void SimpleJIT::Link(lang::Module module, bool optimize) {
  std::string runtime_ir(backends::kRuntimeLlvmIr);
  llvm::SMDiagnostic error;
  auto m = llvm::parseAssemblyString(runtime_ir, error, context());
  auto b = std::make_unique<llvm::IRBuilder<>>(context());

  auto ir_emitter = std::make_unique<CodeGenLLVM>(m.get(), b.get());
  for (auto &buffer : module->buffers) {
    auto expr = runtime::BufferCreate(buffer.as_buffer_ref());
    ir_emitter->Visit(&expr);
  }

  for (auto &fn : module.functions()) {
    LOG(WARNING) << "JIT Linking function [" << fn->name << "]";
    ir::Expr fn_expr(fn);

    auto *fn_ = ir_emitter->Visit(&fn_expr);
    VLOG(1) << DumpToString(*fn_);
  }

  AddModule(std::move(m), optimize);
}

void SimpleJIT::AddModule(std::unique_ptr<llvm::Module> module, bool optimize) {
  CHECK(!llvm::verifyModule(*module, &llvm::errs())) << "Transformation resulted in an invalid module";

  bool debug = false;
  if (optimize) {
    llvm::PassBuilder pass_builder;
    llvm::LoopAnalysisManager loop_analysis_manager(debug);
    llvm::FunctionAnalysisManager function_analysis_manager(debug);
    llvm::CGSCCAnalysisManager cgscc_analysis_manager(debug);
    llvm::ModuleAnalysisManager module_analysis_manager(debug);

    pass_builder.registerModuleAnalyses(module_analysis_manager);
    pass_builder.registerCGSCCAnalyses(cgscc_analysis_manager);
    pass_builder.registerFunctionAnalyses(function_analysis_manager);
    pass_builder.registerLoopAnalyses(loop_analysis_manager);
    pass_builder.crossRegisterProxies(
        loop_analysis_manager, function_analysis_manager, cgscc_analysis_manager, module_analysis_manager);

    llvm::ModulePassManager module_pass_manager =
        pass_builder.buildPerModuleDefaultPipeline(llvm::PassBuilder::OptimizationLevel::O3);
    module_pass_manager.run(*module, module_analysis_manager);
  }

  llvm::orc::ThreadSafeModule tsm(std::move(module), context_);
  llvm::cantFail(jit_->addIRModule(std::move(tsm)));

  if (debug) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    jit_->getExecutionSession().dump(os);
    os.flush();
    LOG(INFO) << "compiled jit:\n" << buffer;
  }
}

}  // namespace backends
}  // namespace cinn
