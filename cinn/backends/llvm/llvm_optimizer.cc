#include "cinn/backends/llvm/llvm_optimizer.h"

#include <glog/logging.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace cinn::backends {

namespace {
template <typename PassManagerT>
class CustomPassManager : public PassManagerT {
 public:
  template <typename... Ts>
  explicit CustomPassManager(bool print_passes, Ts &&... ts) : PassManagerT(std::forward<Ts>(ts)...) {}

  void add(llvm::Pass *pass) override {
    if (print_passes_) {
      if constexpr (std::is_same_v<llvm::legacy::FunctionPassManager, PassManagerT>) {
        VLOG(1) << "llvm run function pass[" << std::string(pass->getPassName()) << "]";
      }

      if constexpr (std::is_same_v<llvm::legacy::PassManager, PassManagerT>) {
        VLOG(1) << "llvm run module pass[" << std::string(pass->getPassName()) << "]";
      }
    }
    PassManagerT::add(pass);
  }

 private:
  bool print_passes_;
};

using CustomFunctionPassManager = CustomPassManager<llvm::legacy::FunctionPassManager>;
using CustomModulePassManager   = CustomPassManager<llvm::legacy::PassManager>;
}  // namespace

LLVMModuleOptimizer::LLVMModuleOptimizer(llvm::TargetMachine *machine,
                                         int opt_level,
                                         llvm::FastMathFlags fast_math_flags,
                                         bool print_passes)
    : opt_level_(opt_level), print_passes_(print_passes) {}

void LLVMModuleOptimizer::operator()(llvm::Module *m) {
  auto fpm = std::make_unique<CustomFunctionPassManager>(print_passes_, m);
  fpm->add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
  fpm->add(llvm::createInstructionCombiningPass());
  fpm->add(llvm::createReassociatePass());
  fpm->add(llvm::createGVNPass());
  fpm->add(llvm::createCFGSimplificationPass());

  auto mpm = std::make_unique<CustomModulePassManager>(print_passes_);
  mpm->add(llvm::createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
  auto builder           = std::make_unique<llvm::PassManagerBuilder>();
  builder->OptLevel      = 3;
  builder->Inliner       = llvm::createFunctionInliningPass();
  builder->LoopVectorize = true;
  builder->SLPVectorize  = true;
  builder->populateFunctionPassManager(*fpm);
  builder->populateModulePassManager(*mpm);

  fpm->doInitialization();
  for (auto &fn : *m) {
    fpm->run(fn);
  }
  fpm->doFinalization();
  // mpm->run(*m);
}

}  // namespace cinn::backends
