#pragma once

#include <llvm/IR/Instruction.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Target/TargetMachine.h>

#include <functional>

namespace cinn::backends {

// TODO(fc500110): define class OptimizeOptions

// llvm module optimizer
class LLVMModuleOptimizer {
 public:
  explicit LLVMModuleOptimizer(llvm::TargetMachine *machine,
                               int opt_level,
                               llvm::FastMathFlags fast_math_flags,
                               bool print_passes = false);
  void operator()(llvm::Module *m);

 private:
  llvm::TargetMachine *machine_;
  int opt_level_;
  bool print_passes_;
};
}  // namespace cinn::backends
