#pragma once

#include <absl/container/flat_hash_map.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "cinn/backends/llvm/codegen_llvm.h"

namespace cinn {
namespace backends {

/**
 * CodeGenCUDA takes a CINN Module with host functions and output a LLVM module.
 */
class CodeGenCUDA_Host : public CodeGenLLVM {
 public:
  explicit CodeGenCUDA_Host(llvm::Module *m, llvm::IRBuilder<> *b, const std::shared_ptr<SymbolTable> &vars = nullptr)
      : CodeGenLLVM(m, b, vars) {}

  static std::string GenKernelPtrVarName(const std::string &kernel_name) { return kernel_name + "_kernel_ptr_"; }
  static std::string GenKernelStreamVarName(const std::string &kernel_name) {
    return kernel_name + "_kernel_stream_ptr_";
  }

  using CodeGenLLVM::Visit;

  llvm::Value *Visit(const ir::_LoweredFunc_ *func) override {
    if (func->is_gpu_host()) {
      return LowerGPUKernelLauncher(func);
    }
    return CodeGenLLVM::Visit(func);
  }

 private:
  /**
   * Lower a CUDA kernel launcher.
   *
   * We launch a CUDA kernel in the following way:
   *
   * 1. a GPU function (called fn) will compiled to PTX and lower by CUDA driver to a function pointer, which we store
   * as a `void*` type global variable [fn_kernel_ptr] in LLVM module.
   * 2. when lower the host launcher, we replace the Call of the original kernel [fn] to a Call of
   * `cinn_call_cuda_kernel` method which is registered as an external function.
   *
   */
  llvm::Value *LowerGPUKernelLauncher(const ir::_LoweredFunc_ *func);
};

}  // namespace backends
}  // namespace cinn
