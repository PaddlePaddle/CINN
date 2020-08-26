#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/lang/packed_func.h"
#ifdef CINN_WITH_CUDA
#include "cinn/runtime/cuda/cuda_module.h"
#endif

namespace cinn {
namespace backends {

class Compiler final {
 public:
  static std::unique_ptr<Compiler> Create(const Target& target) {
    return std::unique_ptr<Compiler>(new Compiler(target));
  }

  /**
   * Compile and link to a CINN module.
   */
  void Build(const lang::Module& module);

  /**
   * Retrieve a function by \p fn_name.
   * @return function address or null if not exists.
   */
  lower_func_ptr_t Lookup(std::string_view fn_name);

 private:
  void CompileCudaModule(const lang::Module& module);

  void CompileX86Module(const lang::Module& module);

  explicit Compiler(const Target& target) : target_(target), engine_(ExecutionEngine::Create(ExecutionOptions())) {}

  CINN_DISALLOW_COPY_AND_ASSIGN(Compiler);

 private:
  Target target_;
  std::unique_ptr<ExecutionEngine> engine_;

#ifdef CINN_WITH_CUDA
  std::unique_ptr<runtime::cuda::CUDAModule> cuda_module_;
#endif
};

}  // namespace backends
}  // namespace cinn
