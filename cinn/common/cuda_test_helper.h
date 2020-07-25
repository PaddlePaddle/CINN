#pragma once

#include <string>
#include <vector>

#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"

namespace cinn {
namespace common {

#ifdef CINN_WITH_CUDA
class CudaModuleTester {
 public:
  CudaModuleTester();

  // Call the host function in JIT.
  void operator()(const std::string& fn_name, void* args, int arg_num);

  void Compile(const lang::Module& m);

  void* LookupKernel(const std::string& name);

  void* CreateDeviceBuffer(const cinn_buffer_t* host_buffer);

  ~CudaModuleTester();

 private:
  std::unique_ptr<backends::SimpleJIT> jit_;

  void* stream_{};

  std::vector<void*> kernel_handles_;

  void* cuda_module_{nullptr};
};

#endif

}  // namespace common
}  // namespace cinn
