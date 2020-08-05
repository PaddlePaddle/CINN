#include "cinn/common/cuda_test_helper.h"

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"

namespace cinn {
namespace common {

#ifdef CINN_WITH_CUDA
void CudaModuleTester::Compile(const lang::Module& m, const std::string& rewrite_cuda_code) {
  auto [host_module, device_module] = backends::SplitCudaAndHostModule(m);  // NOLINT
  CHECK(!host_module.functions().empty());
  CHECK(!device_module.functions().empty());

  backends::CodeGenCUDA_Dev codegen(DefaultHostTarget());
  auto source_code = codegen.Compile(m);

  // compile CUDA kernel.
  backends::NVRTC_Compiler compiler;

  std::string ptx;
  if (rewrite_cuda_code.empty())
    ptx = compiler(source_code);
  else
    ptx = compiler(rewrite_cuda_code);

  cuda_module_ = new runtime::cuda::CUDAModule(ptx, runtime::cuda::CUDAModule::Kind::PTX);

  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    auto fn_kernel = reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_)->GetFunction(0, kernel_fn_name);
    CHECK(fn_kernel);
    kernel_handles_.push_back(fn_kernel);

    backends::RuntimeSymbolRegistry::Global().RegisterFn(kernel_fn_name + "_ptr_",
                                                         reinterpret_cast<void*>(&kernel_handles_.back()));
    backends::RuntimeSymbolRegistry::Global().RegisterVar(kernel_fn_name + "_stream_ptr_", stream_);
  }

  jit_ = backends::SimpleJIT::Create();

  // compile host module
  jit_->Link<backends::CodeGenCUDA_Host>(host_module, false);
}

void* CudaModuleTester::CreateDeviceBuffer(const cinn_buffer_t* host_buffer) {
  CHECK(host_buffer->host_memory);
  int num_bytes = host_buffer->num_elements() * sizeof(float);
  CUdeviceptr data;
  cuMemAlloc(&data, num_bytes);

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data), host_buffer->host_memory, num_bytes, cudaMemcpyHostToDevice));
  return reinterpret_cast<void*>(data);
}

CudaModuleTester::CudaModuleTester() {}

void CudaModuleTester::operator()(const std::string& fn_name, void* args, int arg_num) {
  auto fn  = jit_->Lookup(fn_name);
  auto fnp = reinterpret_cast<lower_func_ptr_t>(fn);
  (*fnp)(args, arg_num);
}

void* CudaModuleTester::LookupKernel(const std::string& name) {
  return reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_)->GetFunction(0, name);
}

CudaModuleTester::~CudaModuleTester() {
  if (cuda_module_) {
    delete reinterpret_cast<runtime::cuda::CUDAModule*>(cuda_module_);
  }
}

#endif

}  // namespace common
}  // namespace cinn
