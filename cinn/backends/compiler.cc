#include "cinn/backends/compiler.h"

#include "cinn/backends/llvm/runtime_symbol_registry.h"
#ifdef CINN_WITH_CUDA
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#endif

namespace cinn {
namespace backends {

void Compiler::Build(const lang::Module& module) {
  if (target_.arch == Target::Arch::NVGPU) {
    CompileCudaModule(module);
  } else if (target_.arch == Target::Arch::X86) {
    CompileX86Module(module);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void Compiler::CompileCudaModule(const Module& module) {
#ifdef CINN_WITH_CUDA
  auto [host_module, device_module] = SplitCudaAndHostModule(module);  // NOLINT
  LOG(INFO) << "host module:\n" << host_module;

  {  // compile cuda device
    LOG(INFO) << "device module:\n" << device_module;
    CodeGenCUDA_Dev codegen(target_);
    auto source_code = codegen.Compile(device_module);
    LOG(INFO) << "source code:\n" << source_code;
    using runtime::cuda::CUDAModule;

    backends::NVRTC_Compiler compiler;

    auto ptx = compiler(source_code);
    CHECK(!ptx.empty());

    // TODO(Superjomn) Whether to support multiple CUDA modules?
    cuda_module_.reset(new CUDAModule(ptx, CUDAModule::Kind::PTX));

    for (auto& fn : device_module.functions()) {
      std::string kernel_fn_name = fn->name;
      auto fn_kernel             = cuda_module_->GetFunction(0, kernel_fn_name);
      CHECK(fn_kernel);

      backends::RuntimeSymbolRegistry::Global().RegisterVar(kernel_fn_name + "_ptr_",
                                                            reinterpret_cast<void*>(fn_kernel));
      cudaStream_t stream = nullptr;
      backends::RuntimeSymbolRegistry::Global().RegisterVar(kernel_fn_name + "_stream_ptr_", stream);
    }
  }

  {  // compile host jit
    engine_ = SimpleJIT::Create();
    engine_->Link<CodeGenCUDA_Host>(host_module);
  }

#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileX86Module(const Module& module) { engine_->Link(module); }

lower_func_ptr_t Compiler::GetFn(std::string_view fn_name) {
  CHECK(engine_);
  return reinterpret_cast<lower_func_ptr_t>(engine_->Lookup(fn_name));
}

}  // namespace backends
}  // namespace cinn
