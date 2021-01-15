#include "cinn/hlir/framework/memory.h"

#ifdef CINN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "cinn/backends/cuda_util.h"
#endif

namespace cinn {
namespace hlir {
namespace framework {

using common::Target;

namespace {

class X86MemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override { return ::malloc(nbytes); }
  void free(void* data) override {
    if (!data) return;
    ::free(data);
  }
  void* aligned_alloc(size_t alignment, size_t nbytes) override { return ::aligned_alloc(alignment, nbytes); }
};

#ifdef CINN_WITH_CUDA
class CudaMemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override {
    void* data;
    CUDA_CALL(cudaMalloc(&data, nbytes));
    return data;
  }

  void free(void* data) override { CUDA_CALL(cudaFree(data)); }
};

#endif

}  // namespace

MemoryManager::MemoryManager() {
  Register(Target::Arch::Unk, new X86MemoryMng);
  Register(Target::Arch::X86, new X86MemoryMng);
#ifdef CINN_WITH_CUDA
  Register(Target::Arch::NVGPU, new CudaMemoryMng);
#endif
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
