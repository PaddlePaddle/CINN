#include "cinn/runtime/cuda/cuda_module.h"

#include <gtest/gtest.h>

#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/cuda/use_extern_funcs.h"

namespace cinn {
namespace runtime {
namespace cuda {

TEST(CUDAModule, basic) {
  backends::NVRTC_Compiler compiler;

  std::string source_code = R"ROC(
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}
)ROC";

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule module(ptx, CUDAModule::Kind::PTX);
  auto func = module.GetFunction(0, "saxpy");
  ASSERT_TRUE(func);
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
