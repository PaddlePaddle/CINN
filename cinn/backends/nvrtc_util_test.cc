#include "cinn/backends/nvrtc_util.h"

#include <gtest/gtest.h>

namespace cinn {
namespace backends {

TEST(NVRTC_Compiler, basic) {
  NVRTC_Compiler compiler;

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

  LOG(INFO) << "ptx:\n" << ptx;
}

}  // namespace backends
}  // namespace cinn
