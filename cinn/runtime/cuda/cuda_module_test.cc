// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

TEST(CUDAModule, int64) {
  backends::NVRTC_Compiler compiler;

  std::string source_code = R"ROC(
extern "C" {

#ifdef __CUDACC_RTC__
typedef long long int int64_t;
typedef int int32_t;
typedef char int8_t;
#endif

__global__
void __launch_bounds__(40) fn_index_select_268_substract_271_fused_kernel(const float* __restrict__ var_2230, const int64_t* __restrict__ bc_idx, const float* __restrict__ bc_v, float* __restrict__ substract_Out_43)
{
  if (((int)threadIdx.x < 40)) {
    substract_Out_43[(int)threadIdx.x] = (var_2230[((int32_t)(bc_idx[(int)threadIdx.x]))] - bc_v[(int)threadIdx.x]);
  };
}
}
)ROC";

  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  CUDAModule module(ptx, CUDAModule::Kind::PTX);
  auto func = module.GetFunction(0, "fn_index_select_268_substract_271_fused_kernel");
  ASSERT_TRUE(func);
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
