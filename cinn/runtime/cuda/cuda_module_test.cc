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

#include "cinn/backends/nvrtc/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/cuda/use_extern_funcs.h"

namespace cinn {
namespace runtime {
namespace cuda {

TEST(CUDAModule, basic) {
  backends::nvrtc::Compiler compiler;

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
