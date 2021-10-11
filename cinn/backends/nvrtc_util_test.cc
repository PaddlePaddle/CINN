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
