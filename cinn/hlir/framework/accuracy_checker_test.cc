// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/framework/accuracy_checker.h"

#include <gtest/gtest.h>

#include <random>
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace hlir {
namespace framework {

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

void GenerateNanTensor(Tensor tensor, Target target) {
  size_t numel = tensor->shape().numel();
  float* dst   = tensor->mutable_data<float>(target);

  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-100.f, 100.f);
  std::vector<float> random_nan_vec(numel);
  for (size_t i = 0; i < numel; i++) {
    random_nan_vec[i] = sqrt(dist(engine));
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst, random_nan_vec.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_nan_vec.begin(), random_nan_vec.end(), dst);
#endif
}

TEST(AccuracyChecker, tensor) {
  Target target = GetTarget();
  Scope scope;
  scope.Var<Tensor>("out");
  auto out = scope.GetTensor("out");
  out->Resize(Shape({16, 16}));
  GenerateNanTensor(out, target);

  AccuracyChecker checker(target, &scope, {}, {"out"});
  CHECK(checker());
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
