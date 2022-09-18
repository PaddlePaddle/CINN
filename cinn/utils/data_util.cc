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

#include "cinn/utils/data_util.h"

#include "iostream"

namespace cinn {

void SetRandInt(hlir::framework::Tensor tensor, const common::Target& target, int seed) {
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<int> dist(1, 10);
  size_t num_ele = tensor->shape().numel();
  std::vector<int> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = static_cast<int>(dist(engine));  // All random data
  }

  auto* data = tensor->mutable_data<int>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data, random_data.data(), num_ele * sizeof(int), cudaMemcpyHostToDevice);
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(random_data.begin(), random_data.end(), data);
  std::cout << "success" << std::endl;
}

template <>
void SetRandData<int>(hlir::framework::Tensor tensor, const common::Target& target, int seed) {
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<int> dist(1, 10);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = static_cast<float>(dist(engine));  // All random data
  }

  auto* data = tensor->mutable_data<float>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(random_data.begin(), random_data.end(), data);
}

template <>
void SetRandData<float>(hlir::framework::Tensor tensor, const common::Target& target, int seed) {
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  std::default_random_engine engine(seed);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }

  auto* data = tensor->mutable_data<float>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
  } else if (target == common::DefaultHostTarget()) {
    std::copy(random_data.begin(), random_data.end(), data);
  } else {
    CINN_NOT_IMPLEMENTED
  }
#else
  CHECK(target == common::DefaultHostTarget());
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

template <>
std::vector<float> GetTensorData<float>(const hlir::framework::Tensor& tensor, const common::Target& target) {
  auto size = tensor->shape().numel();
  std::vector<float> data(size);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(
        data.data(), static_cast<const void*>(tensor->data<float>()), size * sizeof(float), cudaMemcpyDeviceToHost);
  } else if (target == common::DefaultHostTarget()) {
    std::copy(tensor->data<float>(), tensor->data<float>() + size, data.begin());
  } else {
    CINN_NOT_IMPLEMENTED
  }
#else
  CHECK(target == common::DefaultHostTarget());
  std::copy(tensor->data<float>(), tensor->data<float>() + size, data.begin());
#endif
  return data;
}

}  // namespace cinn
