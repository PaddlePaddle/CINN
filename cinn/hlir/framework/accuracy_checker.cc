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

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace hlir {
namespace framework {

bool AccuracyChecker::operator()() {
  bool res = false;
  for (auto& name : out_args_) {
    auto tensor = scope_->GetTensor(name);
    if (tensor->type().is_float()) {
      Tensor cpu_tensor = CopyToCpu<float>(tensor);
      res |= CheckNanOrInf<float>(cpu_tensor);
    }
  }
  return res;
}

template <typename T>
Tensor AccuracyChecker::CopyToCpu(const Tensor& tensor) {
  Tensor cpu_tensor;
  cpu_tensor->Resize(tensor->shape());
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = tensor->data<T>();
  size_t numel = tensor->shape().numel();
#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst, src, numel * sizeof(T), cudaMemcpyDeviceToHost);
#else
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = src[i];
  }
#endif

  return cpu_tensor;
}

template <typename T>
bool AccuracyChecker::CheckNanOrInf(const Tensor& cpu_tensor) {
  bool flag     = false;
  size_t numel  = cpu_tensor->shape().numel();
  const T* data = cpu_tensor->data<T>();
  for (size_t i = 0; i < numel; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      flag = true;
      break;
    }
  }
  return flag;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
