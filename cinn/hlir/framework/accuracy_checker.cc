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

bool AccuracyChecker::operator()(std::map<std::string, bool>* out_args_check_result) {
  bool res = false;
  for (auto& name : out_args_) {
    bool res_cur = false;
    auto tensor  = scope_->GetTensor(name);
    if (tensor->type().is_float()) {
      Tensor cpu_tensor = CopyTensorToCpu<float>(tensor);
      res_cur           = CheckNanOrInf<float>(cpu_tensor);
    } else if (tensor->type().is_int()) {
      Tensor cpu_tensor = CopyTensorToCpu<int>(tensor);
      res_cur           = CheckNanOrInf<int>(cpu_tensor);
    } else {
      CHECK(false) << "Not supported data type.";
    }
    out_args_check_result->emplace(name, res_cur);
    res |= res_cur;
  }
  return res;
}

bool AccuracyChecker::operator()(const std::map<std::string, cinn_pod_value_t>& name2podargs,
                                 std::map<std::string, bool>* out_args_check_result) {
  bool res = false;
  for (auto& name : out_args_) {
    bool res_cur                = false;
    const cinn_buffer_t* buffer = cinn_pod_value_to_buffer_p(const_cast<cinn_pod_value_t*>(&name2podargs.at(name)));
    if (buffer->type == cinn_float32_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<float>(buffer);
      res_cur           = CheckNanOrInf<float>(cpu_tensor);
    } else if (buffer->type == cinn_int32_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<int32_t>(buffer);
      res_cur           = CheckNanOrInf<int32_t>(cpu_tensor);
    } else if (buffer->type == cinn_int64_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<int64_t>(buffer);
      res_cur           = CheckNanOrInf<int64_t>(cpu_tensor);
    } else {
      CHECK(false) << "Not supported data type.";
    }
    out_args_check_result->emplace(name, res_cur);
    res |= res_cur;
  }
  return res;
}

template <typename T>
Tensor AccuracyChecker::CopyTensorToCpu(const Tensor& tensor) {
  Tensor cpu_tensor;
  cpu_tensor->Resize(tensor->shape());
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = tensor->data<T>();
  size_t numel = tensor->shape().numel();
  MemcpyDeviceToHost(src, numel, dst);

  return cpu_tensor;
}

template <typename T>
Tensor AccuracyChecker::CopyBufferToCpu(const cinn_buffer_t* buffer) {
  std::vector<int> shape;
  shape.resize(buffer->dimensions);
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = buffer->dims[i];
  }

  Tensor cpu_tensor;
  cpu_tensor->Resize(Shape(shape));
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = reinterpret_cast<const T*>(buffer->memory);
  size_t numel = cpu_tensor->shape().numel();
  MemcpyDeviceToHost(src, numel, dst);

  return cpu_tensor;
}

template <typename T>
void AccuracyChecker::MemcpyDeviceToHost(const T* src, size_t numel, T* dst) {
#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst, src, numel * sizeof(T), cudaMemcpyDeviceToHost);
#else
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = src[i];
  }
#endif
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
