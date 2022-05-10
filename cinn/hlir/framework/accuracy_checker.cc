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

template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{";
  bool is_first = true;
  for (auto e : vec) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    os << e;
  }
  os << "}";
  return os;
}

template <typename T>
std::string DebugString(const Tensor& cpu_tensor,
                        const std::string& name,
                        const std::string& dtype_str,
                        const CheckResult& res) {
  std::stringstream ss;
  ss << "name=" << name << ", dtype=" << dtype_str << ", shape=" << cpu_tensor->shape().data() << ", data=[";
  size_t numel  = cpu_tensor->shape().numel();
  const T* data = cpu_tensor->data<T>();
  if (numel <= 10) {
    for (size_t i = 0; i < numel; ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << data[i];
    }
  } else {
    for (size_t i = 0; i < 5; ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << data[i];
    }
    ss << " ... ";
    for (size_t i = numel - 5; i < numel; ++i) {
      ss << data[i];
      if (i != numel - 1) {
        ss << ", ";
      }
    }
  }
  ss << "]";
  if (res == CheckResult::kZero) {
    ss << ", Zero";
  } else if (res == CheckResult::kNaN) {
    ss << ", NaN";
  } else if (res == CheckResult::kInf) {
    ss << ", Inf";
  } else {
    ss << ", OK";
  }
  return ss.str();
}

std::string AccuracyChecker::operator()(const std::map<std::string, cinn_pod_value_t>* name2podargs,
                                        const std::string& arg_name) {
  std::string result_str;
  if (!name2podargs) {
    auto tensor = scope_->GetTensor(arg_name);
    if (tensor->type().is_float()) {
      Tensor cpu_tensor = CopyTensorToCpu<float>(tensor);
      auto res          = CheckNanOrInf<float>(cpu_tensor);
      result_str        = DebugString<float>(cpu_tensor, arg_name, "float", res);
    } else if (tensor->type().is_int()) {
      Tensor cpu_tensor = CopyTensorToCpu<int>(tensor);
      auto res          = CheckNanOrInf<int>(cpu_tensor);
      result_str        = DebugString<int>(cpu_tensor, arg_name, "int", res);
    } else {
      CHECK(false) << "Not supported data type.";
    }
  } else {
    const cinn_buffer_t* buffer =
        cinn_pod_value_to_buffer_p(const_cast<cinn_pod_value_t*>(&name2podargs->at(arg_name)));
    if (buffer->type == cinn_float32_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<float>(buffer);
      auto res          = CheckNanOrInf<float>(cpu_tensor);
      result_str        = DebugString<float>(cpu_tensor, arg_name, "float", res);
    } else if (buffer->type == cinn_int32_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<int32_t>(buffer);
      auto res          = CheckNanOrInf<int32_t>(cpu_tensor);
      result_str        = DebugString<int32_t>(cpu_tensor, arg_name, "int32_t", res);
    } else if (buffer->type == cinn_int64_t()) {
      Tensor cpu_tensor = CopyBufferToCpu<int64_t>(buffer);
      auto res          = CheckNanOrInf<int64_t>(cpu_tensor);
      result_str        = DebugString<int64_t>(cpu_tensor, arg_name, "int64_t", res);
    } else {
      CHECK(false) << "Not supported data type.";
    }
  }
  return result_str;
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
CheckResult AccuracyChecker::CheckNanOrInf(const Tensor& cpu_tensor) {
  bool zero_flag = true;
  size_t numel   = cpu_tensor->shape().numel();
  const T* data  = cpu_tensor->data<T>();
  for (size_t i = 0; i < numel; ++i) {
    if (std::isnan(data[i])) {
      return CheckResult::kNaN;
    } else if (std::isinf(data[i])) {
      return CheckResult::kInf;
    } else if (data[i] != static_cast<T>(0)) {
      zero_flag = false;
    }
  }
  return zero_flag ? CheckResult::kZero : CheckResult::kOK;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
