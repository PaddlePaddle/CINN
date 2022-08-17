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
std::string GetTypeString() {
  if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, int32_t>::value) {
    return "int32_t";
  } else if (std::is_same<T, int64_t>::value) {
    return "int64_t";
  } else if (std::is_same<T, bool>::value) {
    return "bool";
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

template <typename T>
std::string DebugString(const Tensor& cpu_tensor, const std::string& name, const CheckResult& res) {
  std::stringstream ss;
  ss << "name=" << name << ", dtype=" << GetTypeString<T>() << ", shape=" << cpu_tensor->shape().data() << ", data=[";
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

std::string AccuracyChecker::operator()(const std::string& arg_name) {
  auto tensor = scope_->GetTensor(arg_name);
  if (tensor->type().is_float()) {
    return CheckTensor<float>(tensor, arg_name);
  } else if (tensor->type().is_int()) {
    return CheckTensor<int32_t>(tensor, arg_name);
  } else if (tensor->type().is_bool()) {
    return CheckTensor<bool>(tensor, arg_name);
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

std::string AccuracyChecker::operator()(const std::map<std::string, cinn_pod_value_t>* name2podargs,
                                        const std::string& arg_name) {
  CHECK(name2podargs) << "name2podargs should not be nullptr.";
  const cinn_buffer_t* buffer = cinn_pod_value_to_buffer_p(const_cast<cinn_pod_value_t*>(&name2podargs->at(arg_name)));
  if (buffer->type == cinn_float32_t()) {
    return CheckBuffer<float>(buffer, arg_name);
  } else if (buffer->type == cinn_int32_t()) {
    return CheckBuffer<int32_t>(buffer, arg_name);
  } else if (buffer->type == cinn_int64_t()) {
    return CheckBuffer<int64_t>(buffer, arg_name);
  } else if (buffer->type == cinn_bool_t()) {
    return CheckBuffer<bool>(buffer, arg_name);
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

template <typename T>
std::string AccuracyChecker::CheckTensor(const Tensor& tensor, const std::string& arg_name) {
  Tensor cpu_tensor;
  cpu_tensor->Resize(tensor->shape());
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = tensor->data<T>();
  size_t numel = tensor->shape().numel();
  MemcpyDeviceToHost(src, numel, dst);

  auto res        = CheckNanOrInf<T>(cpu_tensor);
  auto result_str = DebugString<T>(cpu_tensor, arg_name, res);
  return result_str;
}

template <typename T>
std::string AccuracyChecker::CheckBuffer(const cinn_buffer_t* buffer, const std::string& arg_name) {
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

  auto res        = CheckNanOrInf<T>(cpu_tensor);
  auto result_str = DebugString<T>(cpu_tensor, arg_name, res);
  return result_str;
}

template <typename T>
void AccuracyChecker::MemcpyDeviceToHost(const T* src, size_t numel, T* dst) {
#ifdef CINN_WITH_CUDA
  if (target_ == common::DefaultNVGPUTarget()) {
    cudaMemcpy(dst, src, numel * sizeof(T), cudaMemcpyDeviceToHost);
    return;
  }
#endif
  if (target_ == common::DefaultHostTarget()) {
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = src[i];
    }
  } else {
    CHECK(false) << "Not supported target type.";
  }
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
