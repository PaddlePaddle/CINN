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

#include <string.h>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "cinn/runtime/custom_function.h"

namespace cinn {
namespace runtime {

using common::Target;
using hlir::framework::Shape;
using hlir::framework::Tensor;

namespace utils {
bool MemcpyToHost(void* dst, const void* src, size_t bytes, const Target& input_target, void* stream = nullptr) {
  if (input_target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    const auto& cuda_stream = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    return true;
#else
    LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
    return false;
#endif
  }
  if (input_target == common::DefaultHostTarget()) {
    memcpy(dst, src, bytes);
    return true;
  }
  LOG(FATAL) << "MemcpyToHost Only support cpu or nvgpu -> cpu, but here the input target is " << input_target
             << "! Please check.";
  return false;
}

bool MemcpyToDevice(void* dst, const void* src, size_t bytes, const Target& input_target, void* stream = nullptr) {
#ifdef CINN_WITH_CUDA
  if (input_target == common::DefaultNVGPUTarget()) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
    return true;
  } else if (input_target == common::DefaultHostTarget()) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
    return true;
  } else {
    LOG(FATAL) << "MemcpyToDevice only support cpu or nvgpu -> nvgpu, but here the input target is " << input_target
               << "! Please check.";
    return false;
  }
#else
  LOG(FATAL)
      << "MemcpyToDevice only support nvgpu, and NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
  return false;
#endif
}
}  // namespace utils

void CheckAssertTrue(
    const bool* x, const size_t numel, bool only_warning, const std::string& msg, const Target& target) {
  // check false number and first false offset
  int error_num = 0, first_diff = -1;
  for (int i = 0; i < numel; ++i) {
    if (!x[i]) {
      ++error_num;
      if (first_diff == -1) {
        first_diff = i;
      }
    }
  }

  // raise error information
  if (error_num > 0) {
    std::string error_info = "[AssertTrue] Check failed!\n";
    error_info += "\t- target: " + target.arch_str() + "\n";
    error_info += "\t- assert false number: " + std::to_string(error_num) + "\n";
    error_info += "\t- first false offset: " + std::to_string(first_diff) + "\n";
    error_info += "\t- message: " + msg;

    if (only_warning) {
      LOG(WARNING) << error_info;
    } else {
      LOG(FATAL) << error_info;
    }
  } else {
    VLOG(1) << "[AssertTrue] Check succeed!\n"
            << "\t- message: " + msg;
  }
}

void CinnAssertTrue(const std::vector<int>& attrs,
                    const std::vector<std::string>& str_attrs,
                    cinn_buffer_t* x,
                    cinn_buffer_t* output,
                    const Target& target,
                    void* stream) {
  CHECK(x->type == cinn_bool_t()) << "The input type of AssertTrue should be bool, but here " << x->type.bits
                                  << "! Please check.";
  CHECK(output->type == cinn_bool_t()) << "The output type of AssertTrue should be bool, but here " << output->type.bits
                                       << "! Please check.";

  CHECK_EQ(attrs.size(), 1UL) << "The AssertTrue should has 'only_warning' attribute! Please check.";
  bool only_warning = attrs.front();

  CHECK_EQ(str_attrs.size(), 1UL) << "The AssertTrue should has 'only_warning' attribute! Please check.";
  std::string msg = str_attrs.front();

  // create cpu tensor
  std::vector<int> shape;
  shape.resize(x->dimensions);
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = x->dims[i];
  }

  Tensor cpu_tensor;
  cpu_tensor->Resize(Shape(shape));
  bool* dst = cpu_tensor->mutable_data<bool>(common::DefaultHostTarget());

  // copy data from gpu to cpu
  const bool* src = reinterpret_cast<const bool*>(x->memory);
  size_t numel    = cpu_tensor->shape().numel();
  utils::MemcpyToHost(dst, src, numel * sizeof(bool), target, stream);

  CheckAssertTrue(dst, numel, only_warning, msg, target);

  if (target == common::DefaultNVGPUTarget()) {
    utils::MemcpyToDevice(output->memory, x->memory, numel * sizeof(bool), target, stream);
  } else {
    utils::MemcpyToHost(output->memory, x->memory, numel * sizeof(bool), target, stream);
  }
}

}  // namespace runtime
}  // namespace cinn
