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

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <sstream>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/custom_function.h"

namespace cinn {
namespace runtime {

class CinnBufferAllocHelper {
 public:
  CinnBufferAllocHelper(cinn_device_kind_t device, cinn_type_t type, const std::vector<int>& shape, int align = 0) {
    buffer_ = cinn_buffer_t::new_(device, type, shape, align);
  }

  template <typename T>
  T* mutable_data(const Target& target) {
    if (target_ != common::UnkTarget()) {
      CHECK_EQ(target, target_) << "Cannot alloc twice, the memory had alloced at " << target_ << "! Please check.";
      return reinterpret_cast<T*>(buffer_->memory);
    }

    target_ = target;
    if (target == common::DefaultHostTarget()) {
      cinn_buffer_malloc(nullptr, buffer_);
    } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
      cudaMalloc(&buffer_->memory, buffer_->num_elements() * sizeof(T));
#else
      LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
    } else {
      LOG(FATAL) << "Only support nvgpu and cpu, but here " << target << "! Please check.";
    }

    return reinterpret_cast<T*>(buffer_->memory);
  }

  template <typename T>
  const T* data() {
    if (target_ == common::UnkTarget()) {
      LOG(FATAL) << "No memory had alloced! Please check.";
    }
    return reinterpret_cast<const T*>(buffer_->memory);
  }

  ~CinnBufferAllocHelper() {
    if (buffer_) {
      if (target_ == common::UnkTarget()) {
        // pass
      } else if (target_ == common::DefaultHostTarget()) {
        cinn_buffer_free(nullptr, buffer_);
      } else if (target_ == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
        cudaFree(buffer_->memory);
#else
        LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
      } else {
        LOG(FATAL) << "Only support nvgpu and cpu, but here " << target_ << "! Please check.";
      }
      delete buffer_;
    }
  }

  cinn_buffer_t& operator*() const noexcept { return *buffer_; }
  cinn_buffer_t* operator->() const noexcept { return buffer_; }
  cinn_buffer_t* get() const noexcept { return buffer_; }

 private:
  cinn_buffer_t* buffer_{nullptr};
  Target target_{common::UnkTarget()};
};

template <typename T>
void SetInputValue(T* input, const T* input_h, size_t num, const Target& target) {
  if (target == common::DefaultHostTarget()) {
    for (int i = 0; i < num; ++i) {
      input[i] = input_h[i];
    }
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    cudaMemcpy(input, input_h, num * sizeof(T), cudaMemcpyHostToDevice);
#else
    LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
  }
}

TEST(CinnAssertTrue, test_true) {
  Target target = common::DefaultTarget();

  CinnBufferAllocHelper x(cinn_x86_device, cinn_bool_t(), {1});

  // set inpute value true
  bool input_h = true;
  auto* input  = x.mutable_data<bool>(target);

  SetInputValue(input, &input_h, 1, target);

  CinnBufferAllocHelper y(cinn_x86_device, cinn_bool_t(), {1});
  auto* output = y.mutable_data<bool>(target);

  cinn_pod_value_t v_args[2] = {cinn_pod_value_t(x.get()), cinn_pod_value_t(y.get())};

  std::stringstream ss;
  ss << "Test AssertTrue(true) on " << target;
  cinn_assert_true(v_args, std::hash<std::string>()(ss.str()), false);

  if (target == common::DefaultHostTarget()) {
    ASSERT_EQ(input[0], output[0]) << "The output of AssertTrue should be the same as input";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    bool output_h = false;
    cudaMemcpy(&output_h, output, sizeof(bool), cudaMemcpyDeviceToHost);

    ASSERT_EQ(input_h, output_h) << "The output of AssertTrue should be the same as input";
#endif
  }
}

TEST(CinnAssertTrue, test_false_only_warning) {
  Target target = common::DefaultTarget();

  CinnBufferAllocHelper x(cinn_x86_device, cinn_bool_t(), {1});

  // set inpute value false
  bool input_h = false;
  auto* input  = x.mutable_data<bool>(target);

  SetInputValue(input, &input_h, 1, target);

  CinnBufferAllocHelper y(cinn_x86_device, cinn_bool_t(), {1});
  auto* output = y.mutable_data<bool>(target);

  cinn_pod_value_t v_args[2] = {cinn_pod_value_t(x.get()), cinn_pod_value_t(y.get())};

  std::stringstream ss;
  ss << "Test AssertTrue(false, only_warning=true) on " << target;
  cinn_assert_true(v_args, std::hash<std::string>()(ss.str()), true);

  if (target == common::DefaultHostTarget()) {
    ASSERT_EQ(input[0], output[0]) << "The output of AssertTrue should be the same as input";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    bool output_h = false;
    cudaMemcpy(&output_h, output, sizeof(bool), cudaMemcpyDeviceToHost);

    ASSERT_EQ(input_h, output_h) << "The output of AssertTrue should be the same as input";
#endif
  }
}

}  // namespace runtime
}  // namespace cinn
