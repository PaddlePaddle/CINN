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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <string>
#include <vector>

#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace cuda {

const int kCUDAMaxCards{10};
class CublasHandle {
 public:
  ~CublasHandle();
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  static CublasHandle& get_instance() {
    static CublasHandle instance;
    return instance;
  }
  cublasHandle_t& GetCublasHandle() { return cublas; }

 private:
  CublasHandle();
  cublasHandle_t cublas;
};

/**
 * Call a CUDA compiled kernel.
 *
 * @param kernel_fn the compiled PTX kernel.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_cuda_kernel(void* kernel_fn,
                           cinn_pod_value_t* args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream);

void cinn_gpu_cudnn_conv2d(const absl::flat_hash_map<std::string, int>& attr,
                           cinn_buffer_t* x,
                           cinn_buffer_t* w,
                           cinn_buffer_t* y,
                           const cudaStream_t& stream = nullptr);

void cinn_gpu_cudnn_conv2d_backward_data(const absl::flat_hash_map<std::string, int>& attr,
                                         cinn_buffer_t* w,
                                         cinn_buffer_t* dy,
                                         cinn_buffer_t* dx,
                                         const cudaStream_t& stream = nullptr);

void cinn_gpu_cudnn_conv2d_backward_filter(const absl::flat_hash_map<std::string, int>& attr,
                                           cinn_buffer_t* x,
                                           cinn_buffer_t* dy,
                                           cinn_buffer_t* dw,
                                           const cudaStream_t& stream = nullptr);

void cinn_gpu_cudnn_pool2d(const std::vector<int>& attrs,
                           const std::vector<std::string>& str_attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* output,
                           const cudaStream_t& stream = nullptr);

void cinn_gpu_cudnn_softmax(const std::vector<int>& attrs,
                            cinn_buffer_t* input,
                            cinn_buffer_t* output,
                            const cudaStream_t& stream = nullptr);

void cinn_gpu_cublas_mul(const std::vector<int>& attrs,
                         cinn_buffer_t* input1,
                         cinn_buffer_t* input2,
                         cinn_buffer_t* output,
                         const cudaStream_t& stream = nullptr);
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
