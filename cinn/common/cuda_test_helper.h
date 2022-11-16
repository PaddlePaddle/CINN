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

#include <string>
#include <vector>

#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"

namespace cinn {
namespace common {

#ifdef CINN_WITH_CUDA

#define CUDA_CALL(func)                                            \
  {                                                                \
    auto status = func;                                            \
    if (status != cudaSuccess) {                                   \
      LOG(FATAL) << "CUDA Error : " << cudaGetErrorString(status); \
    }                                                              \
  }

class CudaModuleTester {
 public:
  CudaModuleTester();

  // Call the host function in JIT.
  void operator()(const std::string& fn_name, void* args, int arg_num);

  void Compile(const ir::Module& m, const std::string& rewrite_cuda_code = "");

  void* LookupKernel(const std::string& name);

  void* CreateDeviceBuffer(const cinn_buffer_t* host_buffer);

  ~CudaModuleTester();

 private:
  std::unique_ptr<backends::SimpleJIT> jit_;

  void* stream_{};

  std::vector<void*> kernel_handles_;

  void* cuda_module_{nullptr};
};

class CudaMem {
 public:
  CudaMem() = default;

  void* mutable_data(size_t bytes) {
    CHECK_GT(bytes, 0) << "Cannot allocate empty memory!";
    if (ptr) {
      CHECK_EQ(bytes, bytes_) << "Try allocate memory twice!";
      return ptr;
    }
    CUDA_CALL(cudaMalloc(&ptr, bytes));
    bytes_ = bytes;
    return ptr;
  }

  template <typename T>
  T* mutable_data(size_t num) {
    return reinterpret_cast<T*>(mutable_data(num * sizeof(T)));
  }

  void* data() const {
    CHECK(ptr) << "Try get nullptr!";
    return ptr;
  }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(data());
  }

  void MemcpyFromHost(const void* src, size_t bytes, cudaStream_t stream = nullptr) {
    CHECK_LE(bytes, bytes_) << "Too many data need copy";
    CUDA_CALL(cudaMemcpyAsync(ptr, src, bytes, cudaMemcpyHostToDevice, stream));
  }

  void MemcpyToHost(void* dst, size_t bytes, cudaStream_t stream = nullptr) {
    CHECK_LE(bytes, bytes_) << "Too many data need copy";
    CUDA_CALL(cudaMemcpyAsync(dst, ptr, bytes, cudaMemcpyDeviceToHost, stream));
  }

  ~CudaMem() {
    if (ptr) {
      cudaFree(ptr);
    }
    bytes_ = 0;
  }

 private:
  void* ptr{nullptr};
  size_t bytes_{0};
};

#endif

}  // namespace common
}  // namespace cinn
