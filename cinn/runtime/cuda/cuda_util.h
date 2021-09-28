#pragma once

#include <cudnn.h>

#include <absl/container/flat_hash_map.h>
#include <string>

#include "cinn/runtime/cinn_runtime.h"
#include "cublas_v2.h"

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

class SerialData {
 public:
  ~SerialData();
  SerialData(const SerialData&) = delete;
  SerialData& operator=(const SerialData&) = delete;
  static SerialData& get_instance() {
    static SerialData instance;
    return instance;
  }
  absl::flat_hash_map<std::string, int>& GetMap() { return get_algo; }

 private:
  SerialData();
  absl::flat_hash_map<std::string, int> get_algo;
};

class CudnnHandle {
 public:
  ~CudnnHandle();
  CudnnHandle(const CudnnHandle&) = delete;
  CudnnHandle& operator=(const CudnnHandle&) = delete;
  static CudnnHandle& get_instance() {
    static CudnnHandle instance;
    return instance;
  }
  cudnnHandle_t& GetCudnnHandle() { return cudnn; }
  float* GetWorkSpace(size_t size);

 private:
  CudnnHandle();
  cudnnHandle_t cudnn;
  float* work_space;
  size_t size_;
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

void cinn_gpu_cudnn_conv2d(const std::vector<int>& attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* weights,
                           cinn_buffer_t* output);

void cinn_gpu_cudnn_pool2d(const std::vector<int>& attrs,
                           const std::vector<std::string>& str_attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* output);

void cinn_gpu_cudnn_softmax(const std::vector<int>& attrs, cinn_buffer_t* input, cinn_buffer_t* output);

void cinn_gpu_cublas_mul(const std::vector<int>& attrs,
                         cinn_buffer_t* input1,
                         cinn_buffer_t* input2,
                         cinn_buffer_t* output);
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
