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

#include "cinn/runtime/cuda/cuda_util.h"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <algorithm>

#include <absl/container/flat_hash_map.h>
#include <cublas_v2.h>
#ifdef CINN_WITH_CUDNN
#include <cudnn.h>
#endif

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"
#include "cinn/runtime/flags.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {

class CublasHandle {
 public:
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  ~CublasHandle() { CUBLAS_CALL(cublasDestroy(cublas)); }
  static CublasHandle &GetInstance() {
    static CublasHandle instance;
    return instance;
  }
  cublasHandle_t &GetCublasHandle() { return cuhandle; }

 private:
  CublasHandle() { CUBLAS_CALL(cublasCreate(&cuhandle)); }
  cublasHandle_t cuhandle;
};

void cinn_call_cuda_kernel(void *kernel_fn,
                           void *v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void *stream) {
  VLOG(3) << "cinn_call_cuda_kernel, grid_dim={" << grid_x << ", " << grid_y << ", " << grid_z << "}, block_dim={"
          << block_x << ", " << block_y << ", " << block_z << "}, num_args=" << num_args << ", stream=" << stream;

  std::vector<void *> kernel_args;
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  for (int idx = 0; idx < num_args; ++idx) {
    if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
      kernel_args.push_back(&((cinn_buffer_t *)(args[idx]))->memory);
    } else {
      kernel_args.push_back(args[idx].data_addr());
    }
  }
  CUDA_DRIVER_CALL(cuLaunchKernel(static_cast<CUfunction>(kernel_fn),
                                  grid_x,
                                  grid_y,
                                  grid_z,
                                  block_x,
                                  block_y,
                                  block_z,
                                  0,  // share memory
                                  static_cast<CUstream>(stream),
                                  kernel_args.data(),
                                  nullptr))
}

void cinn_call_cublas(void *v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      float alpha,
                      float beta,
                      int b0,
                      int b1,
                      int m,
                      int n,
                      int k,
                      void *stream) {
  CHECK_EQ(num_args, 3);
  cublasHandle_t &cuhandle = CublasHandle::GetInstance().GetCublasHandle();
  cinn_pod_value_t *args   = static_cast<cinn_pod_value_t *>(v_args);
  cudaStream_t custream    = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(cuhandle, custream));

  void *A = args[0] cinn_buffer_t * ()->memory;
  void *B = args[1] cinn_buffer_t * ()->memory;
  void *C = args[2] cinn_buffer_t * ()->memory;
  if (b0 == 1 && b1 == 1) {
    if (!trans_a && !trans_b) {
      CUBLAS_CALL(cublasSgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
    } else if (trans_a && !trans_b) {
      CUBLAS_CALL(cublasSgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, B, n, A, m, &beta, C, n));
    } else if (!trans_a && trans_b) {
      CUBLAS_CALL(cublasSgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k, &beta, C, n));
    } else {
      CUBLAS_CALL(cublasSgemm(cuhandle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, B, k, A, m, &beta, C, n));
    }
  } else {
    int stride_a = 0;
    int stride_b = 0;
    if (b0 > 1) {
      stride_a = m * k;
    }
    if (b1 > 1) {
      stride_b = n * k;
    }

    if (!trans_a && !trans_b) {
      CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            n,
                                            m,
                                            k,
                                            &alpha,
                                            B,
                                            n,
                                            stride_b,
                                            A,
                                            k,
                                            stride_a,
                                            &beta,
                                            C,
                                            n,
                                            m * n,
                                            b0 > b1 ? b0 : b1));
    } else if (trans_a && !trans_b) {
      CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_T,
                                            n,
                                            m,
                                            k,
                                            &alpha,
                                            B,
                                            n,
                                            stride_b,
                                            A,
                                            m,
                                            stride_a,
                                            &beta,
                                            C,
                                            n,
                                            m * n,
                                            b0 > b1 ? b0 : b1));
    } else if (!trans_a && trans_b) {
      CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            n,
                                            m,
                                            k,
                                            &alpha,
                                            B,
                                            k,
                                            stride_b,
                                            A,
                                            k,
                                            stride_a,
                                            &beta,
                                            C,
                                            n,
                                            m * n,
                                            b0 > b1 ? b0 : b1));
    } else {
      CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_T,
                                            n,
                                            m,
                                            k,
                                            &alpha,
                                            B,
                                            k,
                                            stride_b,
                                            A,
                                            m,
                                            stride_a,
                                            &beta,
                                            C,
                                            n,
                                            m * n,
                                            b0 > b1 ? b0 : b1));
    }
  }
}

#ifdef CINN_WITH_CUDNN
class CudnnHandle {
 public:
  CudnnHandle(const CudnnHandle &) = delete;
  CudnnHandle &operator=(const CudnnHandle &) = delete;
  ~CudnnHandle() {
    CUDNN_CALL(cudnnDestroy(cuhandle_));
    if (workspace_) {
      CUDA_CALL(cudaFree(workspace_));
    }
  }
  static CudnnHandle &GetInstance() {
    static CudnnHandle instance;
    return instance;
  }
  cudnnHandle_t &GetCudnnHandle() { return cuhandle_; }
  float *GetWorkSpace(size_t size) {
    if (size_ >= size) {
      return workspace_;
    } else {
      if (workspace_) {
        CUDA_CALL(cudaFree(workspace_));
      }
      size_ = size;
      CUDA_CALL(cudaMalloc(&workspace_, size_));
      return workspace_;
    }
  }

 private:
  CudnnHandle() : workspace_(nullptr), size_(0) { CUDNN_CALL(cudnnCreate(&cuhandle_)); }
  cudnnHandle_t cuhandle_;
  float *workspace_;
  size_t size_;
};

class ConvAlgoMap {
 public:
  ConvAlgoMap(const ConvAlgoMap &) = delete;
  ConvAlgoMap &operator=(const ConvAlgoMap &) = delete;
  static ConvAlgoMap &GetInstance() {
    static ConvAlgoMap instance;
    return instance;
  }
  void InsertAlgo(const std::string &key, const int algo) { algo_map_[key] = algo; }
  int GetAlgo(const std::string &key) { return algo_map_.count(key) ? algo_map_[key] : -1; }

 private:
  ConvAlgoMap() {}
  absl::flat_hash_map<std::string, int> algo_map_;
};

void cinn_call_cudnn_conv2d_forward(void *v_args,
                                    int num_args,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int filter_n,
                                    int filter_c,
                                    int filter_h,
                                    int filter_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    int groups,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void *stream) {
  CHECK_EQ(num_args, 3);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<stream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  float *_x              = reinterpret_cast<float *>(args[0] cinn_buffer_t * ()->memory);
  float *_w              = reinterpret_cast<float *>(args[1] cinn_buffer_t * ()->memory);
  float *_y              = reinterpret_cast<float *>(args[2] cinn_buffer_t * ()->memory);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weights_n, weights_c, weights_h, weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  auto &conv_algo_map  = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d forward," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  cudnnConvolutionFwdAlgo_t algo;
  int algo_int = conv_algo_map.GetAlgo(hash_key);
  if (algo_int >= 0) {
    algo = cudnnConvolutionFwdAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    GetInstance.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

  float *workspace_data = CudnnHandle::get_instance().GetWorkSpace(workspace_size);
  CUDNN_CALL(cudnnConvolutionForward(
      handle, &alpha, x_desc, _x, w_desc, _w, conv_desc, algo, workspace_data, workspace_size, &beta, y_desc, _y));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_data(void *v_args,
                                         int num_args,
                                         float alpha,
                                         float beta,
                                         int input_n,
                                         int input_c,
                                         int input_h,
                                         int input_w,
                                         int filter_n,
                                         int filter_c,
                                         int filter_h,
                                         int filter_w,
                                         int pad_h,
                                         int pad_w,
                                         int stride_h,
                                         int stride_w,
                                         int dilation_h,
                                         int dilation_w,
                                         int groups,
                                         int output_n,
                                         int output_c,
                                         int output_h,
                                         int output_w,
                                         void *stream) {
  CHECK_EQ(num_args, 3);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<stream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  float *_dy             = reinterpret_cast<float *>(args[0] cinn_buffer_t * ()->memory);
  float *_w              = reinterpret_cast<float *>(args[1] cinn_buffer_t * ()->memory);
  float *_dx             = reinterpret_cast<float *>(args[2] cinn_buffer_t * ()->memory);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weights_n, weights_c, weights_h, weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  auto &conv_algo_map  = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d backward data," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  int algo_int                       = conv_algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdDataAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc, conv_desc, x_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }

  size_t workspace_size = 0;
  CUDNN_CALL(
      cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, algo, &workspace_size));

  float *workspace_data = CudnnHandle::get_instance().GetWorkSpace(workspace_size);

  CUDNN_CALL(cudnnConvolutionBackwardData(
      handle, alpha, w_desc, _w, y_desc, _dy, conv_desc, &algo, workspace_data, workspace_size, &beta, x_desc, _dx));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_filter(void *v_args,
                                           int num_args,
                                           float alpha,
                                           float beta,
                                           int input_n,
                                           int input_c,
                                           int input_h,
                                           int input_w,
                                           int filter_n,
                                           int filter_c,
                                           int filter_h,
                                           int filter_w,
                                           int pad_h,
                                           int pad_w,
                                           int stride_h,
                                           int stride_w,
                                           int dilation_h,
                                           int dilation_w,
                                           int groups,
                                           int output_n,
                                           int output_c,
                                           int output_h,
                                           int output_w,
                                           void *stream) {
  CHECK_EQ(num_args, 3);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<stream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x  = reinterpret_cast<float *>(args[0] cinn_buffer_t * ()->memory);
  float *_dy = reinterpret_cast<float *>(args[1] cinn_buffer_t * ()->memory);
  float *_dw = reinterpret_cast<float *>(args[2] cinn_buffer_t * ()->memory);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weights_n, weights_c, weights_h, weights_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  absl::flat_hash_map<std::string, int> &algo_map = SerialData::get_instance().GetMap();
  std::string hash_str = "conv2d backward filter," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  if (algo_map.count(hash_str) != 0) {
    algo = cudnnConvolutionBwdFilterAlgo_t(algo_map[hash_str]);
  } else {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));
    algo_map[hash_str] = static_cast<int>(algo_perf.algo);
    algo               = algo_perf.algo;
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }

  size_t workspace_size = 0;
  CUDNN_CALL(
      cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, algo, &workspace_size));

  float *workspace_data = CudnnHandle::get_instance().GetWorkSpace(workspace_size);

  CUDNN_CALL(cudnnConvolutionBackwardFilter(
      handle, &alpha, x_desc, _x, y_desc, _dy, conv_desc, algo, workspace_data, workspace_size, &beta, w_desc, _dw));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_pool2d_forward(void *v_args,
                                    int num_args,
                                    int pool_type,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void *stream) {
  CHECK_EQ(num_args, 2);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<stream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x = reinterpret_cast<float *>(args[0] cinn_buffer_t * ()->memory);
  float *_y = reinterpret_cast<float *>(args[1] cinn_buffer_t * ()->memory);

  cudnnPoolingMode_t pool_mode;
  switch (pool_type) {
    case 0:
      pool_mode = CUDNN_POOLING_MAX;
      break;
    case 1:
      pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    default:
      LOG(FATAL) << "Unkown pool_type: " << pool_type;
  }
  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  float alpha = 1.0f;
  float beta  = 0.0f;

  CUDNN_CALL(cudnnPoolingForward(cudnn, pool_desc, &alpha, x_desc, x, &beta, y_desc, y));

  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
}

void cinn_call_cudnn_pool2d_backward(void *v_args,
                                     int num_args,
                                     int pool_type,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int kernel_h,
                                     int kernel_w,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void *stream) {
  CHECK_EQ(num_args, 3);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<stream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_y  = reinterpret_cast<float *>(args[0] cinn_buffer_t * ()->memory);
  float *_dy = reinterpret_cast<float *>(args[1] cinn_buffer_t * ()->memory);
  float *_dx = reinterpret_cast<float *>(args[2] cinn_buffer_t * ()->memory);

  cudnnPoolingMode_t pool_mode;
  switch (pool_type) {
    case 0:
      pool_mode = CUDNN_POOLING_MAX;
      break;
    case 1:
      pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    default:
      LOG(FATAL) << "Unkown pool_type: " << pool_type;
  }

  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  float alpha = 1.f;
  float beta  = 0.f;

  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, &alpha, y_desc, _y, y_desc, _dy, &beta, x_desc, _dx));

  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
}
#endif

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
