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

#include <absl/container/flat_hash_map.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
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
  ~CublasHandle() { CUBLAS_CALL(cublasDestroy(cuhandle)); }
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

void FreeMemory(void *data) { CUDA_CALL(cudaFree(static_cast<float **>(data))); }

void cinn_call_cublas(void *v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      bool trans_o,
                      float alpha,
                      float beta,
                      int a1,
                      int a2,
                      int a3,
                      int a4,
                      int b1,
                      int b2,
                      int b3,
                      int b4,
                      void *stream) {
  CHECK_EQ(num_args, 3);
  cublasHandle_t &cuhandle = CublasHandle::GetInstance().GetCublasHandle();
  cinn_pod_value_t *args   = static_cast<cinn_pod_value_t *>(v_args);
  cudaStream_t custream    = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(cuhandle, custream));

  float *A = reinterpret_cast<float *>(args[0].operator cinn_buffer_t *()->memory);
  float *B = reinterpret_cast<float *>(args[1].operator cinn_buffer_t *()->memory);
  float *C = reinterpret_cast<float *>(args[2].operator cinn_buffer_t *()->memory);

  int m = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int n = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int k = trans_a ? a3 : a4;

  cublasOperation_t trans_op_l =
      trans_o ? (trans_a ? CUBLAS_OP_N : CUBLAS_OP_T) : (trans_b ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t trans_op_r =
      trans_o ? (trans_b ? CUBLAS_OP_N : CUBLAS_OP_T) : (trans_a ? CUBLAS_OP_T : CUBLAS_OP_N);
  int ldl = trans_op_l == CUBLAS_OP_N ? m : k;  // trans_o ? (trans_a ? k : m) : (trans_b ? k : m);
  int ldr = trans_op_r == CUBLAS_OP_N ? k : n;  // trans_o ? (trans_b ? n : k) : (trans_a ? n : k);
  int ldc = m;

  float *lhs = trans_o ? A : B;
  float *rhs = trans_o ? B : A;

  if (a1 * a2 * b1 * b2 == 1) {
    CUBLAS_CALL(cublasSgemm(cuhandle, trans_op_l, trans_op_r, m, n, k, &alpha, lhs, ldl, rhs, ldr, &beta, C, ldc));
  } else if (a1 * b1 == 1) {
    CHECK(a2 == b2 || a2 == 1 || b2 == 1);
    int stride_l = trans_o ? (a2 > 1 ? a3 * a4 : 0) : (b2 > 1 ? b3 * b4 : 0);
    int stride_r = trans_o ? (b2 > 1 ? b3 * b4 : 0) : (a2 > 1 ? a3 * a4 : 0);
    int batch    = std::max(a2, b2);
    CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                          trans_op_l,
                                          trans_op_r,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          lhs,
                                          ldl,
                                          stride_l,
                                          rhs,
                                          ldr,
                                          stride_r,
                                          &beta,
                                          C,
                                          ldc,
                                          m * n,
                                          batch));
  } else {
    int l1 = trans_o ? a1 : b1, l2 = trans_o ? a2 : b2, l3 = trans_o ? a3 : b3, l4 = trans_o ? a4 : b4;
    int r1 = trans_o ? b1 : a1, r2 = trans_o ? b2 : a2, r3 = trans_o ? b3 : a3, r4 = trans_o ? b4 : a4;

    if ((l1 == r1 && l2 == r2) || (l1 == 1 && l2 == 1) || (r1 == 1 && r2 == 1)) {
      int stride_l = (l1 == 1 && l2 == 1) ? 0 : l3 * l4;
      int stride_r = (r1 == 1 && r2 == 1) ? 0 : r3 * r4;

      // four types matmul:
      // (N, L) * (N ,L),(N, 1) * (N ,1)
      // (N, L) * (1, 1),(1, 1) * (N, L)
      CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                            trans_op_l,
                                            trans_op_r,
                                            m,
                                            n,
                                            k,
                                            &alpha,
                                            lhs,
                                            ldl,
                                            stride_l,
                                            rhs,
                                            ldr,
                                            stride_r,
                                            &beta,
                                            C,
                                            ldc,
                                            m * n,
                                            std::max(l1, r1) * std::max(l2, r2)));
    } else {
      // (N, L) / (N, 1) / (1, L)
      int bstride_l = (l1 != 1 && l2 != 1) ? (l2 * m * k) : ((l1 != 1) ? m * k : 0);
      // (N, L) / (N, 1) / (1, L)
      int bstride_r = (r1 != 1 && r2 != 1) ? (r2 * k * n) : ((r1 != 1) ? k * n : 0);
      int bstride_c = std::max(l2, r2) * m * n;

      int stride_l = l2 == 1 ? 0 : l3 * l4;
      int stride_r = r2 == 1 ? 0 : r3 * r4;
      // six type matmul:
      // (N, L) * (N, 1),(N, L) * (1, L)
      // (N, 1) * (N, L),(1, L) * (N, L)
      // (N, 1) * (1, L),(1, L) * (N, 1)
      for (int idx = 0; idx < std::max(l1, r1); ++idx) {
        CUBLAS_CALL(cublasSgemmStridedBatched(cuhandle,
                                              trans_op_l,
                                              trans_op_r,
                                              m,
                                              n,
                                              k,
                                              &alpha,
                                              lhs + idx * bstride_l,
                                              ldl,
                                              stride_l,
                                              rhs + idx * bstride_r,
                                              ldr,
                                              stride_r,
                                              &beta,
                                              C + idx * bstride_c,
                                              ldc,
                                              m * n,
                                              std::max(l2, r2)));
      }
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
                                    int format,
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
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  float *_x              = reinterpret_cast<float *>(args[0].operator cinn_buffer_t *()->memory);
  float *_w              = reinterpret_cast<float *>(args[1].operator cinn_buffer_t *()->memory);
  float *_y              = reinterpret_cast<float *>(args[2].operator cinn_buffer_t *()->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, filter_n, filter_c, filter_h, filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, data_type));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  auto &conv_algo_map  = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d forward," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(filter_n) +
                         "," + std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
                         std::to_string(filter_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
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
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

  float *workspace_data = CudnnHandle::GetInstance().GetWorkSpace(workspace_size);
  CUDNN_CALL(cudnnConvolutionForward(
      handle, &alpha, x_desc, _x, w_desc, _w, conv_desc, algo, workspace_data, workspace_size, &beta, y_desc, _y));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_conv2d_backward_data(void *v_args,
                                          int num_args,
                                          int format,
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
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  float *_w              = reinterpret_cast<float *>(args[0].operator cinn_buffer_t *()->memory);
  float *_dy             = reinterpret_cast<float *>(args[1].operator cinn_buffer_t *()->memory);
  float *_dx             = reinterpret_cast<float *>(args[2].operator cinn_buffer_t *()->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, filter_n, filter_c, filter_h, filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, data_type));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  auto &conv_algo_map  = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d backward data," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(filter_n) +
                         "," + std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
                         std::to_string(filter_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  int algo_int = conv_algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdDataAlgo_t algo;
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

  float *workspace_data = CudnnHandle::GetInstance().GetWorkSpace(workspace_size);

  CUDNN_CALL(cudnnConvolutionBackwardData(
      handle, &alpha, w_desc, _w, y_desc, _dy, conv_desc, algo, workspace_data, workspace_size, &beta, x_desc, _dx));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_conv2d_backward_filter(void *v_args,
                                            int num_args,
                                            int format,
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
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x  = reinterpret_cast<float *>(args[0].operator cinn_buffer_t *()->memory);
  float *_dy = reinterpret_cast<float *>(args[1].operator cinn_buffer_t *()->memory);
  float *_dw = reinterpret_cast<float *>(args[2].operator cinn_buffer_t *()->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc, data_type, tensor_format, filter_n, filter_c, filter_h, filter_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, data_type));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  auto &algo_map       = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d backward filter," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(filter_n) +
                         "," + std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
                         std::to_string(filter_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  int algo_int = algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdFilterAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdFilterAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }

  size_t workspace_size = 0;
  CUDNN_CALL(
      cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, algo, &workspace_size));

  float *workspace_data = CudnnHandle::GetInstance().GetWorkSpace(workspace_size);

  CUDNN_CALL(cudnnConvolutionBackwardFilter(
      handle, &alpha, x_desc, _x, y_desc, _dy, conv_desc, algo, workspace_data, workspace_size, &beta, w_desc, _dw));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_pool2d_forward(void *v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
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
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x = reinterpret_cast<float *>(args[0].operator cinn_buffer_t *()->memory);
  float *_y = reinterpret_cast<float *>(args[1].operator cinn_buffer_t *()->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnPoolingMode_t pool_mode      = static_cast<cudnnPoolingMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, &alpha, x_desc, _x, &beta, y_desc, _y));

  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_pool2d_backward(void *v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
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
  CHECK_EQ(num_args, 4);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x  = reinterpret_cast<float *>((args[0].operator cinn_buffer_t *())->memory);
  float *_y  = reinterpret_cast<float *>((args[1].operator cinn_buffer_t *())->memory);
  float *_dy = reinterpret_cast<float *>((args[2].operator cinn_buffer_t *())->memory);
  float *_dx = reinterpret_cast<float *>((args[3].operator cinn_buffer_t *())->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnPoolingMode_t pool_mode      = static_cast<cudnnPoolingMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, &alpha, y_desc, _y, y_desc, _dy, x_desc, _x, &beta, x_desc, _dx));

  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_softmax_forward(void *v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void *stream) {
  CHECK_EQ(num_args, 2);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_x = reinterpret_cast<float *>((args[0].operator cinn_buffer_t *())->memory);
  float *_y = reinterpret_cast<float *>((args[1].operator cinn_buffer_t *())->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnSoftmaxMode_t softmax_mode   = static_cast<cudnnSoftmaxMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_LOG, softmax_mode, &alpha, x_desc, _x, &beta, y_desc, _y));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_call_cudnn_softmax_backward(void *v_args,
                                      int num_args,
                                      int mode,
                                      int format,
                                      float alpha,
                                      float beta,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w,
                                      int output_n,
                                      int output_c,
                                      int output_h,
                                      int output_w,
                                      void *stream) {
  CHECK_EQ(num_args, 3);
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  float *_y  = reinterpret_cast<float *>((args[0].operator cinn_buffer_t *())->memory);
  float *_dy = reinterpret_cast<float *>((args[1].operator cinn_buffer_t *())->memory);
  float *_dx = reinterpret_cast<float *>((args[2].operator cinn_buffer_t *())->memory);

  CHECK_EQ(args[0].operator cinn_buffer_t *()->type.code, cinn_type_code_t::cinn_type_float);
  cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
  cudnnSoftmaxMode_t softmax_mode   = static_cast<cudnnSoftmaxMode_t>(mode);
  cudnnTensorFormat_t tensor_format = static_cast<cudnnTensorFormat_t>(format);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, tensor_format, data_type, output_n, output_c, output_h, output_w));

  CUDNN_CALL(cudnnSoftmaxBackward(
      handle, CUDNN_SOFTMAX_LOG, softmax_mode, &alpha, y_desc, _y, y_desc, _dy, &beta, x_desc, _dx));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

#endif

/********************to be removed in future***********************/

namespace details {

void Gemm(const cublasHandle_t &cublas,
          bool lhs_trans,
          bool rhs_trans,
          const float alpha,
          const float *lhs_data,
          const std::vector<int> &lhs_shape,
          const float *rhs_data,
          const std::vector<int> &rhs_shape,
          const float *bias_data,
          const float beta,
          float *output_data,
          const std::vector<int> &output_shape,
          cudaStream_t stream) {
  int lhs_row    = lhs_shape[0];
  int lhs_col    = lhs_shape[1];
  int rhs_row    = rhs_shape[0];
  int rhs_col    = rhs_shape[1];
  int output_row = output_shape[0];
  int output_col = output_shape[1];

  // copy values of bias_data to the output_data
  if (bias_data != nullptr) {
    cudaMemcpyAsync(output_data, bias_data, output_row * output_col * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }

  int contracting_size = lhs_trans ? lhs_row : lhs_col;
  CHECK_EQ(contracting_size, (rhs_trans ? rhs_col : rhs_row))
      << "The contracting dimension value of lhs matrix should be equal to the one of rhs matrix.";
  auto trans_a = rhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto trans_b = lhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(cublas,
              trans_a,
              trans_b,
              output_col,
              output_row,
              contracting_size,
              &alpha,
              rhs_data,
              rhs_col,
              lhs_data,
              lhs_col,
              &beta,
              output_data,
              output_col);
}

void GemmStridedBatched(const cublasHandle_t &cublas,
                        bool lhs_trans,
                        bool rhs_trans,
                        const float alpha,
                        const float *lhs_data,
                        const std::vector<int> &lhs_shape,
                        const float *rhs_data,
                        const std::vector<int> &rhs_shape,
                        const float *bias_data,
                        const float beta,
                        float *output_data,
                        const std::vector<int> &output_shape,
                        cudaStream_t stream) {
  int lhs_bs     = lhs_shape[0];
  int lhs_row    = lhs_shape[1];
  int lhs_col    = lhs_shape[2];
  int rhs_bs     = rhs_shape[0];
  int rhs_row    = rhs_shape[1];
  int rhs_col    = rhs_shape[2];
  int output_bs  = output_shape[0];
  int output_row = output_shape[1];
  int output_col = output_shape[2];
  CHECK_EQ(lhs_bs, rhs_bs);
  CHECK_EQ(lhs_bs, output_bs);

  // copy values of bias_data to the output_data
  if (bias_data != nullptr) {
    cudaMemcpyAsync(
        output_data, bias_data, output_bs * output_row * output_col * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }

  int contracting_size = lhs_trans ? lhs_row : lhs_col;
  CHECK_EQ(contracting_size, (rhs_trans ? rhs_col : rhs_row))
      << "The contracting dimension value of lhs matrix should be equal to the one of rhs matrix.";
  auto trans_a          = rhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto trans_b          = lhs_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  int64_t lhs_stride    = lhs_row * lhs_col;
  int64_t rhs_stride    = rhs_row * rhs_col;
  int64_t output_stride = output_row * output_col;
  cublasSgemmStridedBatched(cublas,
                            trans_a,
                            trans_b,
                            output_col,
                            output_row,
                            contracting_size,
                            &alpha,
                            rhs_data,
                            rhs_col,
                            rhs_stride,
                            lhs_data,
                            lhs_col,
                            lhs_stride,
                            &beta,
                            output_data,
                            output_col,
                            output_stride,
                            output_bs);
}

}  // namespace details

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output,
                         cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  cudaStream_t custream  = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));
  float *x_data   = reinterpret_cast<float *>(input1->memory);
  float *y_data   = reinterpret_cast<float *>(input2->memory);
  float *out_data = reinterpret_cast<float *>(output->memory);
  int M           = 1;
  CHECK_GE(attrs.size(), 6);
  for (int i = 0; i < attrs[attrs.size() - 2]; i++) {
    M *= attrs[i];
  }
  int N       = attrs[attrs.size() - 3];
  int K       = attrs[attrs.size() - 4];
  float alpha = 1.f;
  float beta  = 0.f;
  // M,N * N,K
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, y_data, K, x_data, N, &beta, out_data, K);
}

void cinn_gpu_cublas_gemm(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  cudaStream_t custream  = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));

  const float *lhs_data  = reinterpret_cast<const float *>(lhs->memory);
  const float *rhs_data  = reinterpret_cast<const float *>(rhs->memory);
  const float *bias_data = bias ? reinterpret_cast<const float *>(bias->memory) : nullptr;
  float *output_data     = reinterpret_cast<float *>(output->memory);

  CHECK_GE(attrs.size(), 13);
  int lhs_dim_size = attrs[attrs.size() - 7];
  int rhs_dim_size = attrs[attrs.size() - 6];
  int out_dim_size = attrs[attrs.size() - 5];
  bool lhs_trans   = static_cast<bool>(attrs[attrs.size() - 4]);
  bool rhs_trans   = static_cast<bool>(attrs[attrs.size() - 3]);
  bool out_trans   = static_cast<bool>(attrs[attrs.size() - 2]);
  // 1）C = A^T * B    -->  C^T = B^T * A
  // 2）C = A * B^T    -->  C^T = B * A^T
  // 3）C = A^T * B^T  -->  C^T = B * A
  // 4）C = A * B      -->  C^T = B^T * A^T
  if (out_trans) {
    lhs_trans = static_cast<bool>(attrs[attrs.size() - 3]) ^ out_trans;
    rhs_trans = static_cast<bool>(attrs[attrs.size() - 4]) ^ out_trans;
  }
  const float alpha = *reinterpret_cast<const float *>(&attrs[attrs.size() - 1]);
  const float beta  = bias ? 1.f : 0.f;
  VLOG(4) << "The lhs_trans value used by cinn_gpu_cublas_gemm: " << lhs_trans;
  VLOG(4) << "The rhs_trans value used by cinn_gpu_cublas_gemm: " << rhs_trans;
  VLOG(4) << "The out_trans value used by cinn_gpu_cublas_gemm: " << out_trans;
  VLOG(4) << "The alpha value used by cinn_gpu_cublas_gemm: " << alpha;
  VLOG(4) << "The beta value used by cinn_gpu_cublas_gemm: " << beta;
  CHECK_EQ(lhs_dim_size, rhs_dim_size);
  CHECK_EQ(lhs_dim_size, out_dim_size);
  CHECK((lhs_dim_size == 2 || lhs_dim_size == 3));

  if (lhs_dim_size == 2) {
    // [row, col]
    std::vector<int> lhs_shape{attrs[0], attrs[1]};
    std::vector<int> rhs_shape{attrs[2], attrs[3]};
    std::vector<int> output_shape{attrs[4], attrs[5]};
    if (out_trans) {
      std::swap(lhs_shape, rhs_shape);
      std::swap(lhs_data, rhs_data);
    }
    details::Gemm(handle,
                  lhs_trans,
                  rhs_trans,
                  alpha,
                  lhs_data,
                  lhs_shape,
                  rhs_data,
                  rhs_shape,
                  bias_data,
                  beta,
                  output_data,
                  output_shape,
                  stream);
  } else {
    // [batch, row, col]
    std::vector<int> lhs_shape{attrs[0], attrs[1], attrs[2]};
    std::vector<int> rhs_shape{attrs[3], attrs[4], attrs[5]};
    std::vector<int> output_shape{attrs[6], attrs[7], attrs[8]};
    if (out_trans) {
      std::swap(lhs_shape, rhs_shape);
      std::swap(lhs_data, rhs_data);
    }
    details::GemmStridedBatched(handle,
                                lhs_trans,
                                rhs_trans,
                                alpha,
                                lhs_data,
                                lhs_shape,
                                rhs_data,
                                rhs_shape,
                                bias_data,
                                beta,
                                output_data,
                                output_shape,
                                stream);
  }
}

#ifdef CINN_WITH_CUDNN

#define GetAttrValue(attr_map, key_name, default_value)      \
  int key_name = 0;                                          \
  if (attr_map.count(#key_name) != 0) {                      \
    key_name = attr_map.find(#key_name)->second;             \
  } else if (default_value >= 0) {                           \
    key_name = default_value;                                \
  } else {                                                   \
    LOG(FATAL) << #key_name << " is not exist in attr_map!"; \
  }

void cinn_gpu_cudnn_conv2d(const absl::flat_hash_map<std::string, int> &attr,
                           cinn_buffer_t *x,
                           cinn_buffer_t *w,
                           cinn_buffer_t *y,
                           cudaStream_t stream,
                           common::Layout target) {
  cudnnTensorFormat_t cudnn_tensor_format;
  if (target == common::Layout::kNCHW) {
    cudnn_tensor_format = CUDNN_TENSOR_NCHW;
  } else if (target == common::Layout::kNHWC) {
    cudnn_tensor_format = CUDNN_TENSOR_NHWC;
  } else {
    CINN_NOT_IMPLEMENTED
  }

  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  float *_x = reinterpret_cast<float *>(x->memory);
  float *_w = reinterpret_cast<float *>(w->memory);
  float *_y = reinterpret_cast<float *>(y->memory);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, cudnn_tensor_format, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

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
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      y_desc, cudnn_tensor_format, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

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
    conv_algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size));

  float *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);
  float alpha[] = {1.f}, beta[] = {0.f};

  CUDNN_CALL(cudnnConvolutionForward(
      handle, alpha, x_desc, _x, w_desc, _w, conv_desc, algo, ws_data, ws_size, beta, y_desc, _y));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_data(const absl::flat_hash_map<std::string, int> &attr,
                                         cinn_buffer_t *w,
                                         cinn_buffer_t *dy,
                                         cinn_buffer_t *dx,
                                         cudaStream_t stream) {
  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  float *_w  = reinterpret_cast<float *>(w->memory);
  float *_dy = reinterpret_cast<float *>(dy->memory);
  float *_dx = reinterpret_cast<float *>(dx->memory);

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

  int algo_int = conv_algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdDataAlgo_t algo;
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

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, algo, &ws_size));

  float *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);

  float alpha[] = {1.0f}, beta[] = {0.0f};
  CUDNN_CALL(cudnnConvolutionBackwardData(
      handle, alpha, w_desc, _w, y_desc, _dy, conv_desc, algo, ws_data, ws_size, beta, x_desc, _dx));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d_backward_filter(const absl::flat_hash_map<std::string, int> &attr,
                                           cinn_buffer_t *x,
                                           cinn_buffer_t *dy,
                                           cinn_buffer_t *dw,
                                           cudaStream_t stream) {
  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));

  float *_x  = reinterpret_cast<float *>(x->memory);
  float *_dy = reinterpret_cast<float *>(dy->memory);
  float *_dw = reinterpret_cast<float *>(dw->memory);

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

  auto &algo_map       = ConvAlgoMap::GetInstance();
  std::string hash_key = "conv2d backward filter," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  int algo_int = algo_map.GetAlgo(hash_key);
  cudnnConvolutionBwdFilterAlgo_t algo;
  if (algo_int >= 0) {
    algo = cudnnConvolutionBwdFilterAlgo_t(algo_int);
  } else {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));

    algo = algo_perf.algo;
    algo_map.InsertAlgo(hash_key, static_cast<int>(algo_perf.algo));
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, algo, &ws_size));

  float *ws_data = CudnnHandle::GetInstance().GetWorkSpace(ws_size);

  float alpha[] = {1.0}, beta[] = {0.0};
  CUDNN_CALL(cudnnConvolutionBackwardFilter(
      handle, alpha, x_desc, _x, y_desc, _dy, conv_desc, algo, ws_data, ws_size, beta, w_desc, _dw));

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_pool2d(const std::vector<int> &attrs,
                           const std::vector<std::string> &str_attrs,
                           cinn_buffer_t *input,
                           cinn_buffer_t *output,
                           cudaStream_t stream) {
  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  CHECK_EQ(attrs.size(), 17);
  // Here the input paddings are pad_top, pad_bottom, pad_left, pad_right.
  // Since pad_top==pad_bottom and pad_left==pad_rifht, we only take pad_top and pad_left.
  int input_n           = attrs[0];
  int input_c           = attrs[1];
  int input_h           = attrs[2];
  int input_w           = attrs[3];
  int kernel_h          = attrs[4];
  int kernel_w          = attrs[5];
  int pad_h             = attrs[6];
  int pad_w             = attrs[8];
  int stride_h          = attrs[10];
  int stride_w          = attrs[11];
  int output_n          = attrs[12];
  int output_c          = attrs[13];
  int output_h          = attrs[14];
  int output_w          = attrs[15];
  int adaptive          = attrs[16];
  std::string pool_type = str_attrs[0];
  cudnnPoolingDescriptor_t pooling_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
  cudnnPoolingMode_t pool_mode;
  if (pool_type == "max") {
    pool_mode = CUDNN_POOLING_MAX;
  } else if (pool_type == "avg") {
    pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
  }
  if (adaptive == 1) {
    stride_h = input_h / output_h;
    stride_w = input_w / output_w;
    kernel_h = input_h - (output_h - 1) * stride_h;
    kernel_w = input_w - (output_w - 1) * stride_w;
  }

  CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pooling_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

  cudnnTensorDescriptor_t in_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));

  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  cudnnTensorDescriptor_t out_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  float alpha = 1.0f;
  float beta  = 0.0f;

  float *in_data  = reinterpret_cast<float *>(input->memory);
  float *out_data = reinterpret_cast<float *>(output->memory);

  CUDNN_CALL(cudnnPoolingForward(handle, pooling_desc, &alpha, in_desc, in_data, &beta, out_desc, out_data));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
}

void cinn_gpu_cudnn_softmax(const std::vector<int> &attrs,
                            cinn_buffer_t *input,
                            cinn_buffer_t *output,
                            cudaStream_t stream) {
  std::vector<int> shape;
  int rank = attrs.size() - 1;
  for (int i = 0; i < rank; i++) {
    shape.push_back(attrs[i]);
  }
  int axis      = attrs.back();
  axis          = axis < 0 ? rank + axis : axis;
  int inner_num = 1;
  int outer_num = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (i < axis)
      outer_num *= shape[i];
    else if (i > axis)
      inner_num *= shape[i];
  }
  rank = shape.size();

  cudnnHandle_t &handle = CudnnHandle::GetInstance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, static_cast<cudaStream_t>(stream)));
  float *in_data  = reinterpret_cast<float *>(input->memory);
  float *out_data = reinterpret_cast<float *>(output->memory);

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outer_num, shape[axis], inner_num, 1));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outer_num, shape[axis], inner_num, 1));

  float alpha = 1.f;
  float beta  = 0.f;

  CUDNN_CALL(cudnnSoftmaxForward(
      handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in_desc, in_data, &beta, out_desc, out_data));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
}

#endif

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
