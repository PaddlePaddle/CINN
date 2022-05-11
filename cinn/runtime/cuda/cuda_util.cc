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
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"
#include "cinn/runtime/flags.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {
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
          const cudaStream_t &stream) {
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
                        const cudaStream_t &stream) {
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

CublasHandle::CublasHandle() { cublasCreate(&cublas); }

CublasHandle::~CublasHandle() { cublasDestroy(cublas); }

SerialData::SerialData() {}

SerialData::~SerialData() {}

void cinn_call_cuda_kernel(void *kernel_fn,
                           cinn_pod_value_t *args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void *stream) {
  // prepare void**
  VLOG(3) << "In cinn_call_cuda_kernel, grid_dim={" << grid_x << ", " << grid_y << ", " << grid_z << "}, block_dim={"
          << block_x << ", " << block_y << ", " << block_z << "}, num_args=" << num_args << ", stream=" << stream;
  std::vector<void *> arr;
  for (int i = 0; i < num_args; i++) {
    if (args[i].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
      arr.emplace_back(&((cinn_buffer_t *)(args[i]))->memory);  // NOLINT
    } else {
      arr.emplace_back(args[i].data_addr());
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
                                  reinterpret_cast<void **>(arr.data()),
                                  nullptr))
}

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output,
                         const cudaStream_t &stream) {
  cublasHandle_t &cublas = CublasHandle::get_instance().GetCublasHandle();
  cublasSetStream(cublas, stream);
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
  cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, y_data, K, x_data, N, &beta, out_data, K);
}

void cinn_gpu_cublas_gemm(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          const cudaStream_t &stream) {
  cublasHandle_t &cublas = CublasHandle::get_instance().GetCublasHandle();
  cublasSetStream(cublas, stream);

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
    details::Gemm(cublas,
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
    details::GemmStridedBatched(cublas,
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
CudnnHandle::CudnnHandle() {
  CUDNN_CALL(cudnnCreate(&cudnn));
  size_      = 0;
  work_space = nullptr;
}

CudnnHandle::~CudnnHandle() {
  CUDNN_CALL(cudnnDestroy(cudnn));
  if (size_ > 0) {
    CUDA_CALL(cudaFree(work_space));
  }
}

float *CudnnHandle::GetWorkSpace(size_t size) {
  if (size_ >= size) {
    return work_space;
  } else {
    if (size_ > 0) {
      CUDA_CALL(cudaFree(work_space));
    }
    CUDA_CALL(cudaMalloc(&work_space, size));
    size_ = size;
    return work_space;
  }
}

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
                           const cudaStream_t &stream) {
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

  cudnnHandle_t &handle = CudnnHandle::get_instance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, stream));
  float *_x = reinterpret_cast<float *>(x->memory);
  float *_w = reinterpret_cast<float *>(w->memory);
  float *_y = reinterpret_cast<float *>(y->memory);

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
  std::string hash_str = "conv2d forward," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  cudnnConvolutionFwdAlgo_t algo;
  if (algo_map.count(hash_str) != 0) {
    algo = cudnnConvolutionFwdAlgo_t(algo_map[hash_str]);
  } else {
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int count = 0;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, 1, &count, &algo_perf));
    algo_map[hash_str] = static_cast<int>(algo_perf.algo);
    algo               = algo_perf.algo;
  }

  if (GetCinnCudnnDeterministic()) {
    algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size));

  float *ws_data = CudnnHandle::get_instance().GetWorkSpace(ws_size);

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
                                         const cudaStream_t &stream) {
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

  cudnnHandle_t &handle = CudnnHandle::get_instance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, stream));
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

  absl::flat_hash_map<std::string, int> &algo_map = SerialData::get_instance().GetMap();
  std::string hash_str = "conv2d backward data," + std::to_string(input_n) + "," + std::to_string(input_c) + "," +
                         std::to_string(input_h) + "," + std::to_string(input_w) + "," + std::to_string(weights_n) +
                         "," + std::to_string(weights_c) + "," + std::to_string(weights_h) + "," +
                         std::to_string(weights_w) + "," + std::to_string(output_n) + "," + std::to_string(output_c) +
                         "," + std::to_string(output_h) + "," + std::to_string(output_w);

  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  if (algo_map.count(hash_str) != 0) {
    algo = cudnnConvolutionBwdDataAlgo_t(algo_map[hash_str]);
  } else {
    int count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc, conv_desc, x_desc, 1, &count, &algo_perf));
    algo_map[hash_str] = static_cast<int>(algo_perf.algo);
    algo               = algo_perf.algo;
  }

  if (GetCinnCudnnDeterministic()) {
    algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, algo, &ws_size));

  float *ws_data = CudnnHandle::get_instance().GetWorkSpace(ws_size);

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
                                           const cudaStream_t &stream) {
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

  cudnnHandle_t &handle = CudnnHandle::get_instance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(handle, stream));

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

  size_t ws_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, algo, &ws_size));

  float *ws_data = ws_data = CudnnHandle::get_instance().GetWorkSpace(ws_size);

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
                           const cudaStream_t &stream) {
  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(cudnn, stream));
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

  CUDNN_CALL(cudnnPoolingForward(cudnn, pooling_desc, &alpha, in_desc, in_data, &beta, out_desc, out_data));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
}

void cinn_gpu_cudnn_softmax(const std::vector<int> &attrs,
                            cinn_buffer_t *input,
                            cinn_buffer_t *output,
                            const cudaStream_t &stream) {
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

  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  CUDNN_CALL(cudnnSetStream(cudnn, stream));
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
      cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, in_desc, in_data, &beta, out_desc, out_data));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
}
#endif

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
