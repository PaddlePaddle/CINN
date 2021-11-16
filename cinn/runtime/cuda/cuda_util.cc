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

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"
#include "cinn/runtime/flags.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {

CublasHandle::CublasHandle() { cublasCreate(&cublas); }

CublasHandle::~CublasHandle() { cublasDestroy(cublas); }

class CudnnHelper {
 public:
  static CudnnHelper &Instance() {
    static CudnnHelper cudnn_helper;
    return cudnn_helper;
  }

  cudnnHandle_t GetCudnnHandle() { return handle_; }

  void InsertAlgo(const std::string &key, const int value) {
    std::lock_guard<std::mutex> lock(cudnn_algo_mtx_);
    cudnn_algo_map_[key] = value;
  }

  int GetAlgo(const std::string &key) {
    std::lock_guard<std::mutex> lock(cudnn_algo_mtx_);
    if (cudnn_algo_map_.count(key) == 0) {
      return -1;
    } else {
      return cudnn_algo_map_[key];
    }
  }

  std::shared_ptr<int8_t> GetWorkspace(size_t size) {
    static std::mutex workspace_mtx;
    std::lock_guard<std::mutex> lock(workspace_mtx);
    if (size > workspace_size_) {
      int8_t *ptr     = nullptr;
      workspace_size_ = size;
      CUDA_CALL(cudaMalloc(&ptr, size));
      workspace_ptr_ = std::shared_ptr<int8_t>(ptr, [](int8_t *ptr) { CUDA_CALL(cudaFree(ptr)); });
    }

    return workspace_ptr_;
  }

  ~CudnnHelper() { CUDNN_CALL(cudnnDestroy(handle_)); }

  static std::string GenAlgoKey(const std::string &conv_type, const std::vector<std::vector<int>> &shapes) {
    CHECK_EQ(shapes.size(), 3);
    std::string key = conv_type;
    for (auto &shape : shapes) {
      CHECK_EQ(shape.size(), 4);
      for (auto &value : shape) {
        key += "_" + std::to_string(value);
      }
    }
    return key;
  }

 private:
  CudnnHelper() { CUDNN_CALL(cudnnCreate(&handle_)); }

  cudnnHandle_t handle_{nullptr};
  std::mutex cudnn_algo_mtx_;
  absl::flat_hash_map<std::string, int> cudnn_algo_map_;

  size_t workspace_size_{0};
  std::shared_ptr<int8_t> workspace_ptr_;
};

void CUDART_CB ReleaseWorkspace(void *args) {
  std::shared_ptr<int8_t> *share_ptr = reinterpret_cast<std::shared_ptr<int8_t> *>(args);
  delete share_ptr;
}

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output) {
  cublasHandle_t &cublas = CublasHandle::get_instance().GetCublasHandle();
  float *x_data          = reinterpret_cast<float *>(input1->memory);
  float *y_data          = reinterpret_cast<float *>(input2->memory);
  float *out_data        = reinterpret_cast<float *>(output->memory);
  int M                  = 1;
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
  VLOG(3) << "In cinn_call_cuda_kernel,\ngrid xyz is : " << grid_x << ", " << grid_y << ", " << grid_z;
  VLOG(3) << "block xyz is : " << block_x << ", " << block_y << ", " << block_z;
  VLOG(3) << "num_args is : " << num_args;
  void *arr[20];
  CHECK_LT(num_args, 20);
  for (int i = 0; i < num_args; i++) {
    if (args[i].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
      arr[i] = &((cinn_buffer_t *)(args[i]))->memory;  // NOLINT
    } else {
      arr[i] = args[i].data_addr();
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
                                  reinterpret_cast<void **>(arr),
                                  nullptr))
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

template <class Algo, class FindAlgo, class GetWorkspaceSize, class CallConv>
void CallCudnnConv(const std::vector<int> &input,
                   const std::vector<int> &weight,
                   const std::vector<int> &output,
                   const std::vector<int> &padding,
                   const std::vector<int> &stride,
                   const std::vector<int> &dilation,
                   const int groups,
                   const std::string &conv_type,
                   FindAlgo find_algo_func,
                   GetWorkspaceSize get_workspace_size_func,
                   CallConv call_conv_func) {
  cudnnHandle_t handle = CudnnHelper::Instance().GetCudnnHandle();

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input[0], input[1], input[2], input[3]));

  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight[0], weight[1], weight[2], weight[3]));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc,
                                             padding[0],
                                             padding[1],
                                             stride[0],
                                             stride[1],
                                             dilation[0],
                                             dilation[1],
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output[0], output[1], output[2], output[3]));

  Algo algo = static_cast<Algo>(1);
  if (!GetCinnCudnnDeterministic()) {
    auto algo_key = CudnnHelper::GenAlgoKey(conv_type, {input, weight, output});
    int algo_int  = CudnnHelper::Instance().GetAlgo(algo_key);
    if (algo_int >= 0) {
      algo = static_cast<Algo>(algo_int);
    } else {
      algo = find_algo_func(handle, x_desc, w_desc, conv_desc, y_desc);
      CudnnHelper::Instance().InsertAlgo(algo_key, static_cast<int>(algo));
    }
  }

  size_t ws_size = get_workspace_size_func(handle, x_desc, w_desc, conv_desc, y_desc, algo);
  if (ws_size == 0) {
    call_conv_func(handle, x_desc, w_desc, conv_desc, y_desc, algo, nullptr, 0);
  } else {
    auto args = new std::shared_ptr<int8_t>();
    *args     = CudnnHelper::Instance().GetWorkspace(ws_size);
    call_conv_func(handle, x_desc, w_desc, conv_desc, y_desc, algo, args->get(), ws_size);
    CUDA_CALL(cudaLaunchHostFunc(nullptr, ReleaseWorkspace, args));
  }

  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
}

void cinn_gpu_cudnn_conv2d(const absl::flat_hash_map<std::string, int> &attr,
                           cinn_buffer_t *x,
                           cinn_buffer_t *w,
                           cinn_buffer_t *y) {
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

  float *x_ptr = reinterpret_cast<float *>(x->memory);
  float *w_ptr = reinterpret_cast<float *>(w->memory);
  float *y_ptr = reinterpret_cast<float *>(y->memory);

  auto find_algo_func = [](cudnnHandle_t handle,
                           cudnnTensorDescriptor_t x_desc,
                           cudnnFilterDescriptor_t w_desc,
                           cudnnConvolutionDescriptor_t conv_desc,
                           cudnnTensorDescriptor_t y_desc) {
    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, 1, &count, &algo_perf));
    return algo_perf.algo;
  };

  auto get_workspace_size_func = [](cudnnHandle_t handle,
                                    cudnnTensorDescriptor_t x_desc,
                                    cudnnFilterDescriptor_t w_desc,
                                    cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t y_desc,
                                    cudnnConvolutionFwdAlgo_t algo) {
    size_t ws_size = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size));
    return ws_size;
  };

  auto call_conv_func = [x_ptr, w_ptr, y_ptr](cudnnHandle_t handle,
                                              cudnnTensorDescriptor_t x_desc,
                                              cudnnFilterDescriptor_t w_desc,
                                              cudnnConvolutionDescriptor_t conv_desc,
                                              cudnnTensorDescriptor_t y_desc,
                                              cudnnConvolutionFwdAlgo_t algo,
                                              void *workspace,
                                              size_t ws_size) {
    float alpha[] = {1.f}, beta[] = {0.f};
    CUDNN_CALL(cudnnConvolutionForward(
        handle, alpha, x_desc, x_ptr, w_desc, w_ptr, conv_desc, algo, workspace, ws_size, beta, y_desc, y_ptr));
  };

  CallCudnnConv<cudnnConvolutionFwdAlgo_t>({input_n, input_c, input_h, input_w},
                                           {weights_n, weights_c, weights_h, weights_w},
                                           {output_n, output_c, output_h, output_w},
                                           {pad_h, pad_w},
                                           {stride_h, stride_w},
                                           {dilation_h, dilation_w},
                                           groups,
                                           "conv2d_forward",
                                           find_algo_func,
                                           get_workspace_size_func,
                                           call_conv_func);
}

void cinn_gpu_cudnn_conv2d_backward_data(const absl::flat_hash_map<std::string, int> &attr,
                                         cinn_buffer_t *w,
                                         cinn_buffer_t *dy,
                                         cinn_buffer_t *dx) {
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

  float *w_ptr  = reinterpret_cast<float *>(w->memory);
  float *dy_ptr = reinterpret_cast<float *>(dy->memory);
  float *dx_ptr = reinterpret_cast<float *>(dx->memory);

  auto find_algo_func = [](cudnnHandle_t handle,
                           cudnnTensorDescriptor_t x_desc,
                           cudnnFilterDescriptor_t w_desc,
                           cudnnConvolutionDescriptor_t conv_desc,
                           cudnnTensorDescriptor_t y_desc) {
    int count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc, conv_desc, x_desc, 1, &count, &algo_perf));
    return algo_perf.algo;
  };

  auto get_workspace_size_func = [](cudnnHandle_t handle,
                                    cudnnTensorDescriptor_t x_desc,
                                    cudnnFilterDescriptor_t w_desc,
                                    cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t y_desc,
                                    cudnnConvolutionBwdDataAlgo_t algo) {
    size_t ws_size = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, algo, &ws_size));
    return ws_size;
  };

  auto call_conv_func = [w_ptr, dy_ptr, dx_ptr](cudnnHandle_t handle,
                                                cudnnTensorDescriptor_t x_desc,
                                                cudnnFilterDescriptor_t w_desc,
                                                cudnnConvolutionDescriptor_t conv_desc,
                                                cudnnTensorDescriptor_t y_desc,
                                                cudnnConvolutionBwdDataAlgo_t algo,
                                                void *workspace,
                                                size_t ws_size) {
    float alpha[] = {1.f}, beta[] = {0.f};
    CUDNN_CALL(cudnnConvolutionBackwardData(
        handle, alpha, w_desc, w_ptr, y_desc, dy_ptr, conv_desc, algo, workspace, ws_size, beta, x_desc, dx_ptr));
  };

  CallCudnnConv<cudnnConvolutionBwdDataAlgo_t>({input_n, input_c, input_h, input_w},
                                               {weights_n, weights_c, weights_h, weights_w},
                                               {output_n, output_c, output_h, output_w},
                                               {pad_h, pad_w},
                                               {stride_h, stride_w},
                                               {dilation_h, dilation_w},
                                               groups,
                                               "conv2d_backward_data",
                                               find_algo_func,
                                               get_workspace_size_func,
                                               call_conv_func);
}

void cinn_gpu_cudnn_conv2d_backward_filter(const absl::flat_hash_map<std::string, int> &attr,
                                           cinn_buffer_t *x,
                                           cinn_buffer_t *dy,
                                           cinn_buffer_t *dw) {
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

  float *x_ptr  = reinterpret_cast<float *>(x->memory);
  float *dy_ptr = reinterpret_cast<float *>(dy->memory);
  float *dw_ptr = reinterpret_cast<float *>(dw->memory);

  auto find_algo_func = [](cudnnHandle_t handle,
                           cudnnTensorDescriptor_t x_desc,
                           cudnnFilterDescriptor_t w_desc,
                           cudnnConvolutionDescriptor_t conv_desc,
                           cudnnTensorDescriptor_t y_desc) {
    int count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
    CUDNN_CALL(
        cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, y_desc, conv_desc, w_desc, 1, &count, &algo_perf));
    return algo_perf.algo;
  };

  auto get_workspace_size_func = [](cudnnHandle_t handle,
                                    cudnnTensorDescriptor_t x_desc,
                                    cudnnFilterDescriptor_t w_desc,
                                    cudnnConvolutionDescriptor_t conv_desc,
                                    cudnnTensorDescriptor_t y_desc,
                                    cudnnConvolutionBwdFilterAlgo_t algo) {
    size_t ws_size = 0;
    CUDNN_CALL(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, algo, &ws_size));
    return ws_size;
  };

  auto call_conv_func = [x_ptr, dy_ptr, dw_ptr](cudnnHandle_t handle,
                                                cudnnTensorDescriptor_t x_desc,
                                                cudnnFilterDescriptor_t w_desc,
                                                cudnnConvolutionDescriptor_t conv_desc,
                                                cudnnTensorDescriptor_t y_desc,
                                                cudnnConvolutionBwdFilterAlgo_t algo,
                                                void *workspace,
                                                size_t ws_size) {
    float alpha[] = {1.f}, beta[] = {0.f};
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        handle, alpha, x_desc, x_ptr, y_desc, dy_ptr, conv_desc, algo, workspace, ws_size, beta, w_desc, dw_ptr));
  };

  CallCudnnConv<cudnnConvolutionBwdFilterAlgo_t>({input_n, input_c, input_h, input_w},
                                                 {weights_n, weights_c, weights_h, weights_w},
                                                 {output_n, output_c, output_h, output_w},
                                                 {pad_h, pad_w},
                                                 {stride_h, stride_w},
                                                 {dilation_h, dilation_w},
                                                 groups,
                                                 "conv2d_backward_filter",
                                                 find_algo_func,
                                                 get_workspace_size_func,
                                                 call_conv_func);
}

void cinn_gpu_cudnn_pool2d(const std::vector<int> &attrs,
                           const std::vector<std::string> &str_attrs,
                           cinn_buffer_t *input,
                           cinn_buffer_t *output) {
  cudnnHandle_t handle = CudnnHelper::Instance().GetCudnnHandle();
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

void cinn_gpu_cudnn_softmax(const std::vector<int> &attrs, cinn_buffer_t *input, cinn_buffer_t *output) {
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

  cudnnHandle_t handle = CudnnHelper::Instance().GetCudnnHandle();
  float *in_data       = reinterpret_cast<float *>(input->memory);
  float *out_data      = reinterpret_cast<float *>(output->memory);

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

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
