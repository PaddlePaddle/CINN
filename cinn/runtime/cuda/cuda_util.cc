#include "cinn/runtime/cuda/cuda_util.h"

#include <glog/logging.h>

#include <algorithm>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {

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
  void *arr[20];
  CHECK_LT(num_args, 20);
  for (int i = 0; i < num_args; i++) {
    if (args[i].type_code() == cinn_pod_value_t::type_code<cinn_buffer_t *>()) {
      arr[i] = &((cinn_buffer_t *)args[i])->memory;
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
  // CUDA_CALL(cudaDeviceSynchronize());
}

void cinn_gpu_cudnn_conv2d(int input_n,
                           int input_c,
                           int input_h,
                           int input_w,
                           int weights_n,
                           int weights_c,
                           int weights_h,
                           int weights_w,
                           int pad_h,
                           int pad_w,
                           int stride_h,
                           int stride_w,
                           int dilation_h,
                           int dilation_w,
                           int output_n,
                           int output_c,
                           int output_h,
                           int output_w,
                           cinn_buffer_t *input,
                           cinn_buffer_t *weights,
                           cinn_buffer_t *output) {
  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  float alpha          = 1.f;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  float *in_data = reinterpret_cast<float *>(input->memory);

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weights_n, weights_c, weights_h, weights_w));

  float *filt_data = reinterpret_cast<float *>(weights->memory);

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  float *out_data = reinterpret_cast<float *>(output->memory);

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  ws_data = CudnnHandle::get_instance().GetWorkSpace(ws_size);

  float beta = 0.f;

  CUDNN_CALL(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     in_desc,
                                     in_data,
                                     filt_desc,
                                     filt_data,
                                     conv_desc,
                                     algo,
                                     ws_data,
                                     ws_size,
                                     &beta,
                                     out_desc,
                                     out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
}

void cinn_gpu_cudnn_pool2d(int input_n,
                           int input_c,
                           int input_h,
                           int input_w,
                           const std::string &pool_type,
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
                           cinn_buffer_t *input,
                           cinn_buffer_t *output) {
  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();

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

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
