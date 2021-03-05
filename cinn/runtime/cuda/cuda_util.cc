#include "cinn/runtime/cuda/cuda_util.h"

#include <cudnn.h>
#include <glog/logging.h>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {

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
  LOG(INFO) << "Begin test cinn_call_cuda_kernel";
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
  LOG(INFO) << "Begin cinn_gpu_cudnn_conv2d";
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  float alpha = 1.f;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

  float *in_data1 = reinterpret_cast<float *>(input->memory);
  float *in_data;
  CUDA_CALL(cudaMalloc(&in_data, input_n * input_c * input_h * input_w * sizeof(float)));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
      filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weights_n, weights_c, weights_h, weights_w));

  float *filt_data1 = reinterpret_cast<float *>(weights->memory);
  float *filt_data;
  CUDA_CALL(cudaMalloc(&filt_data, weights_n * weights_c * weights_h * weights_w * sizeof(float)));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));
  LOG(INFO) << "output shape is : " << out_n << " " << out_c << " " << out_h << " " << out_w;
  CHECK_EQ(out_n, output_n);
  CHECK_EQ(out_c, output_c);
  CHECK_EQ(out_h, output_h);
  CHECK_EQ(out_w, output_w);

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

  float *out_data1 = reinterpret_cast<float *>(output->memory);
  float *out_data;
  CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  cudnnConvolutionFwdAlgoPerf_t perf_algo;
  int returnedAlgoCount;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      cudnn, in_desc, filt_desc, conv_desc, out_desc, 1, &returnedAlgoCount, &perf_algo));
  cudnnConvolutionFwdAlgo_t algo = perf_algo.algo;

  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
  LOG(INFO) << "ws_size is : " << ws_size;

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  float beta = 0.f;
  // out_data or output?
  CUDA_CALL(
      cudaMemcpy(in_data, in_data1, input_n * input_c * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(
      filt_data, filt_data1, weights_n * weights_c * weights_h * weights_w * sizeof(float), cudaMemcpyHostToDevice));
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
  CUDA_CALL(cudaMemcpy(
      out_data1, out_data, output_n * output_c * output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost));
  LOG(INFO) << "end of this function";
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
