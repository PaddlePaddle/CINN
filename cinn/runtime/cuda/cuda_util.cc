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

SerialData::SerialData() {}

SerialData::~SerialData() {}

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

CublasHandle::CublasHandle() { cublasCreate(&cublas); }

CublasHandle::~CublasHandle() { cublasDestroy(cublas); }

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

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output) {
  cublasHandle_t &cublas = CublasHandle::get_instance().GetCublasHandle();
  float *x_data          = reinterpret_cast<float *>(input1->memory);
  float *y_data          = reinterpret_cast<float *>(input2->memory);
  float *out_data        = reinterpret_cast<float *>(output->memory);
  int M                  = 1;
  CHECK(attrs.size() >= 6);
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
  VLOG(3) << "In cinn_call_cuda_kernel grid xyz is : " << grid_x << ", " << grid_y << ", " << grid_z;
  VLOG(3) << "block xyz is : " << block_x << ", " << block_y << ", " << block_z;
  VLOG(3) << "num_args is : " << num_args;
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
}

void cinn_gpu_cudnn_conv2d(const std::vector<int> &attrs,
                           cinn_buffer_t *input,
                           cinn_buffer_t *weights,
                           cinn_buffer_t *output) {
  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  float alpha          = 1.f;
  CHECK_EQ(attrs.size(), 19);
  int input_n    = attrs[0];
  int input_c    = attrs[1];
  int input_h    = attrs[2];
  int input_w    = attrs[3];
  int weights_n  = attrs[4];
  int weights_c  = attrs[5];
  int weights_h  = attrs[6];
  int weights_w  = attrs[7];
  int pad_h      = attrs[8];
  int pad_w      = attrs[9];
  int stride_h   = attrs[10];
  int stride_w   = attrs[11];
  int dilation_h = attrs[12];
  int dilation_w = attrs[13];
  int groups     = attrs[14];
  int output_n   = attrs[15];
  int output_c   = attrs[16];
  int output_h   = attrs[17];
  int output_w   = attrs[18];
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
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w));

  float *out_data = reinterpret_cast<float *>(output->memory);

  std::unordered_map<std::string, int> &algo_map = SerialData::get_instance().GetMap();

  std::string hash_str = std::to_string(input_n) + "," + std::to_string(input_c) + "," + std::to_string(input_h) + "," +
                         std::to_string(input_w) + "," + std::to_string(weights_n) + "," + std::to_string(weights_c) +
                         "," + std::to_string(weights_h) + "," + std::to_string(weights_w) + "," +
                         std::to_string(output_n) + "," + std::to_string(output_c) + "," + std::to_string(output_h) +
                         "," + std::to_string(output_w);

  cudnnConvolutionFwdAlgo_t algo;
  if (algo_map.count(hash_str) != 0) {
    algo = cudnnConvolutionFwdAlgo_t(algo_map[hash_str]);
  } else {
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int count;
    cudnnFindConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, 1, &count, &algo_perf);
    algo_map[hash_str] = int(algo_perf.algo);
    algo               = algo_perf.algo;
  }
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

void cinn_gpu_cudnn_pool2d(const std::vector<int> &attrs,
                           const std::vector<std::string> &str_attrs,
                           cinn_buffer_t *input,
                           cinn_buffer_t *output) {
  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  CHECK_EQ(attrs.size(), 16);
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

void cinn_gpu_cudnn_softmax(const std::vector<int> &attrs, cinn_buffer_t *input, cinn_buffer_t *output) {
  std::vector<int> shape;
  int rank = attrs.size() - 1;
  for (int i = 0; i < rank; i++) {
    shape.push_back(attrs[i]);
  }
  int axis = attrs.back();
  axis     = axis < 0 ? rank + axis : axis;
  if (shape.size() <= 2) {
    shape.resize(4, 1);
  }
  rank = shape.size();

  cudnnHandle_t &cudnn = CudnnHandle::get_instance().GetCudnnHandle();
  float *in_data       = reinterpret_cast<float *>(input->memory);
  float *out_data      = reinterpret_cast<float *>(output->memory);

  cudnnSoftmaxMode_t mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL;
  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2], shape[3]));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2], shape[3]));

  float alpha = 1.f;
  float beta  = 0.f;

  CUDNN_CALL(
      cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, mode, &alpha, in_desc, in_data, &beta, out_desc, out_data));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
