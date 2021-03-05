#include <cudnn.h>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/common/cas.h"
#include "cinn/runtime/cuda/cuda_util.h"

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
                           void *input,
                           void *weights,
                           void *output) {
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
}

CINN_REGISTER_HELPER(cuda_intrinsics) {
  auto target = cinn::common::DefaultNVGPUTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(cinn_nvgpu_##func__##_fp32, target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log2);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log10);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(floor);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(ceil);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(round);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(trunc);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(isnan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(isinf);

  return true;
}

CINN_REGISTER_HELPER(cinn_call_cuda_kernel) {
  using cinn::runtime::cuda::cinn_call_cuda_kernel;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cuda_kernel, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()              // kernel_fn
      .AddInputType<cinn_pod_value_t *>()  // args
      .AddInputType<int>()                 // num_args
      .AddInputType<int>()                 // grid_x
      .AddInputType<int>()                 // grid_y
      .AddInputType<int>()                 // grid_z
      .AddInputType<int>()                 // block_x
      .AddInputType<int>()                 // block_y
      .AddInputType<int>()                 // block_z
      .AddInputType<void *>()              // stream
      .End();

  return true;
}

CINN_REGISTER_HELPER(cinn_gpu_cudnn_conv2d) {
  // using cinn::runtime::cuda::cinn_gpu_cudnn_conv2d;
  using cinn::backends::FunctionProto;
  using cinn::ir::Expr;
  FunctionProto::shape_inference_t inference_shape_cudnn = [](const std::vector<Expr> &args, int offset) {
    CHECK_EQ(offset, 0UL) << "Only one output";
    auto N = cinn::common::AutoSimplify(args[14]);
    auto C = cinn::common::AutoSimplify(args[15]);
    auto H = cinn::common::AutoSimplify(args[16]);
    auto W = cinn::common::AutoSimplify(args[17]);
    std::vector<Expr> shape;
    shape.push_back(N);
    shape.push_back(C);
    shape.push_back(H);
    shape.push_back(W);
    return shape;
  };

  REGISTER_EXTERN_FUNC_HELPER(cinn_gpu_cudnn_conv2d, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<void *>()
      .AddInputType<void *>()
      .AddOutputType<void *>()
      .SetShapeInference(inference_shape_cudnn)
      .End();

  return true;
}