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

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/common/cas.h"
#include "cinn/runtime/cuda/cuda_util.h"

CINN_REGISTER_HELPER(cuda_intrinsics) {
  auto target = cinn::common::DefaultNVGPUTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(cinn_nvgpu_##func__##_fp32, target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(rsqrt);
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
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tanh);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(cinn_nvgpu_##func__##_fp32, target, float, bool);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(isnan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(isinf);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL

  FunctionProto::shape_inference_t inference_shape_globalpool = [](const std::vector<cinn::ir::Expr> &args,
                                                                   int offset) {
    auto t = args[0].as_tensor();
    std::vector<cinn::ir::Expr> shape;
    shape.push_back(t->shape[0]);
    shape.push_back(t->shape[1]);
    shape.push_back(cinn::ir::Expr(1));
    shape.push_back(cinn::ir::Expr(1));
    return shape;
  };

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_warp_reduce_max, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_warp_reduce_sum, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_warp_reduce_avg, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_sum_internal, target)
      .SetRetType<float>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_prod_internal, target)
      .SetRetType<float>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_max_internal, target)
      .SetRetType<float>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_min_internal, target)
      .SetRetType<float>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_all_internal, target)
      .SetRetType<bool>()
      .AddInputType<bool>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_any_internal, target)
      .SetRetType<bool>()
      .AddInputType<bool>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_sum, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_prod, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_max, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_min, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_all, target)
      .SetRetType<bool>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_any, target)
      .SetRetType<bool>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_cuda_find_int, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_cuda_find_float, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_cuda_find_int_from, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_cuda_find_float_from, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_cuda_index_add, target)
      .SetRetType<float>()
      .AddInputType<float>()
      .AddInputType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_sum, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_prod, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_max, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_min, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_all, target)
      .SetRetType<bool>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_any, target)
      .SetRetType<bool>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .End();

  return true;
}

CINN_REGISTER_HELPER(cinn_cuda_host_api) {
  using cinn::runtime::cuda::cinn_call_cuda_kernel;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cuda_kernel, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // kernel_fn
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // grid_x
      .AddInputType<int>()     // grid_y
      .AddInputType<int>()     // grid_z
      .AddInputType<int>()     // block_x
      .AddInputType<int>()     // block_y
      .AddInputType<int>()     // block_z
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cublas;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cublas, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<bool>()    // trans_a
      .AddInputType<bool>()    // trans_b
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // b0
      .AddInputType<int>()     // b1
      .AddInputType<int>()     // m
      .AddInputType<int>()     // n
      .AddInputType<int>()     // k
      .AddInputType<void *>()  // stream
      .End();
#ifdef CINN_WITH_CUDNN
  using cinn::runtime::cuda::cinn_call_cudnn_conv2d_forward;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_conv2d_forward, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // format
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // fn
      .AddInputType<int>()     // fc
      .AddInputType<int>()     // fh
      .AddInputType<int>()     // fw
      .AddInputType<int>()     // ph
      .AddInputType<int>()     // pw
      .AddInputType<int>()     // sh
      .AddInputType<int>()     // sw
      .AddInputType<int>()     // dh
      .AddInputType<int>()     // dw
      .AddInputType<int>()     // g
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_conv2d_backward_data;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_conv2d_backward_data, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // format
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // fn
      .AddInputType<int>()     // fc
      .AddInputType<int>()     // fh
      .AddInputType<int>()     // fw
      .AddInputType<int>()     // ph
      .AddInputType<int>()     // pw
      .AddInputType<int>()     // sh
      .AddInputType<int>()     // sw
      .AddInputType<int>()     // dh
      .AddInputType<int>()     // dw
      .AddInputType<int>()     // g
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_conv2d_backward_filter;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_conv2d_backward_filter, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // format
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // fn
      .AddInputType<int>()     // fc
      .AddInputType<int>()     // fh
      .AddInputType<int>()     // fw
      .AddInputType<int>()     // ph
      .AddInputType<int>()     // pw
      .AddInputType<int>()     // sh
      .AddInputType<int>()     // sw
      .AddInputType<int>()     // dh
      .AddInputType<int>()     // dw
      .AddInputType<int>()     // g
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_pool2d_forward;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_pool2d_forward, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // mode
      .AddInputType<int>()     // format
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // kh
      .AddInputType<int>()     // kw
      .AddInputType<int>()     // ph
      .AddInputType<int>()     // pw
      .AddInputType<int>()     // sh
      .AddInputType<int>()     // sw
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_pool2d_backward;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_pool2d_backward, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // mode
      .AddInputType<int>()     // format
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // kh
      .AddInputType<int>()     // kw
      .AddInputType<int>()     // ph
      .AddInputType<int>()     // pw
      .AddInputType<int>()     // sh
      .AddInputType<int>()     // sw
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_softmax_forward;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_softmax_forward, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // softmax_type
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();

  using cinn::runtime::cuda::cinn_call_cudnn_softmax_backward;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cudnn_softmax_backward, cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // softmax_type
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // in
      .AddInputType<int>()     // ic
      .AddInputType<int>()     // ih
      .AddInputType<int>()     // iw
      .AddInputType<int>()     // on
      .AddInputType<int>()     // oc
      .AddInputType<int>()     // oh
      .AddInputType<int>()     // ow
      .AddInputType<void *>()  // stream
      .End();
#endif

#ifdef CINN_WITH_MKL_CBLAS

#endif

  return true;
}
