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

#include "cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/runtime/custom_function.h"

#ifdef CINN_WITH_MKL_CBLAS
#include "cinn/runtime/cpu/mkl_math.h"
#endif

extern "C" {

void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out) {
  CINN_CHECK_EQ(x->num_elements(), out->num_elements());
  int xn         = x->num_elements();
  auto* x_data   = (float*)(x->memory);
  auto* out_data = (float*)(out->memory);
  for (int i = 0; i < x->num_elements(); i++) {
    out_data[i] = tanhf(x_data[i]);
  }
}

#define __cinn_host_find_kernel(buf, size, num, type, begin, stride)                   \
  do {                                                                                 \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) {               \
      if (reinterpret_cast<type*>(buf->memory)[i] == num) return (i - begin) / stride; \
    }                                                                                  \
    return -1;                                                                         \
  } while (0)

inline int cinn_host_find_int(const cinn_buffer_t* buf, int size, int num) {
  __cinn_host_find_kernel(buf, size, num, int, 0, 1);
}

inline int cinn_host_find_float(const cinn_buffer_t* buf, int size, float num) {
  __cinn_host_find_kernel(buf, size, num, float, 0, 1);
}

inline int cinn_host_find_int_nd(const cinn_buffer_t* buf, int size, int num, int begin, int stride) {
  __cinn_host_find_kernel(buf, size, num, int, begin, stride);
}

inline int cinn_host_find_float_nd(const cinn_buffer_t* buf, int size, float num, int begin, int stride) {
  __cinn_host_find_kernel(buf, size, num, float, begin, stride);
}

#undef __cinn_host_find_kernel

#define CINN_HOST_LT_NUM(TYPE_SUFFIX, TYPE)                                                           \
  inline int cinn_host_lt_num_##TYPE_SUFFIX(                                                          \
      const cinn_buffer_t* buf, const int size, const TYPE num, const int offset, const int stride) { \
    int out = 0;                                                                                      \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) {                            \
      if (reinterpret_cast<TYPE*>(buf->memory)[i] < num) out++;                                       \
    }                                                                                                 \
    return out;                                                                                       \
  }

CINN_HOST_LT_NUM(fp32, float)
CINN_HOST_LT_NUM(fp64, double)
CINN_HOST_LT_NUM(int32, int)
CINN_HOST_LT_NUM(int64, int64_t)

#undef CINN_HOST_LT_NUM

#define CINN_HOST_GT_NUM(TYPE_SUFFIX, TYPE)                                                           \
  inline int cinn_host_gt_num_##TYPE_SUFFIX(                                                          \
      const cinn_buffer_t* buf, const int size, const TYPE num, const int offset, const int stride) { \
    int out = 0;                                                                                      \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) {                            \
      if (reinterpret_cast<TYPE*>(buf->memory)[i] > num) out++;                                       \
    }                                                                                                 \
    return out;                                                                                       \
  }

CINN_HOST_GT_NUM(fp32, float)
CINN_HOST_GT_NUM(fp64, double)
CINN_HOST_GT_NUM(int32, int)
CINN_HOST_GT_NUM(int64, int64_t)

#undef CINN_HOST_GT_NUM

#define FN_FP32(func) cinn_host_##func##_fp32

inline float FN_FP32(cbrt)(float x) { return cbrt(x); }

inline float FN_FP32(pow)(float x, float y) { return powf(x, y); }

#undef FN_FP32

#define FN_FP64(func) cinn_host_##func##_fp64

inline double FN_FP64(cbrt)(double x) { return cbrt(x); }

inline double FN_FP64(pow)(double x, double y) { return pow(x, y); }

#undef FN_FP64

#define FN_INT32(func) cinn_host_##func##_int32

inline int FN_INT32(pow)(int x, int y) {
  int res = 1;
  for (int i = 0; i < y; ++i) {
    res *= x;
  }
  return res;
}

inline int FN_INT32(clz)(int x) { return __builtin_clz(x); }

inline int FN_INT32(popc)(int x) { return __builtin_popcount(x); }

inline int FN_INT32(logical_right_shift)(int x, int y) { return ((unsigned int)x >> y); }

#undef FN_INT32

#define FN_INT64(func) cinn_host_##func##_int64

inline int64_t FN_INT64(clz)(int64_t x) { return __builtin_clzll(x); }

inline int64_t FN_INT64(popc)(int64_t x) { return __builtin_popcountll(x); }

#undef FN_INT64
}

CINN_REGISTER_HELPER(host_intrinsics) {
  auto host_target = cinn::common::DefaultHostTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(func__) REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(erff);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acosf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acoshf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinhf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanhf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(cinn_host_cbrt_fp32);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, double, double);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64(cinn_host_cbrt_fp64);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32_INT(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, int);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32_INT

#define REGISTER_EXTERN_FUNC_2_IN_1_F(func__) REGISTER_EXTERN_FUNC_2_IN_1_OUT(func__, host_target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_F(powf)

#undef REGISTER_EXTERN_FUNC_2_IN_1_F

#define REGISTER_EXTERN_FUNC_2_IN_1_FP32(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(cinn_host_##func__##_fp32, host_target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_FP32(pow)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP32

#define REGISTER_EXTERN_FUNC_2_IN_1_FP64(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(cinn_host_##func__##_fp64, host_target, double, double, double);

  REGISTER_EXTERN_FUNC_2_IN_1_FP64(pow)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP64

#define REGISTER_EXTERN_FUNC_2_IN_1_INT32(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(cinn_host_##func__##_int32, host_target, int, int, int);

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(pow)

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(logical_right_shift)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT32

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_clz_int32, host_target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_clz_int64, host_target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_popc_int32, host_target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_popc_int64, host_target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_int, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_float, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<float>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_int_nd, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_float_nd, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

#define _REGISTER_CINN_HOST_LT_NUM(TYPE_SUFFIX, TYPE)                             \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_host_lt_num_##TYPE_SUFFIX, host_target) \
      .SetRetType<int>()                                                          \
      .AddInputType<cinn_buffer_t*>()                                             \
      .AddInputType<int>()                                                        \
      .AddInputType<TYPE>()                                                       \
      .AddInputType<int>()                                                        \
      .AddInputType<int>()                                                        \
      .End();

  _REGISTER_CINN_HOST_LT_NUM(fp32, float);
  _REGISTER_CINN_HOST_LT_NUM(fp64, double);
  _REGISTER_CINN_HOST_LT_NUM(int32, int);
  _REGISTER_CINN_HOST_LT_NUM(int64, int64_t);

#undef _REGISTER_CINN_HOST_LT_NUM

#define _REGISTER_CINN_HOST_GT_NUM(TYPE_SUFFIX, TYPE)                             \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_host_gt_num_##TYPE_SUFFIX, host_target) \
      .SetRetType<int>()                                                          \
      .AddInputType<cinn_buffer_t*>()                                             \
      .AddInputType<int>()                                                        \
      .AddInputType<TYPE>()                                                       \
      .AddInputType<int>()                                                        \
      .AddInputType<int>()                                                        \
      .End();

  _REGISTER_CINN_HOST_GT_NUM(fp32, float);
  _REGISTER_CINN_HOST_GT_NUM(fp64, double);
  _REGISTER_CINN_HOST_GT_NUM(int32, int);
  _REGISTER_CINN_HOST_GT_NUM(int64, int64_t);

#undef _REGISTER_CINN_HOST_GT_NUM

  using cinn::runtime::cinn_call_cholesky_host;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cholesky_host, host_target)
      .SetRetType<void>()
      .AddInputType<void*>()  // v_args
      .AddInputType<int>()    // num_args
      .AddInputType<int>()    // batch_size
      .AddInputType<int>()    // m
      .AddInputType<bool>()   // upper
      .End();

  return true;
}
