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

#define __cinn_host_find_kernel(buf, size, num, type)               \
  do {                                                              \
    for (int i = size - 1; i >= 0; --i) {                           \
      if (reinterpret_cast<type*>(buf->memory)[i] == num) return i; \
    }                                                               \
    return -1;                                                      \
  } while (0)

inline int cinn_host_find_int(const cinn_buffer_t* buf, int size, int num) {
  __cinn_host_find_kernel(buf, size, num, int);
}

inline int cinn_host_find_float(const cinn_buffer_t* buf, int size, float num) {
  __cinn_host_find_kernel(buf, size, num, float);
}

#undef __cinn_host_find_kernel

#define __cinn_host_lt_num_kernel(buf, size, num, offset, stride, type) \
  do {                                                                  \
    int out = 0;                                                        \
    for (int i = size - 1; i >= 0; --i) {                               \
      if (num < reinterpret_cast<type*>(buf->memory)[i]) out++;         \
    }                                                                   \
    return out;                                                         \
  } while (0)

inline int cinn_host_lt_num_float(
    const cinn_buffer_t* buf, const float size, const float num, const float offset, const float stride) {
  __cinn_host_lt_num_kernel(buf, size, num, offset, stride, float);
}

inline int cinn_host_lt_num_int(
    const cinn_buffer_t* buf, const float size, const int num, const float offset, const float stride) {
  __cinn_host_lt_num_kernel(buf, size, num, offset, stride, int);
}

#undef __cinn_host_find_kernel
}

CINN_REGISTER_HELPER(host_intrinsics) {
  auto host_target = cinn::common::DefaultHostTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(func__) REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, float);

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32_INT(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, int);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(erff);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acosf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acoshf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinhf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanhf);

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

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_lt_num_int, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_lt_num_float, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  return true;
}
