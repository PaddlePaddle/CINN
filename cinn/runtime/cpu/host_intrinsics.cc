#include "cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/runtime/cpu/mkl_math.h"

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

  return true;
}
