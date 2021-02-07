#include "cinn/runtime/cpu/mkl_math.h"

#include <glog/logging.h>
#include <mkl.h>
#include <mkl_vml_functions.h>

#include <cmath>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

void cinn_mkl_tanh_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vsTanh(x->num_elements(), reinterpret_cast<float *>(x->memory), reinterpret_cast<float *>(out->memory));
}
void cinn_mkl_tanh_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdTanh(x->num_elements(), reinterpret_cast<double *>(x->memory), reinterpret_cast<double *>(out->memory));
}
void cinn_mkl_exp_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vsExp(x->num_elements(), reinterpret_cast<float *>(x->memory), reinterpret_cast<float *>(out->memory));
}
void cinn_mkl_exp_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdExp(x->num_elements(), reinterpret_cast<double *>(x->memory), reinterpret_cast<double *>(out->memory));
}

/*
void cinn_mkl_cos_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vsCosh(x->num_elements(), reinterpret_cast<float *>(x->memory), reinterpret_cast<float *>(out->memory));
}
void cinn_mkl_cos_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) {
  CHECK_EQ(x->num_elements(), out->num_elements());
  vdCosh(x->num_elements(), reinterpret_cast<double *>(x->memory), reinterpret_cast<double *>(out->memory));
}
*/

CINN_REGISTER_HELPER(mkl_math) {
  using cinn::backends::FunctionProto;

  auto host_target = cinn::common::DefaultHostTarget();

  REGISTER_EXTERN_FUNC_HELPER(cinn_mkl_tanh_v_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_mkl_tanh_v_fp64, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_mkl_exp_v_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  return true;
}
