#include "cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/runtime/cpu/host_intrinsics.h"
#include "mkl_math.h"

extern "C" {

using namespace std;
#define CINN_IMP_CPU_FUNC_FP32(name__) \
  float cinn_cpu_##name__##_fp32(float a) { return name__(a); }

#define CINN_IMP_CPU_FUNC_INT_BINARY(name__, rule__) \
  int cinn_cpu_##name__##_int32(int a, int b) { return a rule__ b; }

#define CINN_IMP_CPU_FUNC_INT_UNARY(name__, rule__) \
  int cinn_cpu_##name__##_int32(int a) { return rule__(a); }

CINN_IMP_CPU_FUNC_FP32(exp);
CINN_IMP_CPU_FUNC_FP32(erf);
CINN_IMP_CPU_FUNC_FP32(sqrt);
CINN_IMP_CPU_FUNC_FP32(log);
CINN_IMP_CPU_FUNC_FP32(log2);
CINN_IMP_CPU_FUNC_FP32(log10);
CINN_IMP_CPU_FUNC_FP32(floor);
CINN_IMP_CPU_FUNC_FP32(ceil);
CINN_IMP_CPU_FUNC_FP32(round);
CINN_IMP_CPU_FUNC_FP32(trunc);
CINN_IMP_CPU_FUNC_FP32(cos);
CINN_IMP_CPU_FUNC_FP32(cosh);
CINN_IMP_CPU_FUNC_FP32(tan);
CINN_IMP_CPU_FUNC_FP32(sin);
CINN_IMP_CPU_FUNC_FP32(sinh);
CINN_IMP_CPU_FUNC_FP32(acos);
CINN_IMP_CPU_FUNC_FP32(acosh);
CINN_IMP_CPU_FUNC_FP32(asin);
CINN_IMP_CPU_FUNC_FP32(asinh);
CINN_IMP_CPU_FUNC_FP32(atan);
CINN_IMP_CPU_FUNC_FP32(atanh);
CINN_IMP_CPU_FUNC_FP32(isnan);
CINN_IMP_CPU_FUNC_FP32(tanh);
CINN_IMP_CPU_FUNC_FP32(isfinite);
CINN_IMP_CPU_FUNC_FP32(isinf);

CINN_IMP_CPU_FUNC_INT_BINARY(left_shift, <<);
CINN_IMP_CPU_FUNC_INT_BINARY(right_shift, >>);
CINN_IMP_CPU_FUNC_INT_BINARY(bitwise_or, |);
CINN_IMP_CPU_FUNC_INT_BINARY(bitwise_and, &);
CINN_IMP_CPU_FUNC_INT_BINARY(bitwise_xor, ^);
CINN_IMP_CPU_FUNC_INT_UNARY(bitwise_not, !);

float __cinn_host_tanh_fp32(float x) { return std::tanh(x); }
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out) {
  CINN_CHECK_EQ(x->num_elements(), out->num_elements());
  int xn         = x->num_elements();
  auto* x_data   = (float*)(x->memory);
  auto* out_data = (float*)(out->memory);
  for (int i = 0; i < x->num_elements(); i++) {
    out_data[i] = __cinn_host_tanh_fp32(x_data[i]);
  }
}

float __cinn_host_ceil_fp32(float x) { return std::ceil(x); }
}

REGISTER_EXTERN_FUNC(host_intrinsics) {
  auto host_target = cinn::common::DefaultHostTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(func__) \
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_##func__##_fp32, host_target, float, float);

#define REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_INT(func__) \
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_##func__##_int32, host_target, int, int);

#define REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(func__) \
  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT(cinn_cpu_##func__##_int32, host_target, int, int, int);

  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(exp);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(erf);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(sqrt);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(log);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(log2);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(log10);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(floor);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(ceil);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(round);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(trunc);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(cos);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(cosh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(tan);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(sin);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(sinh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(acos);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(acosh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(asin);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(asinh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(atan);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(atanh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(isnan);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(tanh);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(isfinite);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_FLOAT(isinf);

  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(left_shift);
  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(right_shift);
  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(bitwise_or);
  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(bitwise_and);
  REGISTER_EXTERN_FUNC_TWO_IN_ONE_OUT_INT(bitwise_xor);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT_INT(bitwise_not);

  REGISTER_EXTERN_FUNC_HELPER(__cinn_host_ceil_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .AddOutputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_exp_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();
}
