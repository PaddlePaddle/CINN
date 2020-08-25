#pragma once
/**
 * \file This file implements some intrinsic functions for math operation in host device.
 */
#include "cinn/runtime/cinn_runtime.h"

extern "C" {

#define CINN_DCL_CPU_FUNC_FP32(name__) float cinn_cpu_##name__##_fp32(float a);
#define CINN_DCL_CPU_FUNC_INT_UNARY(name__) int cinn_cpu_##name__##_int(int a);
#define CINN_DCL_CPU_FUNC_INT_BINARY(name__) int cinn_cpu_##name__##_int(int a, int b);

CINN_DCL_CPU_FUNC_FP32(exp);
CINN_DCL_CPU_FUNC_FP32(erf);
CINN_DCL_CPU_FUNC_FP32(sqrt);
CINN_DCL_CPU_FUNC_FP32(log);
CINN_DCL_CPU_FUNC_FP32(log2);
CINN_DCL_CPU_FUNC_FP32(log10);
CINN_DCL_CPU_FUNC_FP32(floor);
CINN_DCL_CPU_FUNC_FP32(ceil);
CINN_DCL_CPU_FUNC_FP32(round);
CINN_DCL_CPU_FUNC_FP32(trunc);
CINN_DCL_CPU_FUNC_FP32(cos);
CINN_DCL_CPU_FUNC_FP32(cosh);
CINN_DCL_CPU_FUNC_FP32(tan);
CINN_DCL_CPU_FUNC_FP32(sin);
CINN_DCL_CPU_FUNC_FP32(sinh);
CINN_DCL_CPU_FUNC_FP32(acos);
CINN_DCL_CPU_FUNC_FP32(acosh);
CINN_DCL_CPU_FUNC_FP32(asin);
CINN_DCL_CPU_FUNC_FP32(asinh);
CINN_DCL_CPU_FUNC_FP32(atan);
CINN_DCL_CPU_FUNC_FP32(atanh);
CINN_DCL_CPU_FUNC_FP32(isnan);
CINN_DCL_CPU_FUNC_FP32(tanh);
CINN_DCL_CPU_FUNC_FP32(isfinite);
CINN_DCL_CPU_FUNC_FP32(isinf);

CINN_DCL_CPU_FUNC_INT_BINARY(left_shift);
CINN_DCL_CPU_FUNC_INT_BINARY(right_shift);
CINN_DCL_CPU_FUNC_INT_BINARY(bitwise_or);
CINN_DCL_CPU_FUNC_INT_BINARY(bitwise_and);
CINN_DCL_CPU_FUNC_INT_BINARY(bitwise_xor);
CINN_DCL_CPU_FUNC_INT_UNARY(bitwise_not);

//! math extern functions
//@{
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out);
//@}
}
