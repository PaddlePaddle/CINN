#pragma once
#include <cmath>

#include "cinn/runtime/cinn_runtime.h"

extern "C" {

float cinn_cpu_tanh_fp32(float a);
float cinn_cpu_exp_fp32(float a);
float cinn_cpu_ceil_fp32(float a);
float cinn_cpu_floor_fp32(float a);

//! math extern functions
//@{
float __cinn_host_tanh_fp32(float x);
float __cinn_host_ceil_fp32(float x);
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out);
//@}
}
