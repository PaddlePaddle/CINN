/**
 * This file defines some math extern functions thouse can be called in CINN IR.
 */
#pragma once

#include "cinn/runtime/cinn_runtime.h"

extern "C" {

//! 1 buffer as input, and 1 buffer as output
// @{
void cinn_mkl_tanh_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out);
void cinn_mkl_tanh_v_fp64(cinn_buffer_t* x, cinn_buffer_t* out);
void cinn_mkl_exp_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out);
void cinn_mkl_cos_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out);
void cinn_mkl_cos_v_fp64(cinn_buffer_t* x, cinn_buffer_t* out);
// @}
}
