/**
 * This file defines some math extern functions thouse can be called in CINN IR.
 */
#pragma once

#include "cinn/runtime/cinn_runtime.h"

extern "C" {
void cinn_mkl_tanh_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out);
void cinn_mkl_tanh_v_fp64(cinn_buffer_t* x, cinn_buffer_t* out);

void cinn_mkl_exp_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out);
}

namespace cinn {
namespace hlir {
namespace instruction {
namespace x86 {}  // namespace x86
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
