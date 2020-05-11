#pragma once
/**
 * \file This file implements some intrinsic functions for math operation in host device.
 */
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace cpu {

/**
 * \brief Do Tanh on a single float value.
 * @param args (float x, float* out)
 * @param nargs should be 2
 */
void __cpu_tanh(cinn_pod_value_t* args, int nargs);

}  // namespace cpu
}  // namespace runtime
}  // namespace cinn
