#pragma once

#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace cuda {

const int kCUDAMaxCards{10};

/**
 * Call a CUDA compiled kernel.
 *
 * @param kernel_fn the compiled PTX kernel.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_cuda_kernel(void* kernel_fn,
                           cinn_pod_value_t* args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream);

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
