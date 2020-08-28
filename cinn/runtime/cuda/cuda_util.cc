#include "cinn/runtime/cuda/cuda_util.h"

#include <glog/logging.h>

#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/common/target.h"

namespace cinn {
namespace runtime {
namespace cuda {

void cinn_call_cuda_kernel(void *kernel_fn,
                           cinn_pod_value_t *args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void *stream) {
  // prepare void**
  void *arr[20];
  CHECK_LT(num_args, 20);
  for (int i = 0; i < num_args; i++) {
    if (args[i].type_code() == cinn_pod_value_t::type_code<cinn_buffer_t *>()) {
      arr[i] = &((cinn_buffer_t *)args[i])->memory;
    } else {
      arr[i] = args[i].data_addr();
    }
  }

  CUDA_DRIVER_CALL(cuLaunchKernel(static_cast<CUfunction>(kernel_fn),
                                  grid_x,
                                  grid_y,
                                  grid_z,
                                  block_x,
                                  block_y,
                                  block_z,
                                  0,  // share memory
                                  static_cast<CUstream>(stream),
                                  reinterpret_cast<void **>(arr),
                                  nullptr))

  CUDA_CALL(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn

