#pragma once
#include "cinn/backends/extern_func_jit_register.h"

#ifdef CINN_WITH_CUDA
CINN_USE_REGISTER(cinn_call_cuda_kernel)
CINN_USE_REGISTER(cinn_gpu_cudnn_conv2d)
CINN_USE_REGISTER(cuda_intrinsics)
#endif
