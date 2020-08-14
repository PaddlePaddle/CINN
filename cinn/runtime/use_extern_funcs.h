#pragma once

#include "cinn/backends/extern_func_jit_register.h"

#ifdef CINN_WITH_CUDA
CINN_USE_REGISTER(cinn_call_cuda_kernel)
#endif
