#pragma once

#include "cinn/backends/extern_func_jit_register.h"

CINN_USE_REGISTER(host_intrinsics)
CINN_USE_REGISTER(mkl_math)
CINN_USE_REGISTER(cinn_cpu_mkl)
CINN_USE_REGISTER(cinn_cpu_mkldnn)
CINN_USE_REGISTER(cinn_backend_parallel)
