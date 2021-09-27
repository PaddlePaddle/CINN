#pragma once

#include "cinn/backends/extern_func_jit_register.h"

CINN_USE_REGISTER(host_intrinsics)
#ifdef CINN_WITH_MKL_CBLAS
CINN_USE_REGISTER(mkl_math)
CINN_USE_REGISTER(cinn_cpu_mkl)
#endif

#ifdef CINN_WITH_MKLDNN
CINN_USE_REGISTER(cinn_cpu_mkldnn)
#endif

CINN_USE_REGISTER(cinn_backend_parallel)
