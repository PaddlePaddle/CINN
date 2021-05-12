#include "cinn/runtime/cpu/thread_backend.h"

#include <algorithm>
#include <vector>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/common/cas.h"
#include "cinn/runtime/intrinsic.h"

int max_concurrency() {
  int max_concurrency = 1;
  const char* val     = getenv("CINN_NUM_THREADS");
  if (val == nullptr) {
    val = getenv("OMP_NUM_THREADS");
  }
  if (val != nullptr) {
    max_concurrency = atoi(val);
  } else {
    max_concurrency = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    max_concurrency /= 2;  // ignore hyper-threading
#endif
  }
  return std::max(max_concurrency, 1);
}

int cinn_backend_parallel_launch(FCINNParallelLambda flambda, void* datas, int num_task) {
  int num_workers = max_concurrency();
  if (num_task == 0) num_task = num_workers;
  omp_set_num_threads(num_task);
#pragma omp parallel num_threads(num_task)
  {
    int thread_num = omp_get_thread_num();
    (*flambda)(thread_num, num_task, datas);
  }
  return 0;
}

CINN_REGISTER_HELPER(cinn_backend_parallel) {
  using namespace cinn;  // NOLINT
  using backends::FunctionProto;
  auto host_target = common::DefaultHostTarget();
  backends::RuntimeSymbolRegistry::Global().RegisterFn(runtime::intrinsic::parallel_launch,
                                                       reinterpret_cast<void*>(&cinn_backend_parallel_launch));
  return true;
}
