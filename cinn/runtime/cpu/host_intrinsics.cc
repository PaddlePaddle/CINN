#include "cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "cinn/backends/llvm/runtime_symbol_registry.h"

namespace cinn {
namespace runtime {
namespace cpu {

void __cpu_tanh(cinn_pod_value_t* args, int nargs) {
  CHECK_EQ(nargs, 2);
  CHECK(args) << "Empty args detected";

  float x    = args[0];
  float* out = static_cast<float*>(static_cast<void*>(args[1]));

  *out = tanh(x);
}

namespace {
bool RegisterRuntimeSymbols() {
  auto& registry = backends::RuntimeSymbolRegistry::Global();
  registry.Register("__cpu_tanh", reinterpret_cast<void*>(&__cpu_tanh));
  return true;
}

[[maybe_unused]] bool unused = RegisterRuntimeSymbols();
}  // namespace

}  // namespace cpu
}  // namespace runtime
}  // namespace cinn
