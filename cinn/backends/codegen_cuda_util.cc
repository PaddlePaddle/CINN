#include "cinn/backends/codegen_cuda_util.h"

#include "cinn/backends/cuda_util.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace backends {

std::tuple<lang::Module, lang::Module> SplitCudaAndHostModule(lang::Module module) {
  detail::CollectHostFunctionVisitor visitor(module->name);
  Expr expr(module);
  return visitor(&expr);
}

bool detail::CollectHostFunctionVisitor::IsCudaFunction(const ir::_LoweredFunc_* func) {
  return func->device_api == ir::DeviceAPI::GPU;
}

}  // namespace backends
}  // namespace cinn
