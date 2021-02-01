#include "cinnrt/kernel/control_flow_kernels.h"
#include <glog/logging.h>
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_function.h"

namespace cinnrt {
namespace kernel {

static void CINNCall(host_context::RemainingArguments args,
                     host_context::RemainingResults results,
                     host_context::Attribute<host_context::MlirFunction*> fn) {
  VLOG(3) << "running call kernel ...";
  CHECK_EQ(fn.get()->num_arguments(), args.size());
  CHECK_EQ(fn.get()->num_results(), results.size());

  for (auto& v : results.values()) {
    CHECK(v.get());
  }
  fn.get()->Execute(args.values(), results.values());
}

void RegisterControlFlowKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("cinn.call", CINN_KERNEL(CINNCall));
}

}  // namespace kernel
}  // namespace cinnrt
