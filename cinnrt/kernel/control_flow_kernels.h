#pragma once

#include "infrt/host_context/function.h"
#include "infrt/host_context/kernel_utils.h"

namespace infrt {

namespace host_context {
struct KernelRegistry;
}  // namespace host_context

namespace kernel {

void RegisterControlFlowKernels(host_context::KernelRegistry* registry);

}  // namespace kernel
}  // namespace infrt
