#pragma once

#include "cinnrt/host_context/function.h"
#include "cinnrt/host_context/kernel_utils.h"

namespace cinnrt {

namespace host_context {
struct KernelRegistry;
}  // namespace host_context

namespace kernel {

void RegisterControlFlowKernels(host_context::KernelRegistry* registry);

}  // namespace kernel
}  // namespace cinnrt
