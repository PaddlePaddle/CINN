#pragma once
#include <string>

namespace infrt::host_context {

struct KernelRegistry;

}  // namespace infrt::host_context

namespace infrt::kernel {

/**
 * Register all the basic kernels to \p registry.
 */
void RegisterBasicKernels(host_context::KernelRegistry* registry);

void RegisterIntBasicKernels(host_context::KernelRegistry* registry);
void RegisterFloatBasicKernels(host_context::KernelRegistry* registry);

}  // namespace infrt::kernel
