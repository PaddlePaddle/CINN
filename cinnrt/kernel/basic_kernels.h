#pragma once
#include <string>

namespace cinnrt::host_context {

struct KernelRegistry;

}  // namespace cinnrt::host_context

namespace cinnrt::kernel {

/**
 * Register all the basic kernels to \p registry.
 */
void RegisterBasicKernels(host_context::KernelRegistry* registry);

void RegisterIntBasicKernels(host_context::KernelRegistry* registry);
void RegisterFloatBasicKernels(host_context::KernelRegistry* registry);

}  // namespace cinnrt::kernel
