#pragma once
#include <string>

namespace cinnrt::host_context {

struct KernelRegistry;

}  // namespace cinnrt::host_context

namespace cinnrt::kernel {

/**
 * Register all the test kernels to registry.
 */
void RegisterTestKernels(host_context::KernelRegistry* registry);

}  // namespace cinnrt::kernel
