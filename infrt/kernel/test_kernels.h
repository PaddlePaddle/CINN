#pragma once
#include <string>

namespace infrt::host_context {

struct KernelRegistry;

}  // namespace infrt::host_context

namespace infrt::kernel {

/**
 * Register all the test kernels to registry.
 */
void RegisterTestKernels(host_context::KernelRegistry* registry);

}  // namespace infrt::kernel
