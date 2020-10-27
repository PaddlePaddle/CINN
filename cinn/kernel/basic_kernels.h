#pragma once
#include <string>

namespace cinn::host_context {

struct KernelRegistry;

}  // namespace cinn::host_context

namespace cinn::kernel {

/**
 * Register all the basic kernels to \p registry.
 */
void RegisterBasicKernels(host_context::KernelRegistry* registry);

void RegisterIntBasicKernels(host_context::KernelRegistry* registry);
void RegisterFloatBasicKernels(host_context::KernelRegistry* registry);

}  // namespace cinn::kernel
