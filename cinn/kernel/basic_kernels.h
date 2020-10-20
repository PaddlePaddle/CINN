#pragma once
#include <string>

namespace cinn::host_context {

struct KernelRegistry;

}  // namespace cinn::host_context

namespace cinn::kernel {

void RegisterIntBasicKernels(host_context::KernelRegistry* registry);
void RegisterFloatBasicKernels(host_context::KernelRegistry* registry);

}  // namespace cinn::kernel
