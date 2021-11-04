#pragma once

namespace infrt::host_context {
struct KernelRegistry;
}  // namespace infrt::host_context

namespace infrt::kernel {

void RegisterTensorKernels(host_context::KernelRegistry* registry);

}  // namespace infrt::kernel
