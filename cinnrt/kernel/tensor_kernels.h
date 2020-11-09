#pragma once

namespace cinnrt::host_context {
struct KernelRegistry;
}  // namespace cinnrt::host_context

namespace cinnrt::kernel {

void RegisterTensorKernels(host_context::KernelRegistry* registry);

}  // namespace cinnrt::kernel
