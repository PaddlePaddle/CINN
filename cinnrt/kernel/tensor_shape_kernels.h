#pragma once

namespace cinnrt::host_context {

class KernelRegistry;

}  // namespace cinnrt::host_context

namespace cinnrt::kernel {

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry);

}  // namespace cinnrt::kernel
