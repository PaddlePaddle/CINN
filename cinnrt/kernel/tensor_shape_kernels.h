#pragma once

namespace infrt::host_context {

class KernelRegistry;

}  // namespace infrt::host_context

namespace infrt::kernel {

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry);

}  // namespace infrt::kernel
