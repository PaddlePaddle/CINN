#pragma once

namespace cinn::host_context {

class KernelRegistry;

}  // namespace cinn::host_context

namespace cinn::kernel {

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry);

}  // namespace cinn::kernel
