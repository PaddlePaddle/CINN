#pragma once

namespace cinn::host_context {
struct KernelRegistry;
}  // namespace cinn::host_context

namespace cinn::kernel {

void RegisterTensorKernels(host_context::KernelRegistry* registry);

}  // namespace cinn::kernel
