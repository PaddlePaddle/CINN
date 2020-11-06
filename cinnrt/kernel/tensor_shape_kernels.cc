#include "cinnrt/kernel/tensor_shape_kernels.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <iostream>

#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinn::kernel {

void PrintShape(const host_context::TensorShape& shape) { std::cout << shape << std::endl; }

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("ts.print_shape", CINN_KERNEL(PrintShape));
}

}  // namespace cinn::kernel
