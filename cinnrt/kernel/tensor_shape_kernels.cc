#include "cinnrt/kernel/tensor_shape_kernels.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <iostream>

#include <llvm/Support/raw_os_ostream.h>
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinnrt::kernel {

void PrintShape(const host_context::TensorShape& shape) {
  llvm::raw_os_ostream oos(std::cout);
  oos << shape << '\n';
}

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("ts.print_shape", CINN_KERNEL(PrintShape));
}

}  // namespace cinnrt::kernel
