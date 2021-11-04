#include "infrt/kernel/tensor_shape_kernels.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

#include "infrt/host_context/kernel_registry.h"
#include "infrt/host_context/kernel_utils.h"
#include "infrt/tensor/tensor_shape.h"

namespace infrt::kernel {

void PrintShape(const tensor::TensorShape& shape) {
  llvm::raw_os_ostream oos(std::cout);
  oos << shape << '\n';
}

void RegisterTensorShapeKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("ts.print_shape", CINN_KERNEL(PrintShape));
}

}  // namespace infrt::kernel
