#include "cinnrt/kernel/tensor_kernels.h"

#include <iostream>
#include <vector>

#include "cinnrt/host_context/dense_host_tensor.h"
#include "cinnrt/host_context/dense_tensor_view.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinnrt::kernel {
using namespace host_context;  // NOLINT

/// ===== Kernel begin ====

template <typename T>
DenseHostTensor CreateUninitTensor(Attribute<std::vector<int64_t>> shape) {
  const auto& shape_data = shape.get();
  auto array             = llvm::ArrayRef<int64_t>(shape_data.data(), shape_data.size());
  auto type              = GetDType<T>();
  return DenseHostTensor(TensorShape(array), type);
}

void PrintTensor(const DenseHostTensor& tensor) { std::cout << tensor << std::endl; }

template <typename T>
void FillTensorWithConstant(DenseHostTensor* tensor, Attribute<T> v) {
  MutableDTArrayView<T>(tensor).Fill(v.get());
}

/// ===== Kernel end ====

void RegisterTensorKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("dt.create_uninit_tensor.f32", CINN_KERNEL(CreateUninitTensor<float>));
  registry->AddKernelAttrNameList("dt.create_uninit_tensor.f32", {"shape"});
  registry->AddKernel("dt.print_tensor", CINN_KERNEL(PrintTensor));
  registry->AddKernel("dt.fill_tensor_with_constant.f32", CINN_KERNEL(FillTensorWithConstant<float>));
  registry->AddKernel("dt.fill_tensor_with_constant.f64", CINN_KERNEL(FillTensorWithConstant<double>));
}

}  // namespace cinnrt::kernel
