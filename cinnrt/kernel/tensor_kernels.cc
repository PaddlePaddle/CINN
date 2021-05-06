#include "cinnrt/kernel/tensor_kernels.h"

#include <iostream>
#include <vector>

#include "cinnrt/common/global.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/tensor/dense_host_tensor.h"
#include "cinnrt/tensor/dense_tensor_view.h"
#include "cinnrt/tensor/tensor_map.h"
#include "cinnrt/tensor/tensor_shape.h"

namespace cinnrt::kernel {
using namespace host_context;  // NOLINT
using namespace tensor;        // NOLINT

/// ===== Kernel begin ====

template <typename T>
DenseHostTensor CreateUninitTensor(Attribute<std::vector<int64_t>> shape) {
  const auto &shape_data = shape.get();
  auto array             = llvm::ArrayRef<int64_t>(shape_data.data(), shape_data.size());
  auto type              = GetDType<T>();
  return DenseHostTensor(TensorShape(array), type);
}

void PrintTensor(const DenseHostTensor &tensor) { std::cout << tensor << std::endl; }

template <typename T>
void FillTensorWithConstant(DenseHostTensor *tensor, Attribute<T> v) {
  MutableDTArrayView<T>(tensor).Fill(v.get());
}

TensorMap LoadParams(const std::string &path) { return *(cinnrt::tensor::LoadParams(path)); }

DenseHostTensor GetParam(TensorMap map, Attribute<std::string> nameAttr) {
  auto &name = nameAttr.get();
  return *(map[name]);
}

DenseHostTensor ShallowCopyTensor(DenseHostTensor v) { return v; }

/// ===== Kernel end ====

void RegisterTensorKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("dt.create_uninit_tensor.f32", CINN_KERNEL(CreateUninitTensor<float>));
  registry->AddKernelAttrNameList("dt.create_uninit_tensor.f32", {"shape"});
  registry->AddKernel("dt.print_tensor", CINN_KERNEL(PrintTensor));
  registry->AddKernel("dt.fill_tensor_with_constant.f32", CINN_KERNEL(FillTensorWithConstant<float>));
  registry->AddKernel("dt.fill_tensor_with_constant.f64", CINN_KERNEL(FillTensorWithConstant<double>));
  registry->AddKernel("dt.load_params", CINN_KERNEL(LoadParams));
  registry->AddKernel("dt.get_param", CINN_KERNEL(GetParam));
  registry->AddKernel("dt.shallow_copy_tensor", CINN_KERNEL(ShallowCopyTensor));
}

}  // namespace cinnrt::kernel
