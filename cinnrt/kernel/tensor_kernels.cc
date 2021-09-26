#include <llvm/Support/raw_ostream.h>
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

void PrintTensor(const DenseHostTensor &tensor) {
    std::cout << "print tensor ref: " << &tensor << std::endl;
    std::cout << tensor << std::endl;
}

void PrintTensorPointer(DenseHostTensor *tensor) {
    std::cout << "print tensor pointer: " << tensor << std::endl;
    std::cout << *tensor << std::endl;
}

template <typename T>
void FillTensorWithConstant(DenseHostTensor *tensor, Attribute<T> v) {
  MutableDTArrayView<T>(tensor).Fill(v.get());
}

template <typename T>
void FillTensorAndReturn(DenseHostTensor* tensor, Attribute<T> v, ReturnNew<DenseHostTensorRef> ret_tensor) {
  //DenseHostTensor* tensor = ret_tensor.get();
  std::cout << "Kernel FillTensorAndReturn" << std::endl;
  llvm::outs() << tensor->shape() << "\n";
  MutableDTArrayView<T>(tensor).Fill(v.get());
  std::cout << "before tensor: " << tensor << " return tensor: " << ret_tensor.get().get() << std::endl;
  ret_tensor.Emplace(DenseHostTensorRef(tensor));
  std::cout << "before tensor: " << tensor << " return tensor: " << ret_tensor.get().get() << std::endl;
 
  
  //MutableDTArrayView<T>(tensor.get()).Fill(v.get());
  //return *tensor;
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
  registry->AddKernel("external.print_tensor_pointer", CINN_KERNEL(PrintTensorPointer));
  registry->AddKernel("dt.fill_tensor_with_constant.f32", CINN_KERNEL(FillTensorWithConstant<float>));
  registry->AddKernel("dt.fill_tensor_with_constant.f64", CINN_KERNEL(FillTensorWithConstant<double>));
  registry->AddKernel("dt.fill_tensor_and_return.f32", CINN_KERNEL(FillTensorAndReturn<float>));
  registry->AddKernel("dt.load_params", CINN_KERNEL(LoadParams));
  registry->AddKernel("dt.get_param", CINN_KERNEL(GetParam));
  registry->AddKernel("dt.shallow_copy_tensor", CINN_KERNEL(ShallowCopyTensor));
}

}  // namespace cinnrt::kernel
