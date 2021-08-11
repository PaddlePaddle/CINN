#include "cinnrt/tensor/dense_host_tensor.h"

#include <llvm/Support/raw_os_ostream.h>

#include "cinnrt/common/buffer.h"

namespace cinnrt::tensor {

DenseHostTensor::DenseHostTensor(const TensorShape& shape, DType dtype) : HostTensor(TensorMetadata{dtype, shape}) {
  CHECK(metadata().IsValid()) << "Tensor construct get invalid metadata";
  buffer_.reset(new cinnrt::Buffer(cinnrt::common::DefaultHostTarget()));
  buffer_->ResizeLazy(dtype.GetHostSize() * shape.GetNumElements());
}

const TensorShape& DenseHostTensor::shape() const { return metadata().shape; }

void DenseHostTensor::Init(const std::vector<int64_t>& shape, DType dtype) {
  auto shape_array = llvm::ArrayRef<int64_t>(shape.data(), shape.size());
  auto metadata    = TensorMetadata(dtype, shape_array);
  setTensorMetadata(metadata);
  buffer_.reset(new cinnrt::Buffer(cinnrt::common::DefaultHostTarget()));
  buffer_->ResizeLazy(dtype.GetHostSize() * metadata.shape.GetNumElements());
}

const cinnrt::Buffer* DenseHostTensor::buffer() const { return buffer_.get(); }

template <typename T>
void DisplayArray(std::ostream& os, T* data, int num_elements) {
  for (int i = 0; i < num_elements - 1; i++) os << data[i] << ", ";
  if (num_elements > 0) os << data[num_elements - 1];
}

std::ostream& operator<<(std::ostream& os, const DenseHostTensor& instance) {
  CHECK(instance.metadata().IsValid()) << "Cann't print tensor with invalid metadata";
  llvm::raw_os_ostream oos(os);
  oos << "tensor: ";
  oos << "shape=";
  oos << instance.shape();
  oos << ", values=[";

  oos.flush();

  if (instance.metadata().dtype == GetDType<float>()) {
    auto* data = reinterpret_cast<float*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<double>()) {
    auto* data = reinterpret_cast<double*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<int32_t>()) {
    auto* data = reinterpret_cast<int32_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<int64_t>()) {
    auto* data = reinterpret_cast<int64_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else {
    LOG(FATAL) << "Not supported dtype [" << instance.metadata().dtype.name() << " "
               << static_cast<int>(instance.metadata().dtype.kind()) << "] in print";
  }

  os << "]";

  return os;
}

DenseHostTensor::~DenseHostTensor() {}

void* DenseHostTensor::raw_data() const { return buffer_->data()->memory; }

}  // namespace cinnrt::tensor
