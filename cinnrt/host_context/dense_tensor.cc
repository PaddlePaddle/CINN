#include "dense_tensor.h"

#include "cinn/hlir/framework/buffer.h"

namespace cinnrt::host_context {

DenseTensor::DenseTensor(const TensorShape& shape, const cinn_type_t& dtype, DeviceKind device)
    : shape_(shape), dtype_(dtype) {
  buffer_.reset(new cinn::hlir::framework::Buffer(cinn::common::DefaultHostTarget()));
  buffer_->ResizeLazy(dtype.bytes() * shape.GetNumElements());
}

const TensorShape& DenseTensor::shape() const { return shape_; }
const cinn::hlir::framework::Buffer* DenseTensor::buffer() const { return buffer_.get(); }

template <typename T>
void DisplayArray(std::ostream& os, T* data, int num_elements) {
  for (int i = 0; i < num_elements - 1; i++) os << data[i] << ", ";
  if (num_elements > 0) os << data[num_elements - 1];
}

std::ostream& operator<<(std::ostream& os, const DenseTensor& instance) {
  os << "tensor: ";
  os << "shape=";
  os << instance.shape_;
  os << ", values=[";

  if (instance.dtype_ == cinn_type_of<float>()) {
    auto* data = reinterpret_cast<float*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.dtype_ == cinn_type_of<double>()) {
    auto* data = reinterpret_cast<double*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.dtype_ == cinn_type_of<int32_t>()) {
    auto* data = reinterpret_cast<int32_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.dtype_ == cinn_type_of<int64_t>()) {
    auto* data = reinterpret_cast<int64_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else {
    LOG(FATAL) << "Not supported dtype in print";
  }

  os << "]";

  return os;
}

DenseTensor::~DenseTensor() {}

void* DenseTensor::data() const { return buffer_->data()->memory; }

}  // namespace cinnrt::host_context
