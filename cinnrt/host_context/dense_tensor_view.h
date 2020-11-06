#pragma once

#include <glog/logging.h>

#include "cinnrt/host_context/dense_tensor.h"

namespace cinn::host_context {

template <typename DType>
class DTArrayView {
 public:
  using UnderlyingT = DenseTensor;

  explicit DTArrayView(const DenseTensor* tensor) : tensor_(*tensor) {}

  const TensorShape& shape() { return tensor_.shape(); }

  size_t GetNumElements() const { return tensor_.shape().GetNumElements(); }

  const DType* data() const { return static_cast<const DType*>(tensor_.data()); }
  DType* data() { return static_cast<DType*>(tensor_.data()); }

  llvm::ArrayRef<DType> Elements() const { return llvm::ArrayRef<DType>(data(), GetNumElements()); }

 private:
  const DenseTensor& tensor_;
};

template <typename DType>
class MutableDTArrayView : public DTArrayView<DType> {
 public:
  explicit MutableDTArrayView(DenseTensor* tensor) : DTArrayView<DType>(tensor) {}

  void Fill(const DType& v) { std::fill(this->data(), this->data() + this->GetNumElements(), v); }

  using DTArrayView<DType>::data;
  using DTArrayView<DType>::GetNumElements;
  llvm::MutableArrayRef<DType> Elements() { return llvm::MutableArrayRef<DType>(data(), this->GetNumElements()); }
};

}  // namespace cinn::host_context
