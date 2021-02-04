#pragma once

#include <glog/logging.h>

#include "cinnrt/tensor/dense_host_tensor.h"

namespace cinnrt::tensor {

template <typename DType>
class DTArrayView {
 public:
  using UnderlyingT = DenseHostTensor;

  explicit DTArrayView(const DenseHostTensor* tensor) : tensor_(*tensor) {}

  const TensorShape& shape() { return tensor_.shape(); }

  size_t GetNumElements() const { return tensor_.shape().GetNumElements(); }

  const DType* data() const { return static_cast<const DType*>(tensor_.raw_data()); }
  DType* data() { return static_cast<DType*>(tensor_.raw_data()); }

  llvm::ArrayRef<DType> Elements() const { return llvm::ArrayRef<DType>(data(), GetNumElements()); }

 private:
  const DenseHostTensor& tensor_;
};

template <typename DType>
class MutableDTArrayView : public DTArrayView<DType> {
 public:
  explicit MutableDTArrayView(DenseHostTensor* tensor) : DTArrayView<DType>(tensor) {}

  void Fill(const DType& v) { std::fill(this->data(), this->data() + this->GetNumElements(), v); }

  using DTArrayView<DType>::data;
  using DTArrayView<DType>::GetNumElements;
  llvm::MutableArrayRef<DType> Elements() { return llvm::MutableArrayRef<DType>(data(), this->GetNumElements()); }
};

}  // namespace cinnrt::tensor
