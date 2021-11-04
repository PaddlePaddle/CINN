#pragma once

#include <glog/logging.h>

#include "infrt/common/dtype.h"
#include "infrt/tensor/tensor_shape.h"

namespace infrt {
namespace tensor {

struct TensorMetadata {
  DType dtype;
  TensorShape shape;

  TensorMetadata() = default;
  TensorMetadata(DType dtype, const TensorShape& shape) : dtype(dtype), shape(shape) { CHECK(IsValid()); }
  TensorMetadata(DType dtype, llvm::ArrayRef<int64_t> shape) : dtype(dtype), shape(shape) { CHECK(IsValid()); }

  size_t GetHostSizeInBytes() const { return dtype.GetHostSize() * shape.GetNumElements(); }

  bool IsValid() const { return dtype.IsValid(); }
  bool IsInvalid() const { return !dtype.IsValid(); }

  bool operator==(const TensorMetadata& other) const { return dtype == other.dtype && shape == other.shape; }
  bool operator!=(const TensorMetadata& other) const { return !(*this == other); }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorMetadata& meta);
};

}  // namespace tensor
}  // namespace infrt
