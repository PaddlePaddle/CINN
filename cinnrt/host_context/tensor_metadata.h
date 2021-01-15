#pragma once

#include "cinnrt/common/dtype.h"
#include "cinnrt/host_context/tensor_shape.h"

namespace cinnrt {
namespace host_context {

struct TensorMetadata {
  DType dtype;
  TensorShape shape;

  TensorMetadata(DType dtype, const TensorShape& shape) : dtype(dtype), shape(shape) {}
  TensorMetadata(DType dtype, llvm::ArrayRef<int64_t> shape) : dtype(dtype), shape(shape) {}

  size_t GetHostSizeInBytes() const { return dtype.GetHostSize() * shape.GetNumElements(); }

  bool operator==(const TensorMetadata& other) const { return dtype == other.dtype && shape == other.shape; }
  bool operator!=(const TensorMetadata& other) const { return !(*this == other); }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorMetadata& meta);
};

}  // namespace host_context
}  // namespace cinnrt
