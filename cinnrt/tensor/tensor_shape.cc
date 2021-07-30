#include "cinnrt/tensor/tensor_shape.h"

#include <glog/logging.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <functional>

namespace cinnrt {
namespace tensor {

TensorShape::TensorShape(llvm::ArrayRef<int64_t> dims) : dims_(dims.begin(), dims.end()) {}

int TensorShape::GetRank() const { return dims_.size(); }

int64_t TensorShape::GetDim(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, GetRank());
  return dims_[idx];
}
int TensorShape::GetNumElements() const {
  int64_t size = 1;
  for (int v : dims_) size *= v;
  return size;
}

DynamicTensorShape::DynamicTensorShape(llvm::Optional<llvm::ArrayRef<int64_t>> dims) {
  if (dims.hasValue()) {
    dims_ = llvm::SmallVector<int64_t, 4>(dims->begin(), dims->end());
  }
}

int DynamicTensorShape::GetRank() const {
  if (dims_.hasValue()) return dims_->size();
  return kUnknownDimSize;
}

int64_t DynamicTensorShape::GetDim(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, GetRank());
  return (*dims_)[idx];
}

bool DynamicTensorShape::IsShapeKnown() const {
  if (!dims_.hasValue()) return false;
  for (int64_t v : *dims_) {
    if (IsDimUnknown(v)) return false;
  }
  return true;
}

llvm::Optional<TensorShape> DynamicTensorShape::ToTensorShape() const {
  if (IsShapeKnown()) {
    return TensorShape(*dims_);
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TensorShape& v) {
  os << "shape[";
  for (int i = 0; i < v.GetRank() - 1; i++) {
    os << v.dims_[i] << ",";
  }
  if (v.GetRank() > 0) os << v.dims_.back();
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DynamicTensorShape& v) {
  os << "dynamic_shape[";
  for (int i = 0; i < v.GetRank() - 1; i++) {
    os << v << ",";
  }
  if (v.GetRank() > 0) os << v.dims_->back();
  os << "]";
  return os;
}

}  // namespace tensor
}  // namespace cinnrt
