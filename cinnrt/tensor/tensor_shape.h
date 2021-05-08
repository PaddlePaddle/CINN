#pragma once

#include <llvm/ADT/ArrayRef.h>

namespace cinnrt {
namespace tensor {

/**
 * TensorShape represents the shape of a Tensor, all the dimensions should be known.
 */
class TensorShape {
 public:
  TensorShape() = default;
  explicit TensorShape(llvm::ArrayRef<int64_t> dims);

  int GetRank() const;

  int64_t GetDim(int idx) const;

  int GetNumElements() const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TensorShape& v);
  friend bool operator==(const TensorShape& a, const TensorShape& b) { return a.dims_ == b.dims_; }

 private:
  llvm::SmallVector<int64_t, 4> dims_;
};

/**
 * DynamicTensorShape represents the shape of a Tensor, with some dimensions or even the rank is unknown.
 */
class DynamicTensorShape {
 public:
  explicit DynamicTensorShape(llvm::Optional<llvm::ArrayRef<int64_t>> dims);

  //! Returns the rank if rank is known, or kUnknownDimSize.
  int GetRank() const;

  int64_t GetDim(int idx) const;

  bool IsShapeKnown() const;

  //! Convert to a TensorShape if all the dimensions are known.
  llvm::Optional<TensorShape> ToTensorShape() const;

  static constexpr int64_t kUnknownDimSize = -1;

  static bool IsDimUnknown(int64_t dim) { return dim == kUnknownDimSize; }

  friend std::ostream& operator<<(std::ostream& os, const DynamicTensorShape& v);
  friend bool operator==(const DynamicTensorShape& a, const DynamicTensorShape& b) { return a.dims_ == b.dims_; }

 private:
  //! Will be std::nullopt if no dim is known.
  llvm::Optional<llvm::SmallVector<int64_t, 4>> dims_;
};

}  // namespace tensor
}  // namespace cinnrt
