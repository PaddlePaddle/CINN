#pragma once

#include <llvm/ADT/ArrayRef.h>

namespace cinn {
namespace host_context {

/**
 * TensorShape represents the shape of a Tensor, all the dimensions should be known.
 */
class TensorShape {
 public:
  explicit TensorShape(llvm::ArrayRef<int64_t> dims);

  int GetRank() const;

  int GetNumElements() const;

  friend std::ostream& operator<<(std::ostream& os, const TensorShape& v);

 private:
  llvm::SmallVector<int64_t, 4> dims_;
};

/**
 * DynamicTensorShape represents the shape of a Tensor, with some dimensions or even the rank is unknown.
 */
class DynamicTensorShape {
 public:
  explicit DynamicTensorShape(std::optional<llvm::ArrayRef<int64_t>> dims);

  //! Returns the rank if rank is known, or kUnknownDimSize.
  int GetRank() const;

  int64_t GetDim(int idx) const;

  bool IsShapeKnown() const;

  //! Convert to a TensorShape if all the dimensions are known.
  std::optional<TensorShape> ToTensorShape() const;

  static constexpr int64_t kUnknownDimSize = -1;

  static bool IsDimUnknown(int64_t dim) { return dim == kUnknownDimSize; }

  friend std::ostream& operator<<(std::ostream& os, const DynamicTensorShape& v);

 private:
  //! Will be std::nullopt if no dim is known.
  std::optional<llvm::SmallVector<int64_t, 4>> dims_;
};

}  // namespace host_context
}  // namespace cinn
