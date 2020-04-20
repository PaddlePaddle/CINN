#pragma once
#include <glog/logging.h>
#include <stddef.h>
#include <vector>

namespace hlir {
namespace instruction {

/**
 * Represent the shape of Tensors in HLIR, it is used in both the analysis and compile phases.
 *
 * ## About the static and dynamic dimension
 * The static dimension stores as the positive integers, and the dynamic shape is stored as -1.
 */
struct Shape {
  //! Add a new dimension to the shape.
  inline void AddDim(int dim) { dims_.push_back(dim); }

  //! number of dimensions.
  inline size_t num_dims() const { return dims_.size(); }

  //! Get the \p offset -th dimension.
  int operator[](int offset) const;
  int& operator[](int offset);

  bool operator==(const Shape& other) const;

  bool operator!=(const Shape& other);

 private:
  std::vector<int> dims_;
};

}  // namespace instruction
}  // namespace hlir
