#pragma once
#include <glog/logging.h>
#include <stddef.h>

#include <variant>
#include <vector>

#include "cinn/ir/ir.h"

namespace hlir {
namespace instruction {

/**
 * Represent the shape of Tensors in HLIR, it is used in both the analysis and compile phases.
 *
 * ## About the static and dynamic dimension
 * The static dimension stores as the positive integers, and the dynamic shape is stored as -1.
 */
struct Shape {
  //! A dimension, can either be a static dimension(int) or a dynamic one(Var).
  using dim_t = std::variant<int, cinn::Var>;

  Shape() = default;

  Shape(std::initializer_list<dim_t> list);

  //! Add a new dimension to the shape.
  inline void AddDim(int dim) { dims_.push_back(dim); }
  inline void AddDim(cinn::Var dim) { dims_.push_back(dim); }
  inline void AddDim(dim_t dim) { dims_.push_back(dim); }

  std::string to_debug_string() const;

  std::vector<cinn::Var> CollectDynamicDims() const;

  //! number of dimensions.
  inline size_t num_dims() const { return dims_.size(); }

  const dim_t& operator[](int offset) const;
  dim_t& operator[](int offset);
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other);

  friend std::ostream& operator<<(std::ostream& os, const Shape& other) {
    os << other.to_debug_string();
    return os;
  }

  std::vector<cinn::Expr> ToCinnShape() const;

 private:
  std::vector<dim_t> dims_;
};

}  // namespace instruction
}  // namespace hlir
