#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir_base.h"

/**
 * \file
 * This file defines Dim class, which represents the dimension in polyhedral.
 */

namespace cinn {
namespace poly {

/**
 * Dimension with name and range.
 *
 * This is used in ISL to define each dimension of a statement.
 */
struct Dim {
  using value_t = ir::Expr;
  using range_t = std::pair<value_t, value_t>;

  //! The id of the dimension.
  std::string id;
  //! The lower bound.
  value_t lower_bound;
  //! The upper bound.
  value_t upper_bound;

  //! Construct a parameter.
  Dim(std::string id) : id(std::move(id)) {}

  //! Construct a dimension with integer range.
  Dim(std::string id, uint32_t lower_bound, uint32_t upper_bound)
      : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {}

  //! Construct a dimension with expression range.
  Dim(std::string id, ir::Expr lower_bound, ir::Expr upper_bound);

  //! Return the range composed of (lower_bound, upper_bound).
  range_t range() const { return std::make_pair(lower_bound, upper_bound); }

  bool is_param() const { return !lower_bound.defined() && !lower_bound.defined(); }

  //! Return the ISL style range representation, such as '0 <= i <= 20'.
  std::string range_repr() const;
};

}  // namespace poly
}  // namespace cinn
