#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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
  using value_t = uint32_t;
  using range_t = std::pair<value_t, value_t>;

  //! The id of the dimension.
  std::string id;
  //! The lower bound.
  value_t lower_bound;
  //! The upper bound.
  value_t upper_bound;

  Dim(std::string id, uint32_t lower_bound, uint32_t upper_bound)
      : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {}

  //! Return the range composed of (lower_bound, upper_bound).
  range_t range() const { return std::make_pair(lower_bound, upper_bound); }

  //! Return the ISL style range representation, such as '0 <= i <= 20'.
  std::string range_repr() const;
};

}  // namespace poly
}  // namespace cinn
