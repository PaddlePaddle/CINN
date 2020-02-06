#pragma once

#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

struct Iterator {
  std::string id;

  Iterator() = default;
  explicit Iterator(const std::string& id) : id(id) {}
  explicit Iterator(const Iterator& x) : id(x.id) {}
  explicit Iterator(Iterator&& x) : id(std::move(x.id)) {}

  Iterator& operator=(const Iterator& other) { id = other.id; }

  friend std::ostream& operator<<(std::ostream& os, const Iterator& x);
};

struct Condition {
  Iterator iterator;
  std::string cond;

  Condition(const Iterator& iterator, std::string cond) : iterator(iterator), cond(std::move(cond)) {}

  friend std::ostream& operator<<(std::ostream& os, const Condition& x) {
    os << x.__str__();
    return os;
  }

  std::string __str__() const { return utils::StringFormat("%s", cond.c_str()); }
};

/**
 * A wrapper on isl::map.
 */
class Map {
 public:
  Map(isl::ctx ctx,
      std::string id,
      std::vector<Iterator> domain_iterators,
      std::vector<Iterator> range_iterators,
      std::vector<Condition> conds,
      std::string range_id = "");

  //! Get the corresponding ISL map.
  isl::map to_isl() const;

  //! Get the ISL style map representation, such as '{ S[i,j] -> [i,j]: }'.
  std::string __str__() const;

 private:
  isl::ctx ctx_;
  std::string id_;
  std::vector<Iterator> domain_iterators_;
  std::vector<Iterator> range_iterators_;
  std::vector<Condition> conds_;
  std::string range_id_;
};

}  // namespace poly
}  // namespace cinn
