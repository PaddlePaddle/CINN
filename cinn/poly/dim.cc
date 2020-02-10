#include "cinn/poly/dim.h"

#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

std::string Dim::range_repr() const {
  return utils::StringFormat("%d <= %s <= %d", lower_bound, id.c_str(), upper_bound);
}

}  // namespace poly
}  // namespace cinn
