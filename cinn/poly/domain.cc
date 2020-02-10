#include "cinn/poly/domain.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>

#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

std::string Domain::__str__() const {
  std::vector<std::string> range_fields;
  CHECK(!dims.empty());
  std::transform(
      dims.begin(), dims.end(), std::back_inserter(range_fields), [](const Dim& x) { return x.range_repr(); });
  std::string range_repr = utils::Join(range_fields, " and ");

  std::vector<std::string> dim_fields;
  std::transform(dims.begin(), dims.end(), std::back_inserter(dim_fields), [](const Dim& x) { return x.id; });
  std::string dims_repr = utils::Join(dim_fields, ", ");

  return utils::StringFormat("{ %s[%s]: %s }", id.c_str(), dims_repr.c_str(), range_repr.c_str());
}

isl::set Domain::to_isl() const { return isl::set(ctx, __str__()); }

}  // namespace poly
}  // namespace cinn