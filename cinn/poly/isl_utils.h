#pragma once

#include <isl/cpp.h>
#include <string>
#include <vector>

namespace cinn {
namespace poly {

//! Get dimension names from isl containers.
// @{
std::vector<std::string> GetDimNames(const isl::set& x);
std::vector<std::string> GetDimNames(const isl::map& x, isl_dim_type dim_type);
// @}

void SetDimNames(isl::set* set, const std::vector<std::string>& names);
void SetDimNames(isl::map* map, isl_dim_type dim_type, const std::vector<std::string>& names);

//! Convert a list of isl::map to isl::union_map
isl::union_map MapsToUnionMap(const std::vector<isl::map>& maps);
isl::union_set SetsToUnionSet(const std::vector<isl::set>& sets);

}  // namespace poly
}  // namespace cinn
