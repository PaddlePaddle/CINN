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

std::vector<std::string> GetDimNames(isl_map* map, isl_dim_type dim_type);
std::vector<std::string> GetDimNames(isl_set* set);

isl::map SetDimNameIfNull(isl_map* __isl_take map, std::function<std::string(isl_dim_type, int)> namer);

isl::set SetDimNameIfNull(isl_set* __isl_take set, std::function<std::string(isl_dim_type, int)> namer);

//! Convert a list of isl::map to isl::union_map
isl::union_map MapsToUnionMap(const std::vector<isl::map>& maps);
isl::union_set SetsToUnionSet(const std::vector<isl::set>& sets);

//! Set get a new set consists of several dimensions.
//! e.g. { s[i,j,k]: 0<i,j,k<100}, get {0,2} dims, get { s[i,k]: 0<i,k<100 }
isl::set SetGetDims(isl::set set, const std::vector<int>& dims);

//! Get a representation of the tuple in the map.
std::string isl_map_get_statement_repr(__isl_keep isl_map* map, isl_dim_type type);

isl_set* __isl_give isl_get_precending_aixs(isl_set* set, int level, bool with_tuple_name);

//! Get the maximum level of axis that is has the same domain.
int isl_max_level_compatible(isl_set* __isl_keep a, isl_set* __isl_keep b);

isl_set* __isl_give isl_remove_axis_by_name(isl_set* __isl_take set, const char* axis_name);
isl_map* __isl_give isl_remove_axis_by_name(isl_map* __isl_take map, isl_dim_type dim_type, const char* axis_name);
isl_set* __isl_give isl_rename_axis(isl_set* __isl_take set, int offset, const char* name);
isl_map* __isl_give isl_rename_axis(isl_map* __isl_take map, isl_dim_type dim_type, int offset, const char* name);

isl_set* __isl_give isl_simplify(isl_set* __isl_take set);

//! get a minimum and maximum range of a set, if the bound not exists, return a INT_MAX instead.
//! NOTE the set should be bound.
//! returns: a tuple of (min, max)
std::tuple<isl::val, isl::val> isl_set_get_axis_range(isl_set* __isl_keep set, int pos);

//! Port the set from \p from to \p to with the \p poses dims constraints remained.
//! @param from The set to port.
//! @param to The set to be.
//! @param poses The dimensions to remained.
isl_set* __isl_give isl_set_port_to_other(isl_set* __isl_give from,
                                          isl_set* __isl_give to,
                                          const std::vector<int>& poses);

}  // namespace poly
}  // namespace cinn
