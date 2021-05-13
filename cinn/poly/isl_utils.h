#pragma once

#include <isl/cpp.h>
#include <llvm/ADT/ArrayRef.h>

#include <string>
#include <vector>

namespace cinn {
namespace poly {

//! Get dimension names from isl containers.
// @{
std::vector<std::string> isl_get_dim_names(const isl::set& x);
std::vector<std::string> isl_get_dim_names(const isl::map& x, isl_dim_type dim_type);
std::vector<std::string> isl_get_dim_names(isl_map* map, isl_dim_type dim_type);
std::vector<std::string> isl_get_dim_names(isl_set* set);
// @}

void isl_set_dim_names(isl::set* __isl_keep set, const std::vector<std::string>& names);
void isl_set_dim_names(isl::map* __isl_keep map, isl_dim_type dim_type, const std::vector<std::string>& names);

isl::union_set isl_union_set_from_sets(llvm::ArrayRef<isl::set> sets);

isl::map isl_set_dim_name_if_null(isl_map* __isl_take map, std::function<std::string(isl_dim_type, int)> namer);
isl::set isl_set_dim_name_if_null(isl_set* __isl_take set, std::function<std::string(isl_dim_type, int)> namer);

//! Convert a list of isl::map to isl::union_map
isl::union_map isl_maps_to_union_map(const std::vector<isl::map>& maps);
isl::union_set isl_sets_to_union_set(const std::vector<isl::set>& sets);

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

// { s[i]: 0 < i < 20 }
bool isl_set_axis_has_noparam_constant_bound(isl_set* __isl_keep set, int pos);

//! get a minimum and maximum range of a set, if the bound not exists, return a INT_MAX instead.
//! NOTE the set should be bound.
//! returns: a tuple of (min, max)
std::tuple<isl::val, isl::val> isl_set_get_axis_range(isl_set* __isl_keep set, int pos);

std::tuple<isl::val, isl::val> isl_set_get_axis_range_by_name(isl_set* __isl_keep set, std::string axis_name);

//! Port the set from \p from to \p to with the \p poses dims constraints remained.
//! @param from The set to port.
//! @param to The set to be.
//! @param poses The dimensions to remained.
isl_set* __isl_give isl_set_port_to_other(isl_set* __isl_give from,
                                          isl_set* __isl_give to,
                                          const std::vector<int>& poses);

//! Set get a new set consists of several dimensions.
//! e.g. { s[i,j,k]: 0<i,j,k<100}, get {0,2} dims, get { s[i,k]: 0<i,k<100 }
isl::set SetGetDims(isl::set set, const std::vector<int>& dims);

/**
 * Given an isl::map and a vector of names of dim_in,
 * remove the input dims in vector and related output dims.
 * @param x The map to edit.
 * @param dim_in_names The names of input dims to remove.
 * @return The edited map.
 */
isl::map RemoveAxiesByNames(const isl::map& x, const std::vector<std::string>& dim_in_names);

/**
 * Given an isl::map and the name of an output dim,
 * get the names of related input dims.
 * @param x The input map.
 * @param dim_in_names The name of an output dim.
 * @return The vector of names of related input dims.
 */
std::vector<std::string> GetRelatedAxies(const isl::map& x, const std::string& dim_out_name);

}  // namespace poly
}  // namespace cinn
