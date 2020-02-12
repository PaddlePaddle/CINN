#include "cinn/poly/isl_utils.h"

#include <glog/logging.h>
#include <isl/cpp.h>

#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

std::vector<std::string> GetDimNames(const isl::set &x) {
  std::vector<std::string> res;
  for (int i = 0; i < isl_set_dim(x.get(), isl_dim_set); i++) {
    res.push_back(isl_set_get_dim_name(x.get(), isl_dim_set, i));
  }
  return res;
}

std::vector<std::string> GetDimNames(const isl::map &x, isl_dim_type dim_type) {
  std::vector<std::string> res;
  for (int i = 0; i < isl_map_dim(x.get(), dim_type); i++) {
    res.push_back(isl_map_get_dim_name(x.get(), dim_type, i));
  }
  return res;
}

void SetDimNames(isl::map *map, isl_dim_type dim_type, const std::vector<std::string> &names) {
  const int dim = isl_map_dim(map->get(), dim_type);
  CHECK_EQ(dim, names.size());

  for (int i = 0; i < dim; i++) {
    *map = isl::manage(isl_map_set_dim_name(map->release(), dim_type, i, names[i].c_str()));
  }
}

void SetDimNames(isl::set *set, const std::vector<std::string> &names) {
  int dim = isl_set_dim(set->get(), isl_dim_set);
  CHECK_EQ(dim, names.size());

  for (int i = 0; i < dim; i++) {
    *set = isl::manage(isl_set_set_dim_name(set->release(), isl_dim_set, i, names[i].c_str()));
  }
}

isl::union_map MapsToUnionMap(const std::vector<isl::map> &maps) {
  CHECK(!maps.empty());
  isl::union_map umap = isl::manage(isl_union_map_from_map(maps.front().copy()));
  for (int i = 1; i < maps.size(); i++) {
    umap = isl::manage(isl_union_map_add_map(umap.release(), maps[i].copy()));
  }
  return umap;
}

isl::union_set SetsToUnionSet(const std::vector<isl::set> &sets) {
  CHECK(!sets.empty());
  isl::union_set uset = isl::manage(isl_union_set_from_set(sets.front().copy()));
  for (int i = 1; i < sets.size(); i++) {
    uset = isl::manage(isl_union_set_add_set(uset.release(), sets[i].copy()));
  }
  return uset;
}

std::string isl_map_get_statement_repr(__isl_keep isl_map *map, isl_dim_type type) {
  CHECK(map);
  auto tuple_name = isl_map_get_tuple_name(map, type);
  std::vector<std::string> dims;

  for (int i = 0; i < isl_map_dim(map, type); i++) {
    dims.push_back(isl_map_get_dim_name(map, type, i));
  }
  return utils::StringFormat("%s[%s]", tuple_name, utils::Join(dims, ", ").c_str());
}

}  // namespace poly
}  // namespace cinn
