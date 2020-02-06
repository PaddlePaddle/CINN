#include "cinn/poly/element.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace poly {

std::vector<Iterator> NamesToIterators(const std::vector<std::string> &names) {
  std::vector<Iterator> res;
  for (auto &name : names) {
    res.emplace_back(name);
  }
  return res;
}

void Element::InitSchedule() {
  std::string id = isl_set_get_tuple_name(domain_.get());

  auto dims      = GetDimNames(domain_);
  auto dims_repr = utils::Join(dims, ", ");

  auto repr = utils::StringFormat("{ %s[%s] -> %s[%s] }", id.c_str(), dims_repr.c_str(), id.c_str(), dims_repr.c_str());
  schedule_ = isl::map(domain_.ctx(), repr);

  // set dimension names
  for (int i = 0; i < dims.size(); i++) {
    schedule_ = isl::manage(isl_map_set_dim_name(schedule_.release(), isl_dim_in, i, dims[i].c_str()));
    schedule_ = isl::manage(isl_map_set_dim_name(schedule_.release(), isl_dim_out, i, dims[i].c_str()));
  }
}

Element::Element(isl::set domain) : domain_(domain) {
  CHECK(!domain_.is_null());
  CHECK(!domain_.is_empty());

  InitSchedule();
}

std::tuple<Iterator, Iterator> Element::Split(const Iterator &level, int factor) {
  int offset = isl_set_find_dim_by_name(domain_.get(), isl_dim_set, level.id.c_str());
  CHECK_GE(offset, 0) << "iterator " << level << " not in " << domain_;
  auto dim_names = GetDimNames(schedule_, isl_dim_out);

  VLOG(2) << "domain: " << domain_;
  VLOG(2) << "schedule: " << schedule_;

  auto from_iters = NamesToIterators(dim_names);
  std::vector<Iterator> to_iters;
  std::vector<Condition> conds;
  Iterator inner_iter(InnerName(level.id));
  Iterator outer_iter(OuterName(level.id));
  for (auto &dim : dim_names) {
    if (dim == level.id) {
      to_iters.push_back(outer_iter);
      to_iters.push_back(inner_iter);

      conds.emplace_back(outer_iter,
                         utils::StringFormat("%s=floor(%s/%d)", outer_iter.id.c_str(), level.id.c_str(), factor));
      VLOG(3) << "outer cond: " << conds.back();
      conds.emplace_back(inner_iter,
                         utils::StringFormat("%s=%s %s %d", inner_iter.id.c_str(), level.id.c_str(), "%", factor));

      VLOG(3) << "inner cond: " << conds.back();
    } else {
      to_iters.emplace_back(dim);
    }
  }

  Map transform(domain_.ctx(), id(), from_iters, to_iters, conds, id());
  VLOG(3) << "transform: " << transform.__str__();
  schedule_ = schedule_.apply_range(transform.to_isl());
  auto range_dims =
      utils::Map<std::vector<Iterator>, std::vector<std::string>>(to_iters, [](const Iterator &x) { return x.id; });
  SetDimNames(&schedule_, isl_dim_out, range_dims);

  VLOG(3) << "transform " << transform.to_isl();
  VLOG(3) << "schedule after transform: " << schedule_;

  VLOG(3) << "iterators: " << outer_iter << " " << inner_iter;
  return std::make_tuple(outer_iter, inner_iter);
}

void Element::Reorder(const std::vector<Iterator> &order) {}

std::tuple<Iterator, Iterator, Iterator, Iterator> Element::Tile(const Iterator &level0,
                                                                 const Iterator &level1,
                                                                 int factor0,
                                                                 int factor1) {
  Iterator level0_inner(InnerName(level0));
  Iterator level0_outer(OuterName(level0));
  Iterator level1_inner(InnerName(level1));
  Iterator level1_outer(OuterName(level1));

  return std::make_tuple(level0_outer, level0_inner, level1_outer, level1_inner);
}

std::tuple<Iterator, Iterator> Element::Skew(const Iterator &i, const Iterator &j, int factor) {
  Iterator i_new(i.id + "_skew");
  Iterator j_new(j.id + "_skew");
  return std::make_tuple(i_new, j_new);
}

Iterator Element::Fuse(const Iterator &level0, const Iterator &level1) {
  auto new_name = utils::StringFormat("%s_%s", level0.id.c_str(), level1.id.c_str());
  return Iterator(new_name);
}

std::string InnerName(const std::string &name) { return name + "_inner"; }
std::string OuterName(const std::string &name) { return name + "_outer"; }
std::string InnerName(const Iterator &iterator) { return InnerName(iterator.id); }
std::string OuterName(const Iterator &iterator) { return OuterName(iterator.id); }

const char *Element::id() const { return isl_set_get_tuple_name(domain_.get()); }

std::tuple<Iterator, Iterator> Element::Split(const std::string &level, int factor) {
  return std::move(Split(Iterator(level), factor));
}

}  // namespace poly
}  // namespace cinn
