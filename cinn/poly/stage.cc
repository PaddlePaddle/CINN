#include "cinn/poly/stage.h"

#include <set>
#include <utility>

#include "cinn/common/axis.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
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

void Stage::InitTransform() {
  std::string id = isl_set_get_tuple_name(domain_.get());

  auto dims      = GetDimNames(domain_);
  auto dims_repr = utils::Join(dims, ", ");

  auto repr = utils::StringFormat("{ %s[%s] -> %s[%s] }", id.c_str(), dims_repr.c_str(), id.c_str(), dims_repr.c_str());
  transform_ = isl::map(domain_.ctx(), repr);

  // set dimension names
  for (int i = 0; i < dims.size(); i++) {
    transform_ = isl::manage(isl_map_set_dim_name(transform_.release(), isl_dim_in, i, dims[i].c_str()));
    transform_ = isl::manage(isl_map_set_dim_name(transform_.release(), isl_dim_out, i, dims[i].c_str()));
  }
}

Stage::Stage(const isl::set &domain, Expr expr) : domain_(domain), expr_(expr) {
  CHECK(!domain_.is_null());
  CHECK(!domain_.is_empty());
  InitTransform();
}

std::tuple<Iterator, Iterator> Stage::Split(const Iterator &level, int factor, SplitRestStrategy strategy) {
  int offset = isl_set_find_dim_by_name(transformed_domain().get(), isl_dim_set, level.id.c_str());
  CHECK_GE(offset, 0) << "iterator " << level << " not in " << domain_;
  auto dim_names = GetDimNames(transform_, isl_dim_out);

  VLOG(2) << "domain: " << domain_;
  VLOG(2) << "schedule: " << transform_;

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
  transform_      = transform_.apply_range(transform.to_isl());
  auto range_dims = utils::Map<std::vector<Iterator>, std::string>(to_iters, [](const Iterator &x) { return x.id; });
  SetDimNames(&transform_, isl_dim_out, range_dims);

  VLOG(3) << "transform " << transform.to_isl();
  VLOG(3) << "schedule after transform: " << transform_;

  VLOG(3) << "iterators: " << outer_iter << " " << inner_iter;

  split_strageties_[inner_iter.id] = strategy;

  return std::make_tuple(outer_iter, inner_iter);
}

void Stage::Reorder(const std::vector<Iterator> &order) {
  auto in_names = GetDimNames(transform_, isl_dim_out);
  CHECK_EQ(order.size(), in_names.size());
  auto in_iters =
      utils::Map<std::vector<std::string>, Iterator>(in_names, [](const std::string &x) { return Iterator(x); });

  Map transform(domain().ctx(), id(), in_iters, order, {}, id());
  VLOG(3) << "reorder transform: " << transform.__str__();
  transform_ = transform_.apply_range(transform.to_isl());
}

std::tuple<Iterator, Iterator, Iterator, Iterator>  //
Stage::Tile(int level0, int level1, int factor0, int factor1) {
  Iterator i0(common::axis_name(level0));
  Iterator i1(common::axis_name(level1));
  return Tile(i0, i1, factor0, factor1);
}

std::tuple<Iterator, Iterator, Iterator, Iterator> Stage::Tile(const Iterator &level0,
                                                               const Iterator &level1,
                                                               int factor0,
                                                               int factor1) {
  Iterator level0_inner, level0_outer;
  Iterator level1_inner, level1_outer;

  std::tie(level0_outer, level0_inner) = Split(level0, factor0);
  std::tie(level1_outer, level1_inner) = Split(level1, factor1);
  return std::make_tuple(level0_outer, level0_inner, level1_outer, level1_inner);
}

void Stage::ComputeAt(Stage *other, int level) {
  // check duplicate
  CHECK(!compute_ats_.count(other->id())) << "duplicate set compute_at the same stage";

  // TODO(Superjomn) Check there are data dependency between `self` and `other`, or the `ComputeAt` is meaningless.

  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;

  CHECK(relation.IsCompatible(this));
  compute_ats_[other->id()] = relation;
}

std::tuple<Iterator, Iterator> Stage::Skew(const Iterator &i, const Iterator &j, int factor) {
  Iterator i_new(i.id + "_skew");
  Iterator j_new(j.id + "_skew");
  return std::make_tuple(i_new, j_new);
}

Iterator Stage::Fuse(const Iterator &level0, const Iterator &level1) {
  auto new_name = utils::StringFormat("%s_%s", level0.id.c_str(), level1.id.c_str());
  return Iterator(new_name);
}

std::vector<std::string> Stage::input_statements() const {
  if (!expr_.defined()) return {};
  VLOG(3) << "stage " << id() << " expr: " << expr_;
  auto load_exprs = ir::CollectIRNodes(expr_, [](const Expr *x) { return x->As<ir::Load>(); });
  std::set<std::string> statements;
  for (auto &expr : load_exprs) {
    auto *load_node = expr.As<ir::Load>();
    CHECK(load_node);
    auto *tensor = load_node->tensor.As<ir::_Tensor_>();
    CHECK(tensor);
    auto tensor_name = tensor->name;
    if (tensor_name != id()) statements.insert(tensor_name);
  }
  return std::vector<std::string>(statements.begin(), statements.end());
}

std::string InnerName(const std::string &name) { return name + "_inner"; }
std::string OuterName(const std::string &name) { return name + "_outer"; }
std::string InnerName(const Iterator &iterator) { return InnerName(iterator.id); }
std::string OuterName(const Iterator &iterator) { return OuterName(iterator.id); }

const char *Stage::id() const { return isl_set_get_tuple_name(domain_.get()); }

std::tuple<Iterator, Iterator> Stage::Split(const std::string &level, int factor, SplitRestStrategy strategy) {
  return std::move(Split(Iterator(level), factor, strategy));
}

Shared<Stage> Stage::New(const isl::set &domain, Expr expr) { return new Stage(domain, expr); }

std::vector<ComputeAtRelation> Stage::compute_ats() const {
  std::vector<ComputeAtRelation> xs;
  for (auto &item : compute_ats_) xs.push_back(item.second);
  return xs;
}

bool ComputeAtRelation::IsCompatible(Stage *self) {
  CHECK_GE(level, 0);
  CHECK(!self->domain().is_null());
  CHECK(!stage->domain().is_null());

  CHECK_LE(level, isl_set_dim(self->transformed_domain().get(), isl_dim_set));
  CHECK_LE(level, isl_set_dim(stage->transformed_domain().get(), isl_dim_set));

  std::vector<int> selected_dims;
  for (int i = 0; i <= level; i++) {
    selected_dims.push_back(i);
  }

  auto stage_partial_set = SetGetDims(stage->transformed_domain(), selected_dims);
  auto self_partial_set  = SetGetDims(self->transformed_domain(), selected_dims);

  stage_partial_set = isl::manage(isl_set_set_tuple_name(stage_partial_set.release(), ""));
  self_partial_set  = isl::manage(isl_set_set_tuple_name(self_partial_set.release(), ""));
  VLOG(3) << "stage0.partial_set " << stage_partial_set;
  VLOG(3) << "stage1.partial_set " << self_partial_set;

  return isl_set_is_equal(stage_partial_set.get(), self_partial_set.get());
}

void Stage::Vectorize(int level, int factor) {
  CHECK_GT(factor, 0);
  auto dim_name = ith_dim_name(level);
  Split(dim_name, factor);
  vectorize_info_.set(level, factor);
}

std::string Stage::ith_dim_name(int level) {
  auto dims = GetDimNames(transformed_domain());
  CHECK_LT(level, dims.size());
  return dims[level];
}

}  // namespace poly
}  // namespace cinn
