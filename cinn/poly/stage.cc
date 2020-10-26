#include "cinn/poly/stage.h"

#include <set>
#include <unordered_set>
#include <utility>

#include "cinn/common/axis.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute.h"
#include "cinn/optim/ir_replace.h"
#include "cinn/poly/compute_at_transform.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/utils/functional.h"
#include "cinn/utils/string.h"

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

  auto dims      = isl_get_dim_names(domain_);
  auto dims_repr = utils::Join(dims, ", ");

  auto repr = utils::StringFormat("{ %s[%s] -> %s[%s] }", id.c_str(), dims_repr.c_str(), id.c_str(), dims_repr.c_str());
  transform_ = isl::map(domain_.ctx(), repr);

  // set dimension names
  for (int i = 0; i < dims.size(); i++) {
    transform_ = isl::manage(isl_map_set_dim_name(transform_.release(), isl_dim_in, i, dims[i].c_str()));
    transform_ = isl::manage(isl_map_set_dim_name(transform_.release(), isl_dim_out, i, dims[i].c_str()));
  }
}

Stage::Stage(const isl::set &domain, Expr expr, ir::_Tensor_ *tensor) : domain_(domain), expr_(expr), tensor_(tensor) {
  CHECK(!domain_.is_null());
  CHECK(!domain_.is_empty());
  InitTransform();
}

std::tuple<Iterator, Iterator> Stage::Split(int level, int factor) {
  AssertAxisIsNotLocked(level);
  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto axis_name = dim_names.at(level);
  return Split(axis_name, factor);
}

std::tuple<Iterator, Iterator> Stage::Split(const Iterator &level, int factor) {
  int offset = isl_set_find_dim_by_name(transformed_domain().get(), isl_dim_set, level.id.c_str());
  CHECK_GE(offset, 0) << "iterator " << level << " not in " << domain_;
  AssertAxisIsNotLocked(offset);

  auto dim_names = isl_get_dim_names(transform_, isl_dim_out);

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

      conds.emplace_back(utils::StringFormat("%s=floor(%s/%d)", outer_iter.id.c_str(), level.id.c_str(), factor));
      VLOG(3) << "outer cond: " << conds.back();
      conds.emplace_back(utils::StringFormat("%s=%s %s %d", inner_iter.id.c_str(), level.id.c_str(), "%", factor));

      VLOG(3) << "inner cond: " << conds.back();
    } else {
      to_iters.emplace_back(dim);
    }
  }

  Map transform(domain_.ctx(), id(), from_iters, to_iters, conds, id());
  VLOG(3) << "transform: " << transform.__str__();
  transform_      = transform_.apply_range(transform.to_isl());
  auto range_dims = utils::Map<std::vector<Iterator>, std::string>(to_iters, [](const Iterator &x) { return x.id; });
  isl_set_dim_names(&transform_, isl_dim_out, range_dims);

  VLOG(3) << "transform " << transform.to_isl();
  VLOG(3) << "schedule after transform: " << transform_;
  VLOG(3) << "iterators: " << outer_iter << " " << inner_iter;

  return std::make_tuple(outer_iter, inner_iter);
}

void Stage::Reorder(const std::vector<Iterator> &order) {
  auto in_names = isl_get_dim_names(transform_, isl_dim_out);
  // assert all the iterators in the isl::set.
  std::unordered_set<std::string> in_name_set(in_names.begin(), in_names.end());
  std::set<Iterator> order_set(order.begin(), order.end());

  std::vector<Iterator> range_iters, domain_iters;
  for (auto &o : order) {
    CHECK(in_name_set.count(o.id)) << "Iterator " << o.id << " not int the exsting axis";
  }

  int order_offset = 0;
  for (auto &iter_name : in_names) {
    Iterator iter(iter_name);

    domain_iters.push_back(iter);

    if (order_set.count(iter)) {
      range_iters.push_back(order[order_offset++]);
    } else {
      range_iters.push_back(iter);
    }
  }

  CHECK_EQ(range_iters.size(), in_names.size());

  Map transform(domain().ctx(), id(), domain_iters, range_iters, {}, id());
  VLOG(3) << "reorder transform: " << transform.__str__();
  transform_ = transform_.apply_range(transform.to_isl());
}

std::tuple<Iterator, Iterator, Iterator, Iterator>  //
Stage::Tile(int level0, int level1, int factor0, int factor1) {
  AssertAxisIsNotLocked(level0);
  AssertAxisIsNotLocked(level1);
  Iterator i0(common::axis_name(level0));
  Iterator i1(common::axis_name(level1));
  return Tile(i0, i1, factor0, factor1);
}

std::tuple<Iterator, Iterator, Iterator, Iterator> Stage::Tile(const Iterator &level0,
                                                               const Iterator &level1,
                                                               int factor0,
                                                               int factor1) {
  auto [level0_outer, level0_inner] = Split(level0, factor0);  // NOLINT
  auto [level1_outer, level1_inner] = Split(level1, factor1);  // NOLINT
  return std::make_tuple(level0_outer, level0_inner, level1_outer, level1_inner);
}

void Stage::ComputeAtSchedule(Stage *other, int level, ComputeAtKind kind) {
  // TODO(Superjomn) Check there are data dependency between `self` and `other`, or the `ComputeAt` is meaningless.
  CHECK(other->tensor());
  CHECK(tensor());

  ComputeAtRelation relation;
  relation.stage = other;
  relation.level = level;

  CHECK(relation.IsCompatible(this));
  compute_ats_[other->id()] = relation;

  // Consider the order if provide.
  switch (kind) {
    case kComputeAtBefore:
      other->CtrlDepend(ir::Tensor(tensor()));
      break;
    case kComputeAtAfter:
      CtrlDepend(ir::Tensor(other->tensor()));
      break;
    case kComputeAtAuto:
      // Do nothing.
      break;
  }

  // Lock all the axis.
  for (int i = 0; i < isl_map_dim(transform_.get(), isl_dim_out); i++) {
    LockAxis(i);
  }
}

void Stage::ComputeAt(Stage *other, int level, Stage::ComputeAtKind kind, const std::string &cached_tensor_name) {
  isl::map access;
  isl_map *access_raw{};
  // For cache_read schedule, it will replace the producer tensor with cache in consumer, so replace the tuple name to
  // cache's in access.
  if (cached_tensor_name.empty())
    access_raw = GatherAccesses(other, tensor_->name);
  else
    access_raw = GatherAccesses(other, cached_tensor_name);

  if (!access_raw) {
    LOG(ERROR) << "ComputeAt: " << other->tensor_->name << " has no access to " << tensor_->name << ", skipped it";
    return;
  }

  if (!cached_tensor_name.empty()) {
    access_raw = isl_map_set_tuple_name(access_raw, isl_dim_out, tensor_->name.c_str());
  }
  access     = isl::manage(access_raw);
  access_raw = nullptr;

  ComputeAtTransform transform(domain_, other->domain(), access, transform_, other->transform(), level);
  transform();

  domain_    = transform.adjusted_pdomain();
  transform_ = transform.adjusted_ptransform();

  // set name of the dimensions if not exists, or it will go wrong in the following process.
  domain_    = isl_set_dim_name_if_null(domain_.release(),
                                     [](isl_dim_type dim_type, int i) { return "pp" + std::to_string(i); });
  transform_ = isl_set_dim_name_if_null(transform_.release(), [](isl_dim_type dim_type, int i) {
    return (dim_type == isl_dim_in ? "pi" : "po") + std::to_string(i);
  });

  auto indice_mins = transform.GetAccessesPrecedingIndicesMinAssumingParamsZero();
  std::vector<int> offsets;
  std::transform(indice_mins.begin(), indice_mins.end(), std::back_inserter(offsets), [&](int x) { return -x; });

  other->meta.compute_at_infos.emplace_back(other->tensor_->name,                  // consumer_tensor_name,
                                            tensor_->name,                         // producer_tensor_name
                                            transform.GetProducerAdjustedShape(),  // adjusted_producer_shape,
                                            indice_mins,                           // preceding_offset_for_producer_load
                                            level);

  ComputeAtSchedule(other, level, kind);
}

std::tuple<Iterator, Iterator> Stage::Skew(const Iterator &i, const Iterator &j, int factor) {
  CINN_NOT_IMPLEMENTED
  Iterator i_new(i.id + "_skew");
  Iterator j_new(j.id + "_skew");

  return std::make_tuple(i_new, j_new);
}

Iterator Stage::Fuse(int level0, int level1) {
  AssertAxisIsNotLocked(level0);
  AssertAxisIsNotLocked(level1);
  auto dims = isl_get_dim_names(transformed_domain());
  CHECK_LT(level0, dims.size());
  CHECK_LT(level1, dims.size());

  Iterator iter0(dims[level0]);
  Iterator iter1(dims[level1]);

  return Fuse(iter0, iter1);
}

Iterator Stage::Fuse(const std::string &level0, const std::string &level1) {
  return Fuse(Iterator(level0), Iterator(level1));
}

/*
 * Fuse use a polyhedral transform.
 */
Iterator Stage::Fuse(const Iterator &level0, const Iterator &level1) {
  int offset0 = isl_set_find_dim_by_name(transformed_domain().get(), isl_dim_set, level0.id.c_str());
  int offset1 = isl_set_find_dim_by_name(transformed_domain().get(), isl_dim_set, level1.id.c_str());
  CHECK_EQ(offset1, offset0 + 1) << "level [" << level0.id << "] and level [" << level1.id << "] should be adjancent";
  AssertAxisIsNotLocked(offset0);
  AssertAxisIsNotLocked(offset1);

  auto new_iter_name = utils::StringFormat("%s_%s_fused", level0.id.c_str(), level1.id.c_str());

  // Aff { s[i,j,k] -> [j] } and get the j's max value
  // to apply something like { S[i,j] -> S[k]: k = i+j }
  auto from_dim_names = isl_get_dim_names(transform_, isl_dim_out);
  auto from_iters     = NamesToIterators(from_dim_names);

  Aff aff(domain_.ctx(), id(), from_iters, std::vector<Iterator>({Iterator(level1.id)}), {});

  int level1_max_val = transformed_domain().max_val(aff.to_isl()).get_num_si() + 1;

  // Map { s[i,j,k] -> s[n,k] : n = i * max_val + j }
  std::vector<Iterator> to_iters;
  {
    Iterator new_iter(new_iter_name);
    for (auto &iter : from_iters) {
      if (iter == level0) {
      } else if (iter == level1) {
        to_iters.push_back(new_iter);
      } else {
        to_iters.push_back(iter);
      }
    }
  }

  std::vector<Condition> conds;
  conds.emplace_back(utils::StringFormat(
      "%s = %s * %d + %s", new_iter_name.c_str(), level0.id.c_str(), level1_max_val, level1.id.c_str()));

  Map trans(domain_.ctx(), id(), from_iters, to_iters, conds, id());

  transform_ = transform_.apply_range(trans.to_isl());
  {
    std::vector<std::string> iter_names;
    for (auto &iter : to_iters) iter_names.push_back(iter.id);

    isl_set_dim_names(&transform_, isl_dim_out, iter_names);
  }

  return Iterator(new_iter_name);
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

std::tuple<Iterator, Iterator> Stage::Split(const std::string &level, int factor) {
  return std::move(Split(Iterator(level), factor));
}

Shared<Stage> Stage::New(const isl::set &domain, Expr expr, ir::_Tensor_ *tensor) {
  return new Stage(domain, expr, tensor);
}

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

  // remove parameters, we don't consider them yet
  auto remove_params = [](isl::set &set) {
    int nparams = isl_set_dim(set.get(), isl_dim_param);
    if (nparams > 0) {
      set = isl::manage(isl_set_remove_dims(set.release(), isl_dim_param, 0, nparams));
    }
  };

  remove_params(stage_partial_set);
  remove_params(self_partial_set);

  VLOG(3) << "stage0.partial_set " << stage_partial_set;
  VLOG(3) << "stage1.partial_set " << self_partial_set;
  return isl_set_is_equal(stage_partial_set.get(), self_partial_set.get());
}

void Stage::Vectorize(int level, int factor) {
  AssertAxisIsNotLocked(level);
  CHECK_LT(level, n_out_dims());
  CHECK_GT(factor, 0);
  auto dim_name = ith_dim_name(level);
  vectorize_info_.set(level /*inner*/, factor);
}

void Stage::Vectorize(const std::string &axis, int factor) {
  auto dims = isl_get_dim_names(transformed_domain());
  auto it   = std::find(dims.begin(), dims.end(), axis);
  CHECK(it != dims.end()) << "No dimension called " << axis;
  Vectorize(std::distance(dims.begin(), it), factor);
}

void Stage::Vectorize(const Iterator &axis, int factor) { return Vectorize(axis.id, factor); }

void Stage::Unroll(int level) {
  AssertAxisIsNotLocked(level);
  unroll_info_.insert(level);
}

std::string Stage::ith_dim_name(int level) {
  auto dims = isl_get_dim_names(transformed_domain());
  CHECK_LT(level, dims.size());
  return dims[level];
}

Iterator Stage::ith_iterator(int level) { return Iterator(ith_dim_name(level)); }

isl::set Stage::transformed_domain() const {
  CHECK(!domain_.is_null());
  CHECK(!transform_.is_null());
  return domain_.apply(transform_);
}

std::vector<std::pair<std::string, std::string>> ExtractExtraDepLinksFromStages(const std::vector<Stage *> &stages) {
  std::vector<std::pair<std::string, std::string>> extra_links;
  for (auto &stage : stages) {
    for (auto &tensor : stage->ctrl_depends()) {
      VLOG(1) << "get extra stage: " << tensor->name << " -> " << stage->id();
      extra_links.emplace_back(tensor->name, stage->id());
    }
  }

  return extra_links;
}

void Stage::Unroll(const std::string &level) {
  auto dim_names = axis_names();
  auto it        = std::find(dim_names.begin(), dim_names.end(), level);
  int l          = std::distance(dim_names.begin(), it);
  AssertAxisIsNotLocked(l);
  Unroll(l);
}

void Stage::Unroll(const Iterator &level) {
  auto dim_names = axis_names();
  auto it        = std::find(dim_names.begin(), dim_names.end(), level.id);
  int l          = std::distance(dim_names.begin(), it);
  AssertAxisIsNotLocked(l);
  Unroll(l);
}

std::vector<std::string> Stage::axis_names() const { return isl_get_dim_names(transformed_domain()); }

void Stage::Bind(int level, const std::string &axis) {
  CHECK_LT(level, n_out_dims());
  LockAxis(level);

  if (axis == "threadIdx.x" || axis == "threadIdx.y" || axis == "threadIdx.z") {
    uint8_t offset = axis.back() - 'x';
    AddForloopInfo(level, StageForloopInfo{ir::ForType::GPUThread, DeviceAPI::GPU, offset});
  } else if (axis == "blockIdx.x" || axis == "blockIdx.y" || axis == "blockIdx.z") {
    uint8_t offset = axis.back() - 'x';
    AddForloopInfo(level, StageForloopInfo{ir::ForType::GPUBlock, DeviceAPI::GPU, offset});
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

Iterator Stage::axis(int i) const {
  auto names = axis_names();
  CHECK_LT(i, names.size());
  return Iterator(names[i]);
}
Iterator Stage::axis(const std::string &i) const {
  auto names = axis_names();
  auto it    = std::find(names.begin(), names.end(), i);
  CHECK(it != names.end());
  return Iterator(*it);
}

bool Stage::has_expression() const {
  CHECK(tensor_);
  return tensor_->has_expression();
}

/*
 * To create a read cache:
 * 1. create a cache write stage for cache assign.
 * 2. add extra deps between cache and tensor to keep SSA order
 * 3. register the readers of the cache to the \p tensor, replace latter in Lower
 */
ir::Tensor Stage::CacheRead(const std::string &memory_type, const std::vector<ir::Tensor> &readers, StageMap stages) {
  CHECK(tensor_);
  auto my_tensor         = ir::Tensor(tensor_);
  std::string cache_name = Context::Global().NewName(tensor_->name + "_read_cache");
  VLOG(4) << "cache_name " << cache_name;
  auto cache_tensor = lang::Compute(
      tensor_->shape, [=](const std::vector<Expr> &dims) { return my_tensor(dims); }, cache_name);
  cache_tensor->WithBuffer(memory_type);

  stages->Insert(cache_tensor, CreateStage(cache_tensor).get());

  for (auto &reader : readers) {
    stages[reader]->CtrlDepend(cache_tensor);
  }

  std::vector<std::string> reader_names;
  std::transform(
      readers.begin(), readers.end(), std::back_inserter(reader_names), [](const ir::Tensor &x) { return x->name; });

  CHECK(!meta.read_cache_relation) << "Duplicate read cache found, just one is allowed";
  meta.read_cache_relation.reset(new ReadCacheRelation{cache_name, reader_names});

  if (memory_type == "shared") {
    stages[cache_tensor]->SetScope(ScopeKind::kShared);
  } else if (memory_type == "local") {
    stages[cache_tensor]->SetScope(ScopeKind::kLocal);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  return cache_tensor;
}

/*
 * Replace the tensor's name to cache_name, and create a cache_stage to copy content from cache to original tensor.
 */
ir::Tensor Stage::CacheWrite(const std::string &memory_type, StageMap stages) {
  CHECK(tensor_);
  CHECK(!tensor_->buffer.defined()) << "This tensor is already binded to a buffer, cannot cache write";
  CHECK(!meta.compute_inline) << "Cannot create a write cache on an inlined tensor";
  auto my_tensor         = ir::Tensor(tensor_);
  std::string cache_name = Context::Global().NewName(tensor_->name + "_cache_write_out");
  // make my_tensor a cache
  my_tensor->WithBuffer(memory_type);

  auto write_stage = lang::Compute(
      tensor_->shape, [=](const std::vector<Expr> &dims) { return my_tensor(dims); }, cache_name);
  stages->Insert(write_stage, CreateStage(write_stage).get());

  stages[write_stage]->CtrlDepend(my_tensor);

  CHECK(!meta.write_cache_relation) << "Duplicate write cache found, just one is allowed";
  meta.write_cache_relation.reset(new WriteCacheRelation{cache_name});

  return write_stage;
}

void Stage::ComputeInline() {
  CHECK(tensor_);
  meta.compute_inline = true;
}

void Stage::ShareBufferWith(Stage *other) {
  CHECK(tensor_);
  CHECK(!other->meta.compute_inline);
  CHECK(!meta.compute_inline);

  meta.tensors_to_share_buffer_with.insert(other->id());
  other->meta.tensors_to_share_buffer_with.insert(tensor_->name);
}

isl_map *__isl_give GatherAccesses(Stage *stage, const std::string &tensor_name) {
  CHECK(stage->tensor_);
  auto loads = ir::CollectIRNodes(stage->tensor_->body(), [&](const Expr *x) {
    return x->As<ir::Load>() && x->As<ir::Load>()->tensor.as_tensor()->name == tensor_name;
  });

  auto vars = stage->tensor_->axis_with_reduce();

  std::string in_tuple_name  = stage->tensor_->name;
  std::string out_tuple_name = tensor_name;
  std::vector<std::string> in_dim_names, out_loads;
  std::transform(vars.begin(), vars.end(), std::back_inserter(in_dim_names), [](const Var &x) { return x->name; });
  std::transform(
      loads.begin(), loads.end(), std::back_inserter(out_loads), [](const Expr &x) { return utils::GetStreamCnt(x); });

  isl_map *res = nullptr;
  for (auto &load : out_loads) {
    std::string repr = utils::StringFormat(
        "{ %s[%s] -> %s }", in_tuple_name.c_str(), utils::Join(in_dim_names, ",").c_str(), load.c_str());
    isl_map *access = isl_map_read_from_str(stage->domain().ctx().get(), repr.c_str());
    if (res) {
      res = isl_map_union(res, access);
    } else {
      res = access;
    }
  }

  return res;
}

void Stage::AddForloopInfo(int level, const StageForloopInfo &info) {
  int num_levels = isl_map_dim(transform_.get(), isl_dim_out);
  CHECK_LT(level, num_levels);
  forloop_infos_[level] = info;
}

void Stage::LockAxis(uint32_t level) {
  CHECK_LT(level, n_out_dims()) << "axis level out of range";
  locked_axis_.insert(level);
}

void Stage::UnlockAxis(uint32_t level) {
  CHECK_LT(level, n_out_dims()) << "axis level out of range";
  locked_axis_.erase(level);
}

bool Stage::is_axis_locked(uint32_t level) const {
  CHECK_LT(level, n_out_dims()) << "axis level out of range";
  return locked_axis_.count(level);
}

void Stage::AssertAxisIsNotLocked(uint32_t level) {
  CHECK(!is_axis_locked(level)) << "The " << level << "-th axis is locked, cannot perform schedule";
}

int Stage::GetTransformedLevel(int level) {
  if (!compute_ats().empty()) {
    // The ComputeAt schedule will insert some consumer axis in the preceding of this, so the raw before ComputeAt
    // should add the numbers of axis inserted.
    CHECK_EQ(compute_ats().size(), 1UL);
    auto &compute_at = compute_ats().front();
    return compute_at.level + level + 1;
  }

  // or just return the original.
  return level;
}

void Stage::CtrlDepend(const ir::Tensor &t) { ctrl_depends_.insert(t); }

const std::set<ir::Tensor> &Stage::ctrl_depends() const { return ctrl_depends_; }

ir::Tensor Stage::LookupCtrlDepend(const std::string &tensor_name) const {
  auto it = std::find_if(
      ctrl_depends_.begin(), ctrl_depends_.end(), [&](const ir::Tensor &x) { return x->name == tensor_name; });
  if (it == ctrl_depends_.end()) return ir::Tensor();
  return *it;
}

Stage *_StageMap_::operator[](const ir::Tensor &tensor) {
  CHECK(data_.count(tensor->name)) << "StageMap has no stage for tensor [" << tensor->name << "]";
  return data_[tensor->name].get();
}
const Stage *_StageMap_::operator[](const ir::Tensor &tensor) const {
  CHECK(data_.count(tensor->name));
  return data_.at(tensor->name).get();
}
Stage *_StageMap_::operator[](const ir::_Tensor_ *tensor) {
  CHECK(data_.count(tensor->name)) << "StageMap has no stage for tensor [" << tensor->name << "]";
  return data_[tensor->name].get();
}
const Stage *_StageMap_::operator[](const ir::_Tensor_ *tensor) const {
  CHECK(data_.count(tensor->name)) << "StageMap has no stage for tensor [" << tensor->name << "]";
  return data_.at(tensor->name).get();
}

Stage *_StageMap_::Insert(const ir::Tensor &key, Stage *stage) {
  CHECK(stage);
  data_[key->name].Reset(stage);
  return stage;
}

Stage *_StageMap_::InsertLazily(const ir::Tensor &key) {
  if (data_.count(key->name)) return operator[](key);
  return Insert(key, ir::CreateStage(key).get());
}

StageMap CreateStages(const std::vector<ir::Tensor> &tensors) {
  StageMap stages;

  std::set<ir::Tensor> all_tensors(tensors.begin(), tensors.end());

  for (auto &tensor : tensors) {
    auto used_tensors = ir::CollectIRNodes(tensor->body(), [](const Expr *x) { return x->as_tensor(); });
    for (const Expr &x : used_tensors) {
      all_tensors.insert(x.as_tensor_ref());
    }
  }

  for (auto &t : all_tensors) {
    stages->Insert(t, ir::CreateStage(t).get());
  }

  return stages;
}

Stage *_StageMap_::Lookup(const std::string &name) const {
  auto it = data_.find(name);
  if (it == data_.end()) return nullptr;
  return it->second.get();
}

}  // namespace poly
}  // namespace cinn
