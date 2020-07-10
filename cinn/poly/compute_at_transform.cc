#include "cinn/poly/compute_at_transform.h"

namespace cinn {
namespace poly {

ComputeAtTransform::ComputeAtTransform(
    isl::set pdomain, isl::set cdomain, const std::vector<isl::map>& accesses, isl::map ptransform, int level)
    : pdomain_(pdomain), cdomain_(cdomain), accesses_(accesses), ptransform_(ptransform), level_(level) {
  int p_num_dims = isl_set_dim(pdomain_.get(), isl_dim_set);
  CHECK_LT(level, p_num_dims) << "level out of range";

  ComputeAdjustedProducerDomain();
  ComputeAdjustedPorducerTransform();
}

void ComputeAtTransform::ComputeAdjustedProducerDomain() {
  // input the accesses, domain, get the adjusted domain
  // the previous level_+1 dimensions are fixed, we need to assume the dimension is zero, and get the range of the new
  // domain, to make a compact buffer (e.g, the domain is [100, 110], the buffer size will be 10, not 110, the new
  // domain will be [0, 10]).
  // TODO(Superjomn) To support the isl parameter.

  isl::set pdomain = pdomain_;

  isl::map access_union = GetAccessUnion();
  VLOG(3) << "access_union: " << access_union;

  // make the precending dim = 0, and get the domain
  isl::set zero_limit;
  {
    auto* tuple    = isl_set_get_tuple_name(cdomain_.get());
    auto dim_names = GetDimNames(cdomain_.get());

    std::vector<std::string> conds;
    for (int i = 0; i < level_ + 1; i++) {
      auto [pmin_v, pmax_v] = isl_set_get_axis_range(pdomain.get(), i);
      conds.push_back(utils::StringFormat("%s = %d", dim_names[i].c_str(), pmin_v.num_si()));
    }

    std::string repr = utils::StringFormat(
        "{ %s[%s]: %s }", tuple, utils::Join(dim_names, ",").c_str(), utils::Join(conds, " and ").c_str());
    zero_limit = isl::manage(isl_set_read_from_str(pdomain.ctx().get(), repr.c_str()));
  }
  zero_limit = zero_limit.apply(access_union);
  /*
  for (i = 0; i < 100; i++) {
    for (j = 0; j < 100; j++) {
      for (j = 0; j < 3; j++) {
        AC(0,j);
      }
      B(i,j) = AC(0,0-1+1) + AC(0,1) + AC(0,1+1);
    }
  }
  */

  offsets_.clear();
  for (int i = 0; i < level_ + 1; i++) {
    auto [min_v, max_v] = isl_set_get_axis_range(zero_limit.get(), i);
    int offset          = -min_v.num_si();
    ranges.emplace_back(min_v.num_si(), max_v.num_si());
    offsets_.push_back(offset);
    VLOG(3) << "axis " << i << ": offset " << offset << " range: " << ranges.back().first << " "
            << ranges.back().second;
  }

  // reconstruct the pdomain with the new range
  isl::set preceending_limit;
  {
    // { s0[p0,p1,a,b] -> s0[p0,p1,a,b]: min+offset <= p0 <= max+offset, p1... }
    auto* tuple = isl_set_get_tuple_name(pdomain.get());
    auto dims   = GetDimNames(pdomain.get());
    std::vector<std::string> conds;
    for (int i = 0; i < level_ + 1; i++) {
      conds.push_back(utils::StringFormat(
          "%d <= %s <= %d", ranges[i].first + offsets_[i], dims[i].c_str(), ranges[i].second + offsets_[i]));
    }
    isl::map t(pdomain_.ctx(),
               utils::StringFormat("{ %s[%s] -> %s[%s]: %s }",
                                   tuple,
                                   utils::Join(dims, ",").c_str(),
                                   tuple,
                                   utils::Join(dims, ",").c_str(),
                                   utils::Join(conds, " and ").c_str()));
    VLOG(3) << "t " << t;

    preceending_limit = pdomain_.apply(t);
    for (int i = 0; i < isl_set_dim(preceending_limit.get(), isl_dim_set); i++) {
      preceending_limit =
          isl::manage(isl_set_set_dim_name(preceending_limit.release(), isl_dim_set, i, dims[i].c_str()));
    }
    pdomain = preceending_limit.intersect(pdomain);
    VLOG(3) << "applied pdomain: " << pdomain;
    adjusted_pdomain_ = pdomain;
  }

  {
    // intersect the adjusted_pdomain with cdomain
    isl::map access_union   = GetAccessUnion();
    isl::set consume_domain = cdomain_.apply(access_union);
    VLOG(3) << "consume domain: " << consume_domain;
    adjusted_pdomain_ = adjusted_pdomain().intersect(consume_domain);
    VLOG(3) << "*adjusted pdomain: " << adjusted_pdomain_;
  }
}

void ComputeAtTransform::ComputeAdjustedPorducerTransform() {
  // { s0[i,j] -> s0[p0,p1,i,j] }
  auto ptransform = ptransform_;

  // insert the consumer dims to ptransform's range
  ptransform = isl::manage(isl_map_insert_dims(ptransform.release(), isl_dim_out, 0, level_ + 1));
  for (int i = 0; i < level_ + 1; i++) {
    std::string dim_name = "_p" + std::to_string(i);
    ptransform           = isl::manage(isl_map_set_dim_name(ptransform.release(), isl_dim_out, i, dim_name.c_str()));
  }
  ptransform = isl::manage(isl_map_set_tuple_name(
      ptransform.release(), isl_dim_out, isl_map_get_tuple_name(ptransform_.get(), isl_dim_out)));
  VLOG(3) << "ptransform: " << ptransform;

  int range_dim = isl_map_dim(ptransform.get(), isl_dim_out);

  isl::set cdomain = cdomain_;
  cdomain          = isl::manage(isl_set_project_out(
      cdomain.release(), isl_dim_set, level_ + 1, isl_set_dim(cdomain.get(), isl_dim_set) - level_ - 1));
  // add dims with none limits.
  cdomain = isl::manage(isl_set_insert_dims(cdomain.release(),
                                            isl_dim_set,
                                            isl_set_dim(cdomain.get(), isl_dim_set),
                                            range_dim - isl_set_dim(cdomain.get(), isl_dim_set)));
  cdomain =
      isl::manage(isl_set_set_tuple_name(cdomain.release(), isl_map_get_tuple_name(ptransform.get(), isl_dim_out)));

  VLOG(3) << "cdomain: " << cdomain;
  VLOG(3) << "ptransform: " << ptransform;
  ptransform = ptransform.intersect_range(cdomain);
  VLOG(3) << "ptransform: " << ptransform;
  adjusted_ptransform_ = ptransform;
}

isl::map ComputeAtTransform::GetAccessUnion() {
  isl::map access_union;

  std::vector<isl::set> access_domains;
  for (auto& access : accesses_) {
    if (!access_union.get()) {
      access_union = access;
    } else {
      access_union = isl::manage(isl_map_union(access_union.release(), access.copy()));
    }
  }

  return access_union;
}

}  // namespace poly
}  // namespace cinn
