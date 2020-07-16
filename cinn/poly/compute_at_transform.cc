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
  // TODO(Superjomn) To support the isl parameter.

  isl::set pdomain = pdomain_;

  isl::map access_union = GetAccessUnion();
  VLOG(3) << "access_union: " << access_union;

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

  // make the pdomain just the consume domain, for it is unnecessary to compute the whole producer domain, just the
  // space where consumer consumes
  adjusted_pdomain_ = cdomain_.apply(access_union);
  LOG(INFO) << "adjusted_pdomain: " << adjusted_pdomain_;
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

void ComputeAtTransform2::AdjustPdomain() {
  isl::map ct_with_params = ctransform_with_params();
  isl::set ct_domain      = ct_with_params.domain();

  isl::set cdomain1 = isl::manage(AddParamsTo(cdomain_.copy()));

  LOG(INFO) << "ct_domain: " << ct_domain.space();
  LOG(INFO) << "cdomain1: " << cdomain1.space();

  ct_domain = ct_domain.intersect(cdomain1);
  LOG(INFO) << "ct_domain: " << ct_domain;

  // get producer domain from access
  isl::map access_with_params = isl::manage(AddParamsTo(access_.copy()));

  isl::set pdomain = ct_domain.apply(access_with_params);

  // intect with the original producer domain
  auto pdomain_params = isl::manage(AddParamsTo(pdomain_.copy()));
  adjusted_pdomain_   = isl::manage(isl_set_intersect(pdomain.release(), pdomain_params.release()));
  adjusted_pdomain_   = isl::manage(isl_simplify(adjusted_pdomain_.release()));
  LOG(INFO) << "adjusted pdomain: " << adjusted_pdomain_;
}

void ComputeAtTransform2::AdjustPtransform() {
  // insert level+1 dims from ctransform's range into ptransform's range

  {
    // insert empty dims to ptransform's range
    adjusted_ptransform_ = ptransform_;
    adjusted_ptransform_ = isl::manage(isl_map_insert_dims(adjusted_ptransform_.release(), isl_dim_out, 0, level_ + 1));

    // update the tuple name
    adjusted_ptransform_ = isl::manage(isl_map_set_tuple_name(adjusted_ptransform_.release(), isl_dim_in, ptuple()));
    adjusted_ptransform_ = isl::manage(isl_map_set_tuple_name(adjusted_ptransform_.release(), isl_dim_out, ptuple()));
  }

  {
    // make ctransform range the same space with ptransform's range so that we can copy the dims
    isl::set ct_range  = cdomain_.apply(ctransform_);
    isl::set ct_range1 = isl::manage(isl_set_project_out(
        ct_range.release(), isl_dim_set, level_ + 1, isl_set_dim(ct_range.get(), isl_dim_set) - level_ - 1));
    ct_range1          = isl::manage(isl_set_add_dims(
        ct_range1.release(), isl_dim_set, isl_map_dim(adjusted_ptransform_.get(), isl_dim_out) - level_ - 1));
    // set as the producer's tuple to make a same space
    ct_range1 = isl::manage(isl_set_set_tuple_name(ct_range1.release(), ptuple()));

    adjusted_ptransform_ = adjusted_ptransform_.intersect_range(ct_range1);
    LOG(INFO) << "adjusted_ptransform: " << adjusted_ptransform_;
  }

  {  // add params
    adjusted_ptransform_ = isl::manage(AddParamsTo(adjusted_ptransform_.release()));
  }
}

isl::set ComputeAtTransform2::cdomain_with_params() {
  // add level+1 param to consumer transform
  isl::set cd_with_params = isl::manage(isl_set_add_dims(cdomain_.copy(), isl_dim_param, level_ + 1));
  return cd_with_params;
}

isl::map ComputeAtTransform2::ctransform_with_params() {
  // add level+1 param to consumer transform
  int num_existing_param  = isl_map_dim(ctransform_.get(), isl_dim_param);
  isl::map ct_with_params = isl::manage(AddParamsTo(ctransform_.copy()));
  {
    isl_local_space* local_space = isl_local_space_from_space(ct_with_params.space().release());
    for (int i = 0; i < level_ + 1; i++) {
      isl_constraint* cst = isl_constraint_alloc_equality(isl_local_space_copy(local_space));
      cst                 = isl_constraint_set_coefficient_val(
          cst, isl_dim_param, num_existing_param + i, isl_val_int_from_si(ctransform_.ctx().get(), -1));
      cst = isl_constraint_set_coefficient_val(cst, isl_dim_out, i, isl_val_int_from_si(ctransform_.ctx().get(), 1));
      ct_with_params = isl::manage(isl_map_add_constraint(ct_with_params.release(), cst));
    }
    isl_local_space_free(local_space);
  }
  return ct_with_params;
}

void ComputeAtTransform2::DisplayC(isl_map* pschedule, isl_map* cschedule) {
  LOG(INFO) << "adjusted cdomain: " << adjusted_cdomain_;
  LOG(INFO) << "adjusted ctransform: " << adjusted_ctransform_;

  auto adjusted_ctransform = adjusted_ctransform_;
  auto adjusted_ptransform = adjusted_ptransform_;

  if (cschedule) {
    adjusted_ctransform = isl::manage(isl_map_apply_range(adjusted_ctransform.release(), cschedule));
  }
  if (pschedule) {
    adjusted_ptransform = isl::manage(isl_map_apply_range(adjusted_ptransform.release(), pschedule));
  }

  auto whole_domain = isl::manage(isl_union_set_from_set(adjusted_pdomain_.copy()));
  whole_domain      = isl::manage(isl_union_set_add_set(whole_domain.release(), adjusted_cdomain_.copy()));
  LOG(INFO) << "whole domain: " << whole_domain;

  auto whole_schedule = isl::manage(isl_union_map_from_map(adjusted_ptransform.copy()));
  whole_schedule      = isl::manage(isl_union_map_add_map(whole_schedule.release(), adjusted_ctransform.copy()));
  LOG(INFO) << "whole_schedule: " << whole_schedule;

  isl::set context(whole_domain.ctx(), "{:}");

  auto intersect_schedule = whole_schedule.intersect_domain(whole_domain);

  auto* build = isl_ast_build_from_context(context.release());
  auto* node  = isl_ast_build_node_from_schedule_map(build, intersect_schedule.release());

  LOG(INFO) << "code:\n\n" << isl_ast_node_to_C_str(node);

  isl_ast_node_free(node);
}

isl_set* ComputeAtTransform2::AddParamsTo(isl_set* set) {
  int existing_params = isl_set_dim(set, isl_dim_param);
  set                 = isl_set_add_dims(set, isl_dim_param, level_ + 1);

  // set name
  for (int i = 0; i < level_ + 1; i++) {
    std::string pname = utils::StringFormat("_%s_%d", ctuple(), i);
    set               = isl_set_set_dim_name(set, isl_dim_param, existing_params + i, pname.c_str());
  }
  return set;
}

isl_map* ComputeAtTransform2::AddParamsTo(isl_map* map) {
  int existing_params = isl_map_dim(map, isl_dim_param);
  map                 = isl_map_add_dims(map, isl_dim_param, level_ + 1);

  // set name
  for (int i = 0; i < level_ + 1; i++) {
    std::string pname = utils::StringFormat("_%s_%d", ctuple(), i);
    map               = isl_map_set_dim_name(map, isl_dim_param, existing_params + i, pname.c_str());
  }
  return map;
}

ComputeAtTransform2::ComputeAtTransform2(
    isl::set pdomain, isl::set cdomain, isl::map access, isl::map ptransform, isl::map ctransform, int level)
    : pdomain_(pdomain),
      cdomain_(cdomain),
      access_(access),
      ptransform_(ptransform),
      ctransform_(ctransform),
      level_(level) {
  LOG(INFO) << "pdomain: " << pdomain;
  LOG(INFO) << "ptransform: " << ptransform;
  LOG(INFO) << "cdomain: " << cdomain;
  LOG(INFO) << "ctransform: " << ctransform;

  adjusted_ctransform_ = isl::manage(AddParamsTo(ctransform_.copy()));
  adjusted_cdomain_    = isl::manage(AddParamsTo(cdomain_.copy()));
}

}  // namespace poly
}  // namespace cinn
