//! This file implements the class ComputeAtTransform, which help to perform the isl transformation in `compute_at`
//! optimization.
#pragma once
#include <isl/constraint.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/map.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

//! Help to mark the consumer parameters in the generated AST.
static const char* kConsumerParamPrefix = "_cp_";

/**
 * Generate a consumer parameter name.
 * @param tuple The tuple name of the consumer set.
 * @param id The id of the parameter.
 * @return the name.
 */
std::string GenConsumerParamName(const char* tuple, int id);

class ComputeAtTransform {
 public:
  ComputeAtTransform(
      isl::set pdomain, isl::set cdomain, isl::map access, isl::map ptransform, isl::map ctransform, int level);

  void operator()() {
    AdjustPdomain();
    AdjustPtransform();
  }

  const isl::set& adjusted_pdomain() const { return adjusted_pdomain_; }
  const isl::map& adjusted_ptransform() const { return adjusted_ptransform_; }

  //! Display C code
  void DisplayC(isl_map* __isl_give pschedule = nullptr, isl_map* __isl_give cschedule = nullptr);

 protected:
  isl_set* __isl_give AddParamsTo(isl_set* __isl_take set);
  isl_map* __isl_give AddParamsTo(isl_map* __isl_take map);

  const char* ptuple() const { return isl_set_get_tuple_name(pdomain_.get()); }
  const char* ctuple() const { return isl_set_get_tuple_name(cdomain_.get()); }

  void AdjustPdomain();

  void AdjustPtransform();

  isl::map ctransform_with_params();
  isl::set cdomain_with_params();

 private:
  isl::set pdomain_;
  isl::set cdomain_;
  isl::map access_;
  isl::map ptransform_;
  isl::map ctransform_;

  isl::set adjusted_pdomain_;
  isl::map adjusted_ptransform_;
  isl::set adjusted_cdomain_;
  isl::map adjusted_ctransform_;

  int level_;
};

}  // namespace poly
}  // namespace cinn
