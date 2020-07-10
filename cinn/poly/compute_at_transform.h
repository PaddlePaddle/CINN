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

/**
 * Given two sets and a level, return a isl map as transformation.
 *
 * e.g.
 * s0: A(i,j) : 0<i,j<100
 * s1: B(a,b,c) = A(a,b) + A(a-1,b) + A(a+1,b) : 0 < a,b,c < 10
 *
 * make s0 compute_at s1 at level 0, that should gain some code like
 *
 * for (a = 0; a < 10; a++) {
 *   for (b = 0; b < 10; b++) {
 *     for (k = 0; k < 3; k++)
 *       A(k,b)
 *       // A's shape is [3, 10], for a is \in (a-1, a+1), shift with -1 then get 3, and b \in [0, 10)
 *   }
 *   for (b = 0; b < 10; b++) {
 *     for (c = 0; c < 10; c++) {
 *       B(0,b,c)
 *     }
 *   }
 * }
 *
 * make s0 compute_at s1 at level 1, that should gain some code like
 *
 * for (a = 0; a < 10; a++) {
 *   for (b = 0; b < 10; b++) {
 *     for (k = 0; k < 3; k++)
 *       A(a,b)
 *   }
 *   for (b = 0; b < 10; b++) {
 *     for (c = 0; c < 10; c++) {
 *       B(0,b,c)
 *     }
 *   }
 * }
 *
 * The procedure includes
 * 1. generate the transformation for s0
 * 2. calculate the new buffer size
 * 3. get the offset need in s1 for each axis in s0's axis
 *
 */
class ComputeAtTransform {
 public:
  //! Consumer ranges.
  std::vector<std::pair<int, int>> ranges;

  /**
   * constructor.
   * @param pdomain producer domain.
   * @param cdomain consumer domain, note that, this should be the transformed domain.
   * @param ptransform the existing producer transform.
   * @param level the level of \p cdomain the computation to place on.
   */
  ComputeAtTransform(
      isl::set pdomain, isl::set cdomain, const std::vector<isl::map>& accesses, isl::map ptransform, int level);

  const isl::set& adjusted_pdomain() const { return adjusted_pdomain_; }
  const isl::map& adjusted_ptransform() const { return adjusted_ptransform_; }
  const std::vector<int>& offsets() const { return offsets_; }

 private:
  //! Compute the new producer domain.
  void ComputeAdjustedProducerDomain();

  //! Compute the new producer map.
  void ComputeAdjustedPorducerTransform();

  //! Get the union of the access map.
  isl::map GetAccessUnion();

 private:
  isl::set pdomain_;
  isl::set cdomain_;
  isl::map ptransform_;
  std::vector<isl::map> accesses_;
  const int level_;

  std::vector<int> offsets_;

  //! the adjusted producer domain that will futher replace the original producer domain.
  isl::set adjusted_pdomain_;
  //! The adjusted producer transform that will futher replace the producer transform.
  isl::map adjusted_ptransform_;
};

}  // namespace poly
}  // namespace cinn
