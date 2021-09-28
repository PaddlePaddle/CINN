#pragma once
#include <functional>
#include <string>
#include <absl/container/flat_hash_map.h>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace common {

/**
 * Interval of a _Var_.
 */
struct CasInterval {
  template <typename T>
  CasInterval(T l, T r) : l(l), r(r) {
    CHECK_LE(l, r) << "left shoud not be larger than right";
  }
  CasInterval(Expr e_l, Expr e_r) : e_l(e_l), e_r(e_r) {}
  int l, r;
  // Note: not verify l <= r and (e_l, e_r) has higher priority than (l, r)
  Expr e_l, e_r;

  friend std::ostream& operator<<(std::ostream& os, const CasInterval& i) {
    if (i.e_l.defined() && i.e_r.defined()) {
      os << "Interval[" << i.e_l << ", " << i.e_r << "]";
    } else {
      os << "Interval[" << i.l << ", " << i.r << "]";
    }
    return os;
  }
};

using cas_intervals_t = absl::flat_hash_map<std::string, CasInterval>;

Expr AutoSimplify(Expr u, const absl::flat_hash_map<std::string, CasInterval>& var_intervals = {});

//! Simplify a CAS expression.
Expr CasSimplify(Expr u, const absl::flat_hash_map<std::string, CasInterval>& var_intervals = {});

/**
 * \brief Solve an equality.
 * Currently this is an naive implementation using the GiNaC.
 *
 * @param inequality The inequality expression containing an LE or LT or GT or GE, such as 2x-1<3
 * @param val The target variable.
 * @return an copied expression looks like x < 100.
 */
Expr SolveInequality(Expr inequality, Var val);
Expr SolveInequalityInt(Expr inequality, Var val);

namespace detail {

//! Whether to treat this expression as a symbol. e.g. Load, Min, Max are treated as symbol to avoid confusing the CAS.
bool CASasSymbol(Expr expr);
//! Convert some nodes to CAS representation, e.g. convert Mul, Add to Product and Sum.
Expr ConvertCinnToCAS(Expr expr);
//! Convert the CAS representation to CINN expression, e.g. convert Product and Sum to Mul and Add.
Expr ConvertCasToCinn(Expr expr);
//! Tell whether this expression is acceptable by CAS.
bool IsExprCasCompatible(Expr expr);

struct ExprPosCmp {
  bool operator()(const Expr& a, const Expr& b);
};

struct CasSimplifyMutator {
  explicit CasSimplifyMutator(const absl::flat_hash_map<std::string, CasInterval> var_intervals)
      : var_intervals(var_intervals) {}

  Expr operator()(Expr u);

  Expr SimplifyRationalNumber(Expr u);
  Expr SimplifyPower(Expr u);
  Expr SimplifySum(Expr u);
  Expr SimplifyProduct(Expr a);
  Expr SimplifyMinAndMax(Expr a);
  Expr SimplifyCmp(Expr a);
  std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& operands);
  std::vector<Expr> SimplifySumRec(const std::vector<Expr>& operands);
  Expr SimplifyMod(Expr u);
  Expr SimplifyFracOp(Expr expr);
  Expr SimplifyCond(Expr u);
  Expr FurtherSimplifyFracWithInterval(Expr expr, const absl::flat_hash_map<std::string, CasInterval>& var_intervals);
  Expr SimplifyIntegerPower(Expr u);
  void AddBaseAndSimplify(Expr* base, Expr bound);
  void UnfoldBound(Expr* lower_bound, Expr* upper_bound, Expr var, bool unfold_const_bound = true);
  bool GetVarBound(Expr* lower_bound, Expr* upper_bound, Expr var, bool unfold_const_bound = true);
  bool GetOperandBound(Expr* lower_bound, Expr* upper_bound, Expr var, bool unfold_const_bound = true);
  bool GetSumBound(Expr* lower_bound, Expr* upper_bound, Expr sum, bool unfold_const_bound = true);
  bool GetMinBound(Expr* lower_bound, Expr* upper_bound, Expr min, bool unfold_const_bound = true);
  bool GetMaxBound(Expr* lower_bound, Expr* upper_bound, Expr max, bool unfold_const_bound = true);
  bool GetExprBound(Expr* lower_bound, Expr* upper_bound, Expr min, bool unfold_const_bound = true);
  bool SimplifySpecificSumMod(Expr* u, Expr a, Expr b);
  Expr SimplifySpecificSum(Expr u);

 private:
  std::vector<Expr> SimplifyBinaryProduct(Expr left, Expr right);
  std::vector<Expr> MergeProduct(const std::vector<Expr>& p, const std::vector<Expr>& q);

  std::vector<Expr> SimplifyBinarySum(Expr left, Expr right);
  std::vector<Expr> MergeSum(const std::vector<Expr>& p, const std::vector<Expr>& q);
  std::vector<Expr> MergeExprs(const std::vector<Expr>& p,
                               const std::vector<Expr>& q,
                               const std::function<std::vector<Expr>(Expr, Expr)>& binary_merge);

  const absl::flat_hash_map<std::string, CasInterval> var_intervals;

  // Computation based on integer if set true(1/2 get 0), false if treat as rational number in mathematics(1/2 is still
  // 1/2), currently it only works with true.
  bool int_compute_{true};
};

}  // namespace detail

}  // namespace common
}  // namespace cinn
