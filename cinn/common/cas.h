#pragma once
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

/**
 * Interval of a _Var_.
 */
struct CasInterval {
  template <typename T>
  CasInterval(T l, T r) : l(l), r(r) {}
  int l, r;

  friend std::ostream& operator<<(std::ostream& os, const CasInterval& i) {
    os << "Interval[" << i.l << ", " << i.r << "]";
    return os;
  }
};

using cas_intervals_t = std::unordered_map<std::string, CasInterval>;

Expr AutoSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals = {});

//! Simplify a CAS expression.
Expr CasSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals = {});

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
  CasSimplifyMutator(const std::unordered_map<std::string, CasInterval> var_intervals) : var_intervals(var_intervals) {}

  Expr operator()(Expr u);

  Expr SimplifyRationalNumber(Expr u);
  Expr SimplifyPower(Expr u);
  Expr SimplifySum(Expr u);
  Expr SimplifyProduct(Expr a);
  std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& operands);
  std::vector<Expr> SimplifySumRec(const std::vector<Expr>& operands);
  Expr SimplifyMod(Expr u);
  Expr SimplifyFracOp(Expr expr);
  Expr FurtherSimplifyFracWithInterval(Expr expr, const std::unordered_map<std::string, CasInterval>& var_intervals);
  Expr SimplifyIntegerPower(Expr u);

 private:
  std::vector<Expr> MergeProduct(const std::vector<Expr>& _p, const std::vector<Expr>& _q);

  std::vector<Expr> MergeSum(const std::vector<Expr>& _p, const std::vector<Expr>& _q);

  const std::unordered_map<std::string, CasInterval> var_intervals;

  // Computation based on integer if set true(1/2 get 0), false if treat as rational number in mathematics(1/2 is still
  // 1/2), currently it only works with true.
  bool int_compute_{true};
};

}  // namespace detail

}  // namespace common
}  // namespace cinn
