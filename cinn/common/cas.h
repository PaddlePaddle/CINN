#pragma once
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

/**
 * Interval of a _Var_.
 */
struct CasInterval {
  CasInterval(int l, int r) : l(l), r(r) {}
  int l, r;

  friend std::ostream& operator<<(std::ostream& os, const CasInterval& i) {
    os << "Interval[" << i.l << ", " << i.r << "]";
    return os;
  }
};

Expr AutoSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals = {});

//! Simplify a CAS expression.
Expr CasSimplify(Expr u, const std::unordered_map<std::string, CasInterval>& var_intervals = {});

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
};

}  // namespace detail

}  // namespace common
}  // namespace cinn
