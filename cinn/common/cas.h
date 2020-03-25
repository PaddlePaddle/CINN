#pragma once
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr AutoSimplify(Expr u);

//! Simplify a CAS expression.
Expr CasSimplify(Expr u);

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

Expr SimplifyRationalNumber(Expr u);
Expr SimplifyPower(Expr u);
Expr SimplifySum(Expr u);
Expr SimplifyProduct(Expr a);
std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& operands);
std::vector<Expr> SimplifySumRec(const std::vector<Expr>& operands);
Expr SimplifyMod(Expr u);
Expr EvaluateSum(Expr v, Expr w);

}  // namespace detail

}  // namespace common
}  // namespace cinn
