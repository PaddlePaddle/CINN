#pragma once
#include <vector>
#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr SimplifyRNE(Expr u);

Expr AutoSimplify(Expr u);

namespace detail {

struct ExprPosCmp {
  bool operator()(const Expr& a, const Expr& b);
};

/**
 * Add(Add(w,x), Add(y,z)) => Sum(w,x,y,z)
 */
Expr TransformMulToProduct(Expr u);
/**
 * Mul(Mul(w,x), Mul(y,z)) => Product(w,x,y,z)
 */
Expr TransformAddToSum(Expr u);

Expr SimplifyRationalNumber(Expr u);
Expr SimplifyPower(Expr u);
Expr SimplifySum(Expr u);
Expr SimplifyProduct(Expr a);
Expr EvaluateSum(Expr v, Expr w);
Expr EvaluateProd(Expr v, Expr w);
std::vector<Expr> SimplifyProductRec(const std::vector<Expr>& operands);
std::vector<Expr> SimplifySumRec(const std::vector<Expr>& operands);

}  // namespace detail

}  // namespace common
}  // namespace cinn
