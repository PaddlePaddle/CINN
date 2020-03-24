#pragma once
#include <vector>
#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr AutoSimplify(Expr u);

namespace detail {

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
