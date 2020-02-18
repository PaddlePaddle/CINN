#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace the variable with a expression.
 * @param var The variable to replace.
 * @param expr The candidate expression.
 */
void ReplaceVarWithExpr(Expr *source, const Var &var, const Expr &expr);

}  // namespace optim
}  // namespace cinn
