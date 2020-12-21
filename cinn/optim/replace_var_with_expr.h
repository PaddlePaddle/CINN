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

void ReplaceVarWithExpr2(Expr *source,
                         const Var &var,
                         const Expr &expr,
                         std::map<std::string, ir::Tensor> *global_tensor_map,
                         bool blockidx);
}  // namespace optim
}  // namespace cinn
