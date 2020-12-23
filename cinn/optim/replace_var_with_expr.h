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

/**
 * In cuda backend, replace the var binded to 'threadIdx.x'/'blockIdx.x'
 * of the cache tensor with expr.
 * @param var The variable to replace.
 * @param expr The candidate expression.
 * @param global_tensor_map The global tensor map.
 * @param blockidx If the var to be replaced is binded to blockIdx.
 */
void CUDAReplaceIndexOfCachePass(Expr *source,
                                 const Var &var,
                                 const Expr &expr,
                                 const std::map<std::string, ir::Tensor> *global_tensor_map,
                                 bool blockidx);
}  // namespace optim
}  // namespace cinn
