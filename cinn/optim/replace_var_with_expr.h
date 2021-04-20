#pragma once
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace the variable with a expression.
 * @param var The variable to replace.
 * @param expr The candidate expression.
 */
void ReplaceVarWithExpr(Expr *source, const Var &var, const Expr &expr);

std::vector<std::vector<Expr>> CollectTensorIndex(Expr *source, const std::string &tensor_name);

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
                                 std::map<std::string, ir::Tensor> *global_tensor_map,
                                 std::unordered_set<std::string> &resized_buffer,
                                 bool blockidx,
                                 const Expr &extent);
}  // namespace optim
}  // namespace cinn
