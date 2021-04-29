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
 * @param tensor_name Name of the tensor whose indices will be edited. If it is empty, means we will
 * do the replace in all Expr instead of only in specific tensor's indices.
 */
/**
 * Example 1: ReplaceVarWithExpr(source, Var("i"), Expr(0), "A")
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[i,j]
 *
 * =>
 *
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[0,j]
 *
 * Example 2: ReplaceVarWithExpr(source, Var("i"), Expr(Var("k")))
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[i,j]
 *
 * =>
 *
 * for(k, 0, 10)
 *   for(j, 0, 10)
 *      B[k,j] = A[k,j]
 */
void ReplaceVarWithExpr(Expr *source, const Var &var, const Expr &expr, const std::string &tensor_name = "");

/**
 * Collect the specific tensor's indices.
 * @param tensor_name The specific tensor's name.
 * @return Return a vector containing all the indices of the specific tensor appeared in source.
 */
/**
 * Example: CollectTensorIndex(source, "A")
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      C[i,j] = A[i,j] + A[0,j] + B[j,i] + B[i,0]
 *
 * =>
 *
 * Return value:
 * {{i,j},{0,j}}
 */
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
