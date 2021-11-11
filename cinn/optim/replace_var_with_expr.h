// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
 * @param source The expression to be visited and edited.
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
 * @param source The expression to be visited and edited.
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
 * In cuda backend, replace a var to another expr.
 * There are two classic examples:
 * 1.Remove the loop iterators within the compute_at level in temp tensor's indices and resize tensor/buffer's shape
 * correspondingly.
 *
 * If A_write_cache->ComputeAt(A, 1)
 * \code
 * for (i, 0, 10)
 *   for (j, 0, 10)
 *      A_write_cache(i,j) = i * j
 *      A(i,j) = A_write_cache(i,j)
 * \endcode
 *
 * will be replaced to
 *
 * \code
 * for (i, 0, 10)
 *   for (j, 0, 10)
 *      A_write_cache(0) = i * j
 *      A(i,j) = A_write_cache(0)
 * \endcode
 *
 * And the shape will be resized from (10*10) to 1.
 *
 * 2.Erase `blockIdx` and 'threadIDx' in MemoryType::GPULocal; erase `blockIdx` in MemoryType::GPUShared and resize
 * corresponding tensor/buffer's shape.
 *
 * If A_write_cache's memory type is MemoryType::GPULocal:
 * \code
 * for (blockIdx.x, 0, 10)
 *   for (threadIdx.x, 0, 10)
 *      A_write_cache(blockIdx.x,threadIdx.x) = blockIdx.x * threadIdx.x
 * \endcode
 *
 * will be replaced to
 *
 * \code
 * for (blockIdx.x, 0, 10)
 *   for (threadIdx.x, 0, 10)
 *      A_write_cache(0) = blockIdx.x * threadIdx.x
 * \endcode
 *
 * And the shape will be resized from (10*10) to 1.
 *
 * @param source The expression to be visted and edited.
 * @param var The variable to be replaced.
 * @param expr The candidate expression.
 * @param global_tensor_map The global tensor map.
 * @param resized_buffer The set of ID which indicates buffers already been resized. This is used to avoid duplication
 * when resizing temp buffer's shape.
 * @param blockidx If the var to be replaced is binded to blockIdx.
 * @param extent The variable's extent. This is used to resize tensor's shape.
 * @param tensor_name If this param is not nullptr, we will do the replacement only in this tensor's ir::Load and
 * ir::Store.
 */
void CUDAReplaceIndexOfCachePass(Expr *source,
                                 const Var &var,
                                 const Expr &expr,
                                 std::map<std::string, ir::Tensor> *global_tensor_map,
                                 std::unordered_set<std::string> &resized_buffer,
                                 bool blockidx,
                                 const Expr &extent,
                                 std::string tensor_name = "");
}  // namespace optim
}  // namespace cinn
