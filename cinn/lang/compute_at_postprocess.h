//! \file This file contains some post process of ComputeAt schedule.
#pragma once
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace lang {

/**
 * Deal with the `compute_at` transform, in the stage transform phase, we modified the domain and transform of the
 * producer tensor, after isl Ast generation, there remains some postprocess here include
 *
 * 1. in producer tensor load, make each axis to zero
 * 2. add offset
 *
 * e.g.
 *
 * auto A_cache = Compute({M, N}, [&](Expr i, Expr j) { return A(i, j); }, "cache");
 * auto C = Compute(
 *     {Expr(10), Expr(10)}, [&](Expr i, Expr j) { return A_cache(i, j) + A_cache(i+1,j) + B(i, j); }, "C");
 * A_cache->stage()->ComputeAt2(C->stage(), 0);
 *
 * \code
 * function fn (_A, _B, _cache, _C)
 * {
 *   for (_p0, 10)
 *   {
 *     for (i, 10)
 *     {
 *       if ((i <= 1)) {
 *         for (j, 10)
 *         {
 *           cache[i, j] = A[i, j]
 *         }
 *       }
 *       C[_p0, i] = (cache[_p0, i] + (cache[(1 + _p0), i] + B[_p0, i]))
 *      }
 *   }
 * }
 * \endcode
 *
 * The expression `C[_p0, i] = (cache[_p0, i] + (cache[(1 + _p0), i] + B[_p0, i]))` produces tensor `C`, but the cache
 * should start from zero.
 */
void ProcessComputeAtInfo(Expr* expr);

/**
 * Resize the compute_at consumer buffer size.
 */
void UpdateComputeAtBufferShape(Expr* expr);

namespace detail {

/**
 * Replace isl parameters with consumer iterators.
 * @param info ComputeAt schedule related information.
 * @param axis The consumer axis.
 * @param consumer_forloop_root The first forloop level of consumer expression.
 */
void ReplaceParamWithConsumerAxis(const ir::ComputeAtInfo& info,
                                  const std::vector<Var>& axis,
                                  Expr* consumer_forloop_root);

}  // namespace detail

}  // namespace lang
}  // namespace cinn
