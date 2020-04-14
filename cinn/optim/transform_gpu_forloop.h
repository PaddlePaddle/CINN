#pragma once
#include <cinn/poly/stage.h>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

using forloop_infos_t = std::map<std::string, std::map<std::string, poly::StageForloopInfo>>;

/**
 * Mark the fortype and device of forloops if is GPU related, replace the loop iterators to GPU related axis(threadIdx.x
 * and so on).
 *
 * For example, input the code
 * \code
 * for (i, 0, 10)
 *   for (j, 0, 20)
 *     A(i,j)
 * \endcode
 *
 * with the `i` set as CUDA block axis, `j` set as CUDA thread axis, the original forloop will be modified to
 *
 * \code
 * for (blockIdx.x, 0, 10)
 *   for (threadIdx.x, 0, 20)
 *     A(blockIdx.x, threadIdx.x)
 * \endcode
 *
 * @param expr The expression to modify.
 * @param statement The target statement.
 * @param forloop_infos A map of forloop to their infomation.
 */
void TransformGpuForloop(const forloop_infos_t& forloop_infos, Expr* expr);

/**
 * Remove the forloops of block and thread axis, add the kernel dimension information to the outermost LoweredFunc.
 *
 * For example, input the code:
 * \code
 * // Note here, the outermost expression should be a LoweredFunc
 * _LoweredFunc_:
 *   for (blockIdx.x, 0, 10)
 *     for (threadIdx.x, 0, 20)
 *       A(blockIdx.x, threadIdx.x)
 * \endcode
 *
 * will be modified to
 * \code
 * _LoweredFunc_<blockDim:10, threadDim:20>:
 *   A(blockIdx.x, threadIdx.x)
 * \endcode
 *
 * \note For that the dimensions of each threadIdx or blockIdx should be constant, so this only takes For nodes, not
 * \note PolyFor nodes is allowed to be GPU related.
 */
void RemoveGpuForloopsAxis(Expr* expr);

}  // namespace optim
}  // namespace cinn
