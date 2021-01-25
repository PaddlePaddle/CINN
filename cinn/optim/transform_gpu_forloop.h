#pragma once
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "cinn/ir/ir.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

using forloop_infos_t = std::map<std::string, std::map<std::string, poly::StageForloopInfo>>;

/**
 * Collect the grid and block dims from a group of stages.
 * The dims is the maximum extent of each GPU related forloops.
 */
ir::CudaAxisInfo GatherAxisInfoFromStages(const std::vector<poly::Stage*>& stage_group);

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
 * @param global_tensor_map The map mapping a tensor's name to itself.
 * @param forloop_infos A map of forloop to their infomation.
 */
void TransformGpuForloops(const forloop_infos_t& forloop_infos,
                          std::map<std::string, ir::Tensor>* global_tensor_map,
                          std::unordered_set<std::string>* resized_buffer,
                          Expr* expr);

/**
 * Remove the forloops of block and thread axis, add the kernel launch thread dimension information to the outermost
 * LoweredFunc.
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

/**
 * Add __syncthreads() to shared memory producer.
 */
void CudaSyncThreadsDropIfThenElse(Expr* expr);

}  // namespace optim
}  // namespace cinn
