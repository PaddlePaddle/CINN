#pragma once
#include <cinn/poly/stage.h>

#include "cinn/ir/ir.h"
#include "cinn/poly/isl_utils.h"

namespace cinn {
namespace optim {

using forloop_infos_t = std::map<std::string, std::map<std::string, poly::StageForloopInfo>>;

using dim3_t = std::array<int, 3>;

struct CudaAxisInfo {
  CudaAxisInfo() {
    for (int& v : grid_dims_) v = 1;
    for (int& v : block_dims_) v = 1;
  }

  void set_grid_dim(int offset, int x);
  void set_block_dim(int offset, int x);

  int grid_dim(int offset) const;
  int block_dim(int offset) const;

  void CopyGridDimsTo(std::vector<int>* dest) const;
  void CopyBlockDimsTo(std::vector<int>* dest) const;

  inline void set_valid(bool x = false) { valid_ = x; }
  inline bool valid() const { return valid_; }

  //! Extend the axis dims and keep the larger dims.
  void ExtendWith(const CudaAxisInfo& other);

 private:
  dim3_t grid_dims_;
  dim3_t block_dims_;
  bool valid_{false};
};

std::ostream& operator<<(std::ostream& os, const CudaAxisInfo& x);

/**
 * Collect the grid and block dims from a group of stages.
 * The dims is the maximum extent of each GPU related forloops.
 */
CudaAxisInfo GatherAxisInfoFromStages(const std::vector<poly::Stage*>& stage_group);

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
void TransformGpuForloops(const forloop_infos_t& forloop_infos, Expr* expr);

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

}  // namespace optim
}  // namespace cinn
