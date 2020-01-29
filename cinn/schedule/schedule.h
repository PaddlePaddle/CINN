#pragma once

#include <cinn/ir/node.h>
namespace cinn {
namespace schedule {

using ir::Expr;
using ir::IrNode;
using ir::IrNodeRef;

enum class AttachType : int {
  kInline = 0,
};

class _Stage_;

class Stage : public ir::IrNodeRef {
 public:
  Stage() = default;
  Stage(IrNode* n) : ir::IrNodeRef(n) {}

  //! Get the internal _Stage_ object.
  //! @{
  const _Stage_* operator->() const;
  _Stage_* operator->();
  //! @}

  /**
   * @brief Set the memory scope of the stage.
   * @param scope The memory scope.
   */
  Stage& SetScope(const std::string& scope);

  /**
   * @brief Specify the schedule to be computed at the parent schedule's scope.
   * @param parent The parent schedule.
   * @param scope The iteration point to carray the schedule.
   */
  Stage& ComputeAt(const Stage& parent, const ir::Var& scope);

  /**
   * @brief Compute the function inline.
   */
  Stage& ComputeInline();

  /**
   * @brief Compute the function at group root.
   */
  Stage& ComputeRoot();

  /**
   * @brief Bind the var to thread index.
   * @param ivar the var to be bound.
   * @param thread_ivar the thread axis to be bound.
   */
  Stage& Bind(ir::Var ivar, ir::Var thread_ivar);

  /**
   * @brief Set the predicate to determine whether a store to the array should be performed.
   * Use this when there are multiple threads performing the same store and we only need one of them to do the store.
   *
   * @param predicate The condition to be checked.
   */
  Stage& SetStorePredicate(Expr predicate);

  /**
   * @brief Split the parent by factor.
   * @param parent The parent iteration domain.
   * @param factor The split factor of the loop.
   * @param p_outer The result outer domain.
   * @param p_inner The result inner domain.
   */
  Stage& Split(ir::Var parent, Expr factor, ir::Var* p_outer, ir::Var* p_inner);

  /**
   * @brief Fuse the inner and outer domain to the target.
   * @param outer The outer domain to be fused.
   * @param inner The inner domain to be fused.
   * @param p_target The result target domain.
   */
  Stage& Fuse(ir::Var outer, ir::Var inner, ir::Var* p_target);

  /**
   * @brief Reorder the iteration.
   * @param order The order of iteration variable.
   */
  Stage& Reorder(const std::vector<ir::Var>& order);

  /**
   * @brief Perform tiling on two dimensions.
   * The final loop order from out-most to inner-most are
   * [x_outer, y_outer, x_inner, y_inner]
   * @param x_parent The original x dimension.
   * @param y_parent The original y dimension.
   * @param x_factor The stride factor on x axis.
   * @param y_factor The stride factor on y axis.
   * @param p_x_outer The outer axis of x dimension.
   * @param p_y_outer The outer axis of y dimension.
   * @param p_x_inner The inner axis of x dimension.
   * @param p_y_inner The inner axis of y dimension.
   */
  Stage& Tile(ir::Var x_parent,
              ir::Var y_parent,
              Expr x_factor,
              Expr y_factor,
              ir::Var* p_x_outer,
              ir::Var* p_y_outer,
              ir::Var* p_x_inner,
              ir::Var* p_y_inner);

  /**
   * @brief Vectorize the iteration.
   * @param var The axis to be vectorized.
   */
  Stage& Vectorize(ir::Var var);

  /**
   * @brief Unroll the iteration.
   * @param var The axis to be unrolled.
   */
  Stage& Unroll(ir::Var var);

  /**
   * @brief Parallelize iteration.
   * @param var The axis to be parallelized.
   */
  Stage& Parallelize(ir::Var var);
};

}  // namespace schedule
}  // namespace cinn
