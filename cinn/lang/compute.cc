#include "cinn/lang/compute.h"

#include "cinn/common/common.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace lang {

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var)> fn) {
  return Compute(dims, [fn](const std::vector<Var> &axis) -> Expr {
    CHECK_EQ(axis.size(), 1);
    return fn(axis[0]);
  });
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var)> fn) {
  return Compute(dims, [fn](const std::vector<Var> &axis) -> Expr {
    CHECK_EQ(axis.size(), 2);
    return fn(axis[0], axis[1]);
  });
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var, Var)> fn) {
  return Compute(dims, [fn](const std::vector<Var> &axis) -> Expr {
    CHECK_EQ(axis.size(), 3);
    return fn(axis[0], axis[1], axis[2]);
  });
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var, Var, Var)> fn) {
  return Compute(dims, [fn](const std::vector<Var> &axis) -> Expr {
    CHECK_EQ(axis.size(), 4);
    return fn(axis[0], axis[1], axis[2], axis[3]);
  });
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var, Var, Var, Var)> fn) {
  return Compute(dims, [fn](const std::vector<Var> &axis) -> Expr {
    CHECK_EQ(axis.size(), 5);
    return fn(axis[0], axis[1], axis[2], axis[3], axis[4]);
  });
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(const std::vector<Var> &)> fn) {
  auto axis = detail::GenDefaultAxis(dims.size());
  Expr expr = fn(axis);

  std::vector<Expr> shape;
  for (int v : dims) shape.emplace_back(v);

  ir::Tensor tensor(shape, axis, expr.type(), expr);
  return tensor;
}

namespace detail {
std::vector<Var> GenDefaultAxis(int naxis) {
  std::vector<Var> axis;
  for (int i = 0; i < naxis; i++) {
    axis.emplace_back(common::axis_name(i));
  }
  return axis;
}
}  // namespace detail

}  // namespace lang
}  // namespace cinn
