#include "cinn/lang/compute.h"

#include "cinn/common/common.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace lang {

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Expr)> fn, const std::string &name) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 1);
        return fn(axis[0]);
      },
      name);
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Expr, Expr)> fn, const std::string &name) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 2);
        return fn(axis[0], axis[1]);
      },
      name);
}

ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Expr, Expr, Expr)> fn, const std::string &name) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 3);
        return fn(axis[0], axis[1], axis[2]);
      },
      name);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 4);
        return fn(axis[0], axis[1], axis[2], axis[3]);
      },
      name);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 5);
        return fn(axis[0], axis[1], axis[2], axis[3], axis[4]);
      },
      name);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(const std::vector<Expr> &)> fn,
                   const std::string &name) {
  auto axis = common::GenDefaultAxis(dims.size());
  std::vector<Expr> _axis;
  for (auto &x : axis) _axis.push_back(x);
  Expr expr = fn(_axis);

  std::vector<Expr> shape;
  for (int v : dims) shape.emplace_back(v);

  auto unique_name = name.empty() ? Context::Global().NewName("tensor") : name;

  // auto op = ir::ComputeOp::Make(unique_name, "" /*tag*/, {}, axis, {expr}, shape);
  auto op = ir::ComputeOp::Make(unique_name, "" /*tag*/, {}, fn, shape);
  return ir::_Tensor_::Make(unique_name, shape, op);
}

}  // namespace lang
}  // namespace cinn
