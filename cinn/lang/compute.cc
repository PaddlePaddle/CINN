#include "cinn/lang/compute.h"

#include "cinn/common/common.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace lang {

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 1);
        return fn(axis[0]);
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 2);
        return fn(axis[0], axis[1]);
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 3);
        return fn(axis[0], axis[1], axis[2]);
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 4);
        return fn(axis[0], axis[1], axis[2], axis[3]);
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 5);
        return fn(axis[0], axis[1], axis[2], axis[3], axis[4]);
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(const std::vector<Expr> &)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  auto axises = common::GenDefaultAxis(dims.size());
  std::vector<Expr> _axis;
  for (auto &x : axises) _axis.push_back(x);
  Expr expr = fn(_axis);

  // shape is the buffer's shape.
  std::vector<Expr> shape;
  // domain is the domain of all the loop axis.
  std::vector<Expr> domain;
  for (int i = 0; i < dims.size(); i++) {
    domain.emplace_back(dims[i]);
  }

  // append the reduce aixs to the domain, make the domain contain the range of all the forloop variables.
  if (!reduce_axis.empty()) {
    // We ignore the lower bound currently.
    for (auto &axis : reduce_axis) {
      CHECK_EQ(axis->lower_bound.as_int32(), 0);
      domain.emplace_back(axis->upper_bound);
      axises.push_back(axis);
    }
  }

  // construct the shape.
  for (int i = 0; i < dims.size(); i++) {
    shape.emplace_back(dims[i]);
  }

  auto unique_name = name.empty() ? Context::Global().NewName("tensor") : name;

  auto op        = ir::ComputeOp::Make(unique_name, "" /*tag*/, {}, fn, shape, domain, reduce_axis);
  auto tensor    = ir::_Tensor_::Make(unique_name, shape, op);
  tensor->axis   = axises;
  tensor->domain = domain;
  return tensor;
}

}  // namespace lang
}  // namespace cinn
