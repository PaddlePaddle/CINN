#include "cinn/lang/compute.h"

#include "cinn/backends/extern_func_protos.h"
#include "cinn/common/common.h"
#include "cinn/ir/operation.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace lang {

ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr()> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  return Compute(
      dims,
      [fn](const std::vector<Expr> &axis) -> Expr {
        // CHECK_EQ(axis.size(), 0);
        return fn();
      },
      name,
      reduce_axis);
}

ir::Tensor Compute(const std::vector<Expr> &dims,
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

ir::Tensor Compute(const std::vector<Expr> &dims,
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

ir::Tensor Compute(const std::vector<Expr> &dims,
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

ir::Tensor Compute(const std::vector<Expr> &dims,
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

ir::Tensor Compute(const std::vector<Expr> &dims,
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

ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(const std::vector<Expr> &)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis) {
  auto axises = common::GenDefaultAxis(dims.size());
  std::vector<Expr> _axis;
  for (auto &x : axises) _axis.push_back(x);
  Expr fn_body = fn(_axis);

  // shape is the buffer's shape.
  std::vector<Expr> shape;

  // construct the shape.
  for (auto dim : dims) {
    auto copied = dim;

    optim::Simplify(&copied);
    shape.push_back(copied);
  }

  // domain is the domain of all the loop axis.
  std::vector<Expr> domain = shape;

  // append the reduce aixs to the domain, make the domain contain the range of all the forloop variables.
  if (!reduce_axis.empty()) {
    // We ignore the lower bound currently.
    for (auto &axis : reduce_axis) {
      CHECK_EQ(axis->lower_bound.as_int32(), 0);
      domain.emplace_back(axis->upper_bound);
      axises.push_back(axis);
    }
  }

  auto unique_name = name.empty() ? Context::Global().NewName("tensor") : name;

  auto op        = ir::ComputeOp::Make(unique_name, "" /*tag*/, {}, fn, shape, domain, reduce_axis);
  auto tensor    = ir::_Tensor_::Make(unique_name, shape, op);
  tensor->axis   = axises;
  tensor->domain = domain;
  return tensor;
}

ir::Tensor Call(const std::string &target,
                Type type,
                const std::vector<Expr> &dims,
                const std::vector<Expr> &args,
                const std::string &name) {
  auto call       = ir::Call::Make(type, target, args, {}, ir::Call::CallType::CINN, ir::FunctionRef(), 0, Expr());
  auto call_op    = ir::CallOp::Make(target, args, 0, call);
  auto new_tensor = ir::_Tensor_::Make(name, dims, call_op);
  new_tensor->WithBuffer();
  // Append write tensors in the tail.
  call.As<ir::Call>()->write_args.push_back(new_tensor);
  return new_tensor;
}

std::vector<ir::Tensor> Call(const std::string &target,
                             const std::vector<Expr> &args,
                             const std::vector<ReturnType> &return_types) {
  auto call = ir::Call::Make(Void(), target, args, {}, ir::Call::CallType::CINN, ir::FunctionRef(), 0, Expr());
  std::vector<ir::Tensor> new_tensors;
  for (int i = 0; i < return_types.size(); i++) {
    auto &return_type = return_types[i];
    auto call_op      = ir::CallOp::Make(target, args, i, call);
    auto new_tensor   = ir::_Tensor_::Make(return_type.name, return_type.dims, call_op);
    // Append write tensors in the tail.
    call.As<ir::Call>()->write_args.push_back(new_tensor);
    new_tensor->WithBuffer();
    new_tensors.push_back(new_tensor);
  }

  return new_tensors;
}

Expr CallExtern(const std::string &target, const std::vector<Expr> &args) {
  auto *proto = backends::ExternFunctionProtoRegistry::Global().Lookup(target);
  CHECK(proto) << "No extern function " << target << " found";
  auto call = ir::Call::Make(proto->ret_type, target, args, {}, ir::Call::CallType::Extern);
  return call;
}

}  // namespace lang
}  // namespace cinn
