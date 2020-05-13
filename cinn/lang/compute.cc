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

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr()> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        // CHECK_EQ(axis.size(), 0);
        return fn();
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 1);
        return fn(axis[0]);
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 2);
        return fn(axis[0], axis[1]);
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 3);
        return fn(axis[0], axis[1], axis[2]);
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 4);
        return fn(axis[0], axis[1], axis[2], axis[3]);
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  return Compute(
      domain,
      [fn](const std::vector<Expr> &axis) -> Expr {
        CHECK_EQ(axis.size(), 5);
        return fn(axis[0], axis[1], axis[2], axis[3], axis[4]);
      },
      name,
      reduce_axis,
      shape);
}

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(const std::vector<Expr> &)> fn,
                   const std::string &name,
                   const std::vector<Var> &reduce_axis,
                   const std::vector<Expr> &shape) {
  auto axises = common::GenDefaultAxis(domain.size());
  std::vector<Expr> _axis;
  for (auto &x : axises) _axis.push_back(x);
  Expr fn_body = fn(_axis);

  // shape is the buffer's shape.
  std::vector<Expr> domain_without_reduce_axis;

  // construct the shape.
  for (auto dim : domain) {
    auto copied = dim;
    optim::Simplify(&copied);
    domain_without_reduce_axis.push_back(copied);
  }

  auto real_shape = shape.empty() ? domain_without_reduce_axis : shape;

  auto unique_name = name.empty() ? Context::Global().NewName("tensor") : name;

  auto op     = ir::ComputeOp::Make(unique_name, fn, real_shape, domain, reduce_axis);
  auto tensor = ir::_Tensor_::Make(unique_name, fn_body.type(), real_shape, domain, op, reduce_axis);
  return tensor;
}

ir::Tensor Call(const std::string &target,
                Type type,
                const std::vector<Expr> &dims,
                const std::vector<Expr> &args,
                const std::string &name) {
  auto call       = ir::Call::Make(type, target, args, {}, ir::CallType::CINN, ir::FunctionRef(), 0, Expr());
  auto call_op    = ir::CallOp::Make(target, args, 0, call);
  auto new_tensor = ir::_Tensor_::Make(name, type, dims, {Expr(1)}, call_op, {});
  new_tensor->WithBuffer();
  // Append write tensors in the tail.
  call.As<ir::Call>()->write_args.push_back(new_tensor);
  return new_tensor;
}

std::vector<ir::Tensor> Call(const std::string &target,
                             const std::vector<Expr> &args,
                             const std::vector<ReturnType> &return_types) {
  auto call = ir::Call::Make(Void(), target, args, {}, ir::CallType::CINN, ir::FunctionRef(), 0, Expr());
  std::vector<ir::Tensor> new_tensors;
  for (int i = 0; i < return_types.size(); i++) {
    auto &return_type = return_types[i];
    auto call_op      = ir::CallOp::Make(target, args, i, call);
    auto new_tensor   = ir::_Tensor_::Make(return_type.name, Void(), return_type.dims, {Expr(1)}, call_op, {});
    // Append write tensors in the tail.
    call.As<ir::Call>()->write_args.push_back(new_tensor);
    new_tensor->set_type(return_type.type);
    new_tensor->WithBuffer();
    new_tensors.push_back(new_tensor);
  }

  return new_tensors;
}

Expr CallExtern(const std::string &target, const std::vector<Expr> &args) {
  auto *proto = backends::ExternFunctionProtoRegistry::Global().Lookup(target);
  CHECK(proto) << "No extern function " << target << " found";
  auto call = ir::Call::Make(proto->ret_type, target, args, {}, ir::CallType::Extern);
  return call;
}

}  // namespace lang
}  // namespace cinn
