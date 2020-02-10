#include "cinn/lang/compute.h"

#include "cinn/poly/dim.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace lang {

using ir::Expr;

template <>
Tensor Compute<compute_handle_1_t>(const std::vector<int>& dims, compute_handle_1_t handle) {
  CHECK_EQ(dims.size(), 1);
  poly::Dim dim("i", 0, dims[0] - 1);
  Var i("i", Int(32));
  auto expr = handle(i);

  std::vector<Expr> shape;
  for (int v : dims) shape.emplace_back(v);

  Tensor tensor(shape, {i}, expr.type(), expr);
  return std::move(tensor);
}

template <>
Tensor Compute<compute_handle_2_t>(const std::vector<int>& dims, compute_handle_2_t handle) {
  CHECK_EQ(dims.size(), 2);
  poly::Dim dim("i", 0, dims[0] - 1);
  Var i("i", Int(32));
  Var j("j", Int(32));
  auto expr = handle(i, j);

  std::vector<Expr> shape;
  for (int v : dims) shape.emplace_back(v);

  Tensor tensor(shape, {i, j}, expr.type(), expr);
  CHECK(tensor.get());
  LOG(INFO) << "tensor.get " << tensor.get();
  LOG(INFO) << "expr: " << static_cast<ir::_Tensor_*>(tensor.get())->expr;
  LOG(INFO) << "tensor.expr " << tensor->expr;
  return std::move(tensor);
}

template <>
Tensor Compute<compute_handle_3_t>(const std::vector<int>& dims, compute_handle_3_t handle) {
  CHECK_EQ(dims.size(), 3);
  poly::Dim dim("i", 0, dims[0] - 1);
  Var i("i", Int(32));
  Var j("j", Int(32));
  Var k("k", Int(32));
  auto expr = handle(i, j, k);

  std::vector<Expr> shape;
  for (int v : dims) shape.emplace_back(v);

  Tensor tensor(shape, {i, j}, expr.type(), expr);
  return std::move(tensor);
}

}  // namespace lang
}  // namespace cinn
