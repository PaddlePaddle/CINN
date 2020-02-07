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

}  // namespace lang
}  // namespace cinn
