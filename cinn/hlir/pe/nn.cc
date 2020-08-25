#include "cinn/hlir/pe/nn.h"

#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Expr;
using ir::Tensor;

template <typename T>
Tensor Relu(const Tensor& A, T threshold, const std::string& output_name) {
  return Compute(
      A->shape, [&](const std::vector<Expr>& indice) { return Relu(A(indice), threshold); }, output_name);
}

Tensor LeakyRelu(const Tensor& A, double alpha, const std::string& output_name) {
  return Compute(
      A->shape, [&](const std::vector<Expr>& indice) { return LeakyRelu(A(indice), alpha); }, output_name);
}

Tensor PRelu(const Tensor& A, const Tensor& slope, const int axis, const std::string& output_name) {
  CHECK_LT(axis, A->shape.size()) << "Wrong axis value: " << axis << std::endl;
  CHECK(A->shape[axis] == slope->shape[0]) << "Wrong slope shape: " << slope->shape[0] << std::endl;
  return Compute(
      A->shape,
      [&](const std::vector<Expr>& indice) { return LeakyRelu(A(indice), slope(indice[axis])); },
      output_name);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
