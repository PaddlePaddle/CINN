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

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
