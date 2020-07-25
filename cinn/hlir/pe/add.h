#pragma once
#include <string>
#include <vector>
#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/tensor.h"

using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

namespace cinn {
namespace hlir {
namespace pe {

Tensor Add(const Tensor &A, const Tensor &B, const std::string output_name = "") {
  CHECK(A->SameShapeWith(B)) << "The 2 inputs have different shapes with each other. "
                                "The Add fucntion needs two inputs to have identical shape.";
  const std::vector<Expr> output_shape = A->shape;
  CHECK_GE(output_shape.size(), 1) << "The input shape of pe::Add function is " << output_shape.size()
                                   << " and it should be >= 1.";
  CHECK_LE(output_shape.size(), 4) << "The input shape of pe::Add function is " << output_shape.size()
                                   << " and it should be <= 4.";

  Tensor output = Compute(
      output_shape, [&](const std::vector<Expr> &indice) { return A(indice) + B(indice); }, output_name);
  return output;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
