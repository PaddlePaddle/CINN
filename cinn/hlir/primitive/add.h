#pragma once
#include <string>
#include <vector>
#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"

using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Buffer;
using cinn::lang::Compute;
using cinn::lang::Placeholder;

namespace cinn {
namespace primitive {

template <typename T>
Tensor add(const Placeholder<T> &A, const Placeholder<T> &B, const std::string &output_name) {
  _Tensor_ *input_A = A.tensor().self();
  CHECK(input_A->SameShapeWith(B.tensor())) << "The 2 inputs have different shapes with each other. "
                                               "The add fucntion needs two inputs to have identical shape.";
  const std::vector<Expr> output_shape = input_A->shape;
  CHECK_GE(output_shape.size(), 1) << "The input shape of primitive::add function is " << output_shape.size()
                                   << " and it should be >= 1.";
  CHECK_LE(output_shape.size(), 4) << "The input shape of primitive::add function is " << output_shape.size()
                                   << " and it should be <= 4.";

  Tensor output;
  switch (output_shape.size()) {
    case 1:
      output = Compute(
          output_shape, [&](Var i) { return A(i) + B(i); }, output_name);
      return output;
    case 2:
      output = Compute(
          output_shape, [&](Var i, Var j) { return A(i, j) + B(i, j); }, output_name);
      return output;
    case 3:
      output = Compute(
          output_shape, [&](Var i, Var j, Var k) { return A(i, j, k) + B(i, j, k); }, output_name);
      return output;
    case 4:
      output = Compute(
          output_shape, [&](Var i, Var j, Var k, Var p) { return A(i, j, k, p) + B(i, j, k, p); }, output_name);
      return output;
  }
  return output;
}

}  // namespace primitive
}  // namespace cinn
