#include "hlir/instruction/primitive/binary.h"

#include "hlir/instruction/context.h"
#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {
namespace primitive {
using cinn::ir::_Tensor_;
using cinn::ir::Expr;
using cinn::ir::Tensor;
using cinn::ir::Var;
using cinn::lang::Compute;

cinn::ir::Tensor BinaryImpl::operator()(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b, const std::string& name) {
  CHECK(a.defined());
  CHECK(b.defined());

  int ndims = a->shape.size();
  auto axis = cinn::common::GenDefaultAxis(ndims);

  std::vector<Expr> shape;
  Tensor out_tensor;
  switch (ndims) {
    case 1:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i) -> Expr { return opr_(a(i), b(i)); }, name);
      break;
    case 2:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i, Var j) -> Expr { return opr_(a(i, j), b(i, j)); }, name);
      break;
    case 3:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i, Var j, Var k) -> Expr { return opr_(a(i, j, k), b(i, j, k)); }, name);
      break;
    case 4:
      out_tensor = Compute(
          a->shape,
          [a, b, this](Var i, Var j, Var k, Var m) -> Expr { return opr_(a(i, j, k, m), b(i, j, k, m)); },
          name);
      break;
    default:
      NOT_IMPLEMENTED
  }

  if (!inlined_) out_tensor->WithBuffer();
  return out_tensor;
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
