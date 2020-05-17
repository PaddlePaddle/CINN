#include "cinn/hlir/instruction/primitive/binary.h"

#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
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

  std::vector<Expr> shape;
  Tensor out_tensor;
  switch (b->shape.size()) {
    case 1:
      out_tensor = RunWithArgb1Dim(a, b);
      break;
    case 2:
      out_tensor = RunWithArgb2Dim(a, b);
      break;
    case 3:
      out_tensor = RunWithArgb3Dim(a, b);
      break;
    case 4:
      out_tensor = RunWithArgb4Dim(a, b);
      break;
    case 5:
      out_tensor = RunWithArgb5Dim(a, b);
      break;
    default:
      NOT_IMPLEMENTED
  }

  if (!inlined_) out_tensor->WithBuffer();
  return out_tensor;
}

cinn::ir::Tensor BinaryImpl::RunWithArgb1Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 1UL);

  Tensor out_tensor;
  switch (a->shape.size()) {
    case 1:
      out_tensor = Compute(a->shape, [a, b, this](Var i) -> Expr { return opr_(a(i), b(i)); });
      break;
    case 2:
      out_tensor = Compute(a->shape, [a, b, this](Var i, Var j) -> Expr { return opr_(a(i, j), b(j)); });
      break;
    case 3:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var j) -> Expr { return opr_(a(i0, i1, j), b(j)); });
      break;
    case 4:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i0, Var i1, Var i2, Var j) -> Expr { return opr_(a(i0, i1, i2, j), b(j)); });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryImpl::RunWithArgb2Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 2UL);
  Tensor out_tensor;
  switch (a->shape.size()) {
    case 2:
      out_tensor = Compute(a->shape, [a, b, this](Var i, Var j) -> Expr { return opr_(a(i, j), b(i, j)); });
      break;
    case 3:
      out_tensor =
          Compute(a->shape, [a, b, this](Var i0, Var i1, Var j) -> Expr { return opr_(a(i0, i1, j), b(i1, j)); });
      break;
    case 4:
      out_tensor = Compute(
          a->shape, [a, b, this](Var i0, Var i1, Var i2, Var j) -> Expr { return opr_(a(i0, i1, i2, j), b(i2, j)); });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryImpl::RunWithArgb3Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 3UL);
  Tensor out_tensor;
  switch (a->shape.size()) {
    case 3:
      out_tensor =
          Compute(a->shape, [a, b, this](Var i0, Var i1, Var j) -> Expr { return opr_(a(i0, i1, j), b(i0, i1, j)); });
      break;
    case 4:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var j) -> Expr {
        return opr_(a(i0, i1, i2, j), b(i1, i2, j));
      });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryImpl::RunWithArgb4Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 4UL);
  Tensor out_tensor;
  switch (a->shape.size()) {
    case 4:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var j) -> Expr {
        return opr_(a(i0, i1, i2, j), b(i0, i1, i2, j));
      });
      break;
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(i1, i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

cinn::ir::Tensor BinaryImpl::RunWithArgb5Dim(const Tensor& a, const Tensor& b) {
  CHECK_EQ(b->shape.size(), 5UL);
  Tensor out_tensor;
  switch (a->shape.size()) {
    case 5:
      out_tensor = Compute(a->shape, [a, b, this](Var i0, Var i1, Var i2, Var i3, Var j) -> Expr {
        return opr_(a(i0, i1, i2, i3, j), b(i0, i1, i2, i3, j));
      });
      break;
    default:
      NOT_IMPLEMENTED
  }
  return out_tensor;
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
