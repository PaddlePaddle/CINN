#include "cinn/lang/tensor.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"

namespace cinn {
namespace lang {

Tensor::Tensor(const std::vector<Expr> &shape, const std::vector<Var> &iterators, Type dtype, ir::Expr expr)
    : IrNodeRef(ir::_Tensor_::Make(shape, iterators, dtype, expr)) {}

size_t Tensor::ndims() const { return operator->()->shape.size(); }

Expr Tensor::operator()(const std::vector<Expr> &indices) const {
  CHECK_EQ(indices.size(), ndims()) << "number of indices not match the dimension";
  auto n =
      ir::Call::Make(operator->()->type().ElementOf(), ir::ExternOp::buffer_get_element, indices, ir::Call::Halide);
  n->set_type(operator->()->type());
  return n;
}

}  // namespace lang
}  // namespace cinn