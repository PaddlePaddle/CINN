#include "cinn/ir/tensor.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

const _Tensor_ *Tensor::operator->() const { return As<_Tensor_>(); }
size_t Tensor::ndims() const { return operator->()->shape.size(); }
Expr Tensor::operator()(const std::vector<Expr> &indices) { return Expr(); }
Expr Tensor::operator()(const std::vector<Var> &indices) { return Expr(); }

void _Tensor_::Accept(IrVisitor *v) const { v->Visit(this); }

}  // namespace ir
}  // namespace cinn
