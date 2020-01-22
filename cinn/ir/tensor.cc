#include "cinn/ir/tensor.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

const _Tensor_ *Tensor::operator->() const { return As<_Tensor_>(); }
size_t Tensor::ndims() const { return operator->()->shape.size(); }
Expr Tensor::operator()(const std::vector<Expr> &indices) { return Expr(); }
Expr Tensor::operator()(const std::vector<Var> &indices) { return Expr(); }
bool Tensor::operator==(const Tensor &other) const {
  if (get() == other.get()) return true;
  if (!get() || !get()) return false;
  if (operator->()->op.defined() && other->op.defined()) {
    return operator->()->op == other->op && operator->()->value_index == other->value_index;
  }
}

void _Tensor_::Accept(IrVisitor *v) const { v->Visit(this); }

Tensor _Tensor_::Make(const std::vector<Expr> &shape, Type dtype, Operation op, int value_index) {
  auto *node        = common::make_shared<_Tensor_>();
  node->shape       = shape;
  node->dtype       = dtype;
  node->op          = op;
  node->value_index = value_index;
  return Tensor(node);
}

}  // namespace ir
}  // namespace cinn
