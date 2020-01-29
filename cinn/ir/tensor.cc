#include "cinn/ir/tensor.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

Tensor::Tensor(const std::vector<Var> &shape, Type type) : IrNodeRef(common::make_shared<_Tensor_>()) {
  LOG(INFO) << "tensor to set type " << type;
  operator->()->shape.clear();
  for (auto &v : shape) {
    operator->()->shape.push_back(Expr(v));
  }

  operator->()->set_type(type);
}
Tensor::Tensor(const std::vector<Expr> &shape, Type type) : IrNodeRef(common::make_shared<_Tensor_>()) {
  operator->()->shape = shape;
  operator->()->set_type(type);
}

const _Tensor_ *Tensor::operator->() const {
  auto *p = As<_Tensor_>();
  CHECK(p) << "type not match";
  return p;
}
_Tensor_ *Tensor::operator->() {
  auto *p = As<_Tensor_>();
  CHECK(p) << "type not match";
  return p;
}
size_t Tensor::ndims() const { return operator->()->shape.size(); }

Expr Tensor::operator()(const std::vector<Expr> &indices) const {
  CHECK_EQ(indices.size(), ndims()) << "dimension not match";
  auto n = Call::Make(operator->()->type().ElementOf(),  //
                                                         // operator->()->op->name,  //
                      "cinn_buffer_get_element",
                      indices,           //
                      Call::Halide,      //
                      operator->()->op,  //
                      operator->()->value_index);
  n->set_type(operator->()->type());
  return n;
}

Expr Tensor::operator()(const std::vector<Var> &indices) const {
  std::vector<Expr> _indices(indices.begin(), indices.end());
  return operator()(_indices);
}

bool Tensor::operator==(const Tensor &other) const {
  if (get() == other.get()) return true;
  if (!get() || !get()) return false;
  if (operator->()->op.defined() && other->op.defined()) {
    return operator->()->op == other->op && operator->()->value_index == other->value_index;
  }
}

IrNodeTy Tensor::node_type() const { return ir::IrNodeTy ::_Tensor_; }

void _Tensor_::Accept(IrVisitor *v) const { v->Visit(this); }

Tensor _Tensor_::Make(const std::vector<Expr> &shape, Type dtype, Operation op, int value_index) {
  auto *node  = common::make_shared<_Tensor_>();
  node->shape = shape;
  node->set_type(dtype);
  node->op          = op;
  node->value_index = value_index;
  return Tensor(node);
}

const _Operation_ *Operation::operator->() const { return ptr()->As<_Operation_>(); }
Tensor Operation::output(size_t i) const { return Tensor(); }

}  // namespace ir
}  // namespace cinn
