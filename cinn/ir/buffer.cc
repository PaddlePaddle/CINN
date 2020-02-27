#include "cinn/ir/buffer.h"

#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

const _Buffer_ *Buffer::operator->() const { return IrNodeRef::As<_Buffer_>(); }
_Buffer_ *Buffer::operator->() { return IrNodeRef::As<_Buffer_>(); }

Buffer _Buffer_::Make(Var data,
                      Type dtype,
                      const std::vector<Expr> &shape,
                      const std::vector<Expr> &strides,
                      Expr elem_offset,
                      const std::string &name,
                      const std::string &scope,
                      int data_alignment,
                      int offset_factor) {
  auto *node           = common::make_shared<_Buffer_>();
  node->data           = data;
  node->shape          = shape;
  node->strides        = strides;
  node->elem_offset    = elem_offset;
  node->name           = name;
  node->scope          = scope;
  node->data_alignment = data_alignment;
  node->offset_factor  = offset_factor;
  node->set_type(dtype);
  return Buffer(node);
}

Buffer _Buffer_::Make(const std::string &name, const std::vector<Expr> &shape) {
  auto *node  = common::make_shared<_Buffer_>();
  node->name  = name;
  node->shape = shape;
  return Buffer(node);
}

Buffer _Buffer_::Make() {
  auto *node = common::make_shared<_Buffer_>();
  return Buffer(node);
}

void _Buffer_::Accept(IRVisitor *v) const { v->Visit(this); }
IrNodeTy _Buffer_::node_type() const { return _node_type_; }

void _Buffer_::BindTo(const Tensor &tensor) { BindTo(tensor.As<_Tensor_>()); }

void _Buffer_::BindTo(const _Tensor_ *tensor) {
  if (name.empty()) name = tensor->name;
  if (!data.defined()) data = _Var_::Make(name, tensor->type()).As<ir::_Var_>();
  bound_tensors_names_.insert(tensor->name);
}

}  // namespace ir
}  // namespace cinn
