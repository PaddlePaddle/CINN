#include "cinn/ir/buffer.h"

#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

const _Buffer_ *Buffer::operator->() const { return IrNodeRef::As<_Buffer_>(); }

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

void _Buffer_::Accept(IrVisitor *v) const { v->Visit(this); }
IrNodeTy _Buffer_::node_type() const { return _node_type_; }

}  // namespace ir
}  // namespace cinn
