#include "cinn/ir/buffer.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/runtime/intrinsic.h"

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
  CHECK(!tensor->shape.empty()) << "Tensor should have shape to bind to a Buffer";
  shape = tensor->shape;
  if (!data.defined()) data = _Var_::Make(name, tensor->type()).As<ir::_Var_>();
  bound_tensors_names_.insert(tensor->name);
}

Expr Buffer::LoadExpr(const std::vector<Expr> &indice) const {
  auto *node = operator->();
  return Load::Make(node->data, AbsOffset(indice));
}

Expr Buffer::StoreExpr(const std::vector<Expr> &indice, Expr value) const {
  auto *node = operator->();
  return Store::Make(node->data, value, AbsOffset(indice));
}

Expr Buffer::AbsOffset(const std::vector<Expr> &indice) const {
  auto *node = operator->();
  CHECK(!node->shape.empty());
  CHECK_EQ(node->shape.size(), indice.size()) << "shape and indice not match";
  Expr res = indice.front() * node->shape[1];
  for (int i = 1; i < node->shape.size() - 1; i++) {
    res = res + indice[i] * node->shape[i + 1];
  }
  if (node->shape.size() > 1) res = res + indice.back();
  return res;
}

Expr Buffer::CreateExpr() const {
  const auto *node = operator->();
  std::vector<Expr> args;
  args.push_back(node->data);
  return ir::Call::Make(Void(), runtime::buffer_create, {node->data}, Call::CallType::Intrinsic);
}

Expr Buffer::DestroyExpr() const {
  auto *node = operator->();
  return ir::Call::Make(Void(), runtime::buffer_destroy, {node->data}, Call::CallType::Intrinsic);
}

}  // namespace ir
}  // namespace cinn
