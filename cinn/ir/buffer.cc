#include "cinn/ir/buffer.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

std::string TensorGetBufferName(const _Tensor_ *tensor) {
  CHECK(!tensor->name.empty());
  CHECK(!utils::Startswith(tensor->name, "_")) << "the name with prefix _ is not allowed for tensor";
  return "_" + tensor->name;
}
std::string BufferGetTensorName(const _Buffer_ *buffer) {
  CHECK(!buffer->name.empty());
  CHECK(utils::Startswith(buffer->name, "_")) << "buffer's name should start with _";
  return buffer->name.substr(1);
}

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
                      int offset_factor,
                      Target target) {
  CHECK(dtype.valid());
  auto *node           = common::make_shared<_Buffer_>();
  node->shape          = shape;
  node->strides        = strides;
  node->elem_offset    = elem_offset;
  node->name           = name;
  node->scope          = scope;
  node->data_alignment = data_alignment;
  node->offset_factor  = offset_factor;
  node->target         = target;
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
  if (name.empty()) name = TensorGetBufferName(tensor);
  CHECK(!tensor->shape.empty()) << "Tensor should have shape to bind to a Buffer";
  shape = tensor->shape;
  binded_tensors_names_.insert(tensor->name);
}

Expr Buffer::LoadExpr(const std::vector<Expr> &indice) const {
  auto *node = operator->();
  return Load::Make(Expr(*this), AbsOffset(indice));
}

Expr Buffer::StoreExpr(const std::vector<Expr> &indice, Expr value) const {
  auto *node = operator->();
  return Store::Make(Expr(*this), value, AbsOffset(indice));
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

Expr Buffer::DestroyExpr() const {
  auto *node = operator->();
  return ir::Call::Make(
      Void(), runtime::buffer_destroy, {ir::_Var_::Make(node->name, node->type())}, Call::CallType::Intrinsic);
}

}  // namespace ir
}  // namespace cinn
