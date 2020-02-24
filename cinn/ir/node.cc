#include "cinn/ir/node.h"

#include "cinn/common/pod_value.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

//! Implementations for Ir Expr Nodes.
// @{
#define __m(t__)                                             \
  template <>                                                \
  void ExprNode<t__>::Accept(cinn::ir::IRVisitor *v) const { \
    v->Visit(const_self());                                  \
  }
NODETY_FORALL(__m)
#undef __m
// @}

std::ostream &operator<<(std::ostream &os, IrNodeTy type) {
  switch (type) {
#define __m(t__)                    \
  case IrNodeTy::t__:               \
    os << "<node: " << #t__ << ">"; \
    break;

    NODETY_FORALL(__m)
#undef __m

    default:
      LOG(FATAL) << "unknown IrNodeTy found";
  }

  return os;
}

Expr::Expr(const Var &var) { *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&var); }
Expr::Expr(const Buffer &buffer) { *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&buffer); }

int32_t Expr::as_int32() const {
  CHECK(type().is_int(32));
  return As<IntImm>()->value;
}

Expr &Expr::operator=(const Expr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

}  // namespace ir
}  // namespace cinn
