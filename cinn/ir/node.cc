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
  void ExprNode<t__>::Accept(cinn::ir::IrVisitor *v) const { \
    v->Visit(const_self());                                  \
  }
NODETY_FORALL(__m)
#undef __m
// @}

//! Implementations for Ir Stmt Nodes.
// @{
#define __m(t__)                                             \
  template <>                                                \
  void StmtNode<t__>::Accept(cinn::ir::IrVisitor *v) const { \
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

}  // namespace ir
}  // namespace cinn
