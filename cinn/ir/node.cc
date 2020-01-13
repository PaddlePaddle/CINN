#include "cinn/ir/node.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

//! Implementations for Ir nodes.
// @{
#define __m(t__) \
  template <>    \
  void ExprNode<t__>::Accept(cinn::ir::IRVisitor *v) const {}
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

}  // namespace ir
}  // namespace cinn
