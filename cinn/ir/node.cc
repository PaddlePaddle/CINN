#include "cinn/ir/node.h"

#include "cinn/common/common.h"
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

Expr Zero(const Type &type) {
  if (type.is_float(32)) return Expr(0.f);
  if (type.is_float(64)) return Expr(double(0.));  // NOLINT
  if (type.is_bool()) return Expr(false);
  if (type.is_int(32)) return Expr(int32_t(0));
  if (type.is_int(64)) return Expr(int64_t(0));
  if (type.is_uint(32)) return Expr(uint32_t(0));
  if (type.is_uint(64)) return Expr(uint64_t(0));
  NOT_IMPLEMENTED
  return Expr();
}

Expr::Expr(const Var &var) { *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&var); }

int32_t Expr::as_int32() const {
  CHECK(type().is_int(32));
  return As<IntImm>()->value;
}
int64_t Expr::as_int64() const {
  CHECK(type().is_int(64));
  return As<IntImm>()->value;
}
float Expr::as_float() const {
  CHECK(type().is_float(32));
  return As<FloatImm>()->value;
}
double Expr::as_double() const {
  CHECK(type().is_float(64));
  return As<FloatImm>()->value;
}

Expr &Expr::operator=(const Expr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

Expr::operator Var() {
  auto *x = As<ir::_Var_>();
  CHECK(x);
  return ir::Var(x);
}

}  // namespace ir
}  // namespace cinn
