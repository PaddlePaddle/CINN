#pragma once
#include <vector>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

//-- left hand --
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator+(Expr a, POD b) {
  return Add::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator-(Expr a, POD b) {
  return Sub::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator*(Expr a, POD b) {
  return Mul::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator/(Expr a, POD b) {
  return Div::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator%(Expr a, POD b) {
  return Mod::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<(Expr a, POD b) {
  return LT::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<=(Expr a, POD b) {
  return LE::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>(Expr a, POD b) {
  return GT::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>=(Expr a, POD b) {
  return GE::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator==(Expr a, POD b) {
  return EQ::Make(Expr(a), Expr(b));
}

//- right hand --
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator+(POD a, Expr b) {
  return Add::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator-(POD a, Expr b) {
  return Sub::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator*(POD a, Expr b) {
  return Mul::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator/(POD a, Expr b) {
  return Div::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator%(POD a, Expr b) {
  return Mod::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<(POD a, Expr b) {
  return LT::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator<=(POD a, Expr b) {
  return LE::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>(POD a, Expr b) {
  return GT::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator>=(POD a, Expr b) {
  return GE::Make(Expr(a), Expr(b));
}
template <typename POD, typename = typename std::enable_if<std::is_pod<POD>::value>::type>
Expr operator==(POD a, Expr b) {
  return EQ::Make(Expr(a), Expr(b));
}

//--
inline Expr operator+(Expr a, Expr b) { return Add::Make(a, b); }
inline Expr operator-(Expr a, Expr b) { return Sub::Make(a, b); }
inline Expr operator*(Expr a, Expr b) { return Mul::Make(a, b); }
inline Expr operator/(Expr a, Expr b) { return Div::Make(a, b); }
inline Expr operator%(Expr a, Expr b) { return Mod::Make(a, b); }

inline Expr operator&&(Expr a, Expr b) { return And::Make(Expr(a), Expr(b)); }
inline Expr operator||(Expr a, Expr b) { return Or::Make(Expr(a), Expr(b)); }
inline Expr operator>=(Expr a, Expr b) { return GE::Make(Expr(a), Expr(b)); }
inline Expr operator<=(Expr a, Expr b) { return LE::Make(Expr(a), Expr(b)); }
inline Expr operator>(Expr a, Expr b) { return GT::Make(Expr(a), Expr(b)); }
inline Expr operator<(Expr a, Expr b) { return LT::Make(Expr(a), Expr(b)); }

inline Expr operator-(Expr a) { return Minus::Make(Expr(a)); }
inline Expr operator!(Expr a) { return Not::Make(Expr(a)); }

Expr operator<<(Expr a, Expr b);
Expr operator>>(Expr a, Expr b);
Expr operator^(Expr a, Expr b);
Expr operator|(Expr a, Expr b);
Expr operator&(Expr a, Expr b);
Expr operator~(Expr a);

//! Get the ALL of the conditions.
Expr logic_and(const std::vector<Expr>& conds);
Expr logic_or(const std::vector<Expr>& conds);

//! extern call op
#define EXTERN_CALL_DCL(name__) Expr name__(Expr e);

EXTERN_CALL_DCL(Exp);
EXTERN_CALL_DCL(Erf);
EXTERN_CALL_DCL(Sqrt);
EXTERN_CALL_DCL(Log);
EXTERN_CALL_DCL(Log2);
EXTERN_CALL_DCL(Log10);
EXTERN_CALL_DCL(Floor);
EXTERN_CALL_DCL(Ceil);
EXTERN_CALL_DCL(Round);
EXTERN_CALL_DCL(Trunc);
EXTERN_CALL_DCL(Cos);
EXTERN_CALL_DCL(Cosh);
EXTERN_CALL_DCL(Tan);
EXTERN_CALL_DCL(Sin);
EXTERN_CALL_DCL(Sinh);
EXTERN_CALL_DCL(Acos);
EXTERN_CALL_DCL(Acosh);
EXTERN_CALL_DCL(Asin);
EXTERN_CALL_DCL(Asinh);
EXTERN_CALL_DCL(Atan);
EXTERN_CALL_DCL(Atanh);
EXTERN_CALL_DCL(Isnan);
EXTERN_CALL_DCL(Tanh);
EXTERN_CALL_DCL(Isfinite);
EXTERN_CALL_DCL(Isinf);

inline Expr Sigmoid(Expr e) {
  auto one = make_const(e->type(), 1);
  return one / (one + Exp(-e));
}

inline Expr Sign(Expr e) {
  auto zero    = make_const(e->type(), 0);
  auto one     = make_const(e->type(), 1);
  auto neg_one = make_const(e->type(), -1);
  auto ret1    = Select::Make(e > zero, one, zero);
  auto ret2    = Select::Make(e < zero, neg_one, ret1);
  return ret2;
}

inline Expr Abs(Expr e) { return Select::Make(e > make_const(e->type(), 0), e, -e); }

inline Expr Rsqrt(Expr e) {
  auto one = make_const(e->type(), 1);
  return one / Sqrt(e);
}

inline Expr Negative(Expr e) { return -e; }
inline Expr Identity(Expr e) { return e; }
inline Expr LogicalNot(Expr e) { return !e; }
inline Expr BitwiseNot(Expr e) { return ~e; }

template <typename T>
inline Expr Relu(Expr e, T threshold = static_cast<T>(0)) {
  return Max::Make(e, make_const(e->type(), threshold));
}

template <typename T>
inline Expr Relu6(Expr e, T threshold = static_cast<T>(0)) {
  return Min::Make(Max::Make(e, make_const(e->type(), threshold)), make_const(e->type(), 6));
}

inline Expr LeakyRelu(Expr e, double alpha) {
  auto zero = make_const(e->type(), 0);
  return Select::Make(e > zero, e, e * make_const(e->type(), alpha));
}

inline Expr LeakyRelu(Expr e, Expr alpha) {
  auto zero = make_const(e->type(), 0);
  return Select::Make(e > zero, e, e * alpha);
}

inline Expr ReduceSum(Expr e, Expr initial) {
  if (!initial.defined()) {
    initial = Zero(e->type());
  }
  return Reduce::Make(Reduce::kSum, initial, e);
}
inline Expr ReduceMul(Expr e, Expr initial) {
  if (!initial.defined()) {
    initial = make_const(e->type(), 1);
  }
  return Reduce::Make(Reduce::kMul, initial, e);
}
inline Expr ReduceMax(Expr e, Expr initial) { return Reduce::Make(Reduce::kMax, initial, e); }
inline Expr ReduceMin(Expr e, Expr initial) { return Reduce::Make(Reduce::kMin, initial, e); }

Expr min_value(const Type& type);

}  // namespace ir
}  // namespace cinn
