#pragma once
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

//! Get the ALL of the conditions.
Expr logic_and(const std::vector<Expr>& conds);
Expr logic_or(const std::vector<Expr>& conds);

}  // namespace ir
}  // namespace cinn
