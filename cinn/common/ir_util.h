#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr IndiceToAbsOffset(const std::vector<Expr> &shape, const std::vector<Expr> &indices);
Expr IndiceToAbsOffset(const std::vector<int> &shape, const std::vector<Expr> &indices);

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape, int preceding_n_axis);

Expr CastIfNeeded(Expr body, Type type);

//! Substitute vars to other expressions.
//! @param expr The expression to do modification.
//! @param var_map The map from variables to the target expressions.
void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map);

template <typename T>
Expr make_const(Type t, T v);

//! Get a stack of forloops(For and PolyFor nodes) to a Store node target to \p tensor_name
std::vector<Expr *> GetForloopStackToStore(Expr *expr, const std::string &tensor_name);

// make const
// @{
inline Expr make_const(int32_t x) { return make_const(Int(32), static_cast<int64_t>(x)); }
inline Expr make_const(int64_t x) { return make_const(Int(64), static_cast<int64_t>(x)); }
inline Expr make_const(float x) { return make_const(Float(32), static_cast<double>(x)); }
inline Expr make_const(double x) { return make_const(Float(64), static_cast<double>(x)); }
inline Expr make_const(bool x) { return make_const(Bool(1), static_cast<bool>(x)); }
// @}

//! maker for some general consts.
// @{
template <typename T = int32_t>
inline Expr make_zero() {
  return make_const(static_cast<T>(0));
}
template <typename T = int32_t>
inline Expr make_one() {
  return make_const(static_cast<T>(1));
}
inline Expr make_bool(bool x) { return common::make_shared<ir::UIntImm>(Bool(), x); }
// @}

/**
 * \brief Check all the tensors are unique in an expression.
 */
void CheckTensorUniqueInExpr(Expr expr);

/**
 * \brief Check all the buffers are uniuqe in an expression.
 */
void CheckBufferUniqueInExpr(Expr expr);

std::vector<std::string> GatherItersToTensorProducer(const std::string &target_tensor_name, Expr *expr);

bool is_zero(Expr v);

bool MathEqual(const Expr &a, const Expr &b);

//! helper function to get a ir::Select node.
Expr select(Expr cond, Expr true_value, Expr false_value);

//! helper function to get the And of all the conditions.
Expr and_all(const std::vector<Expr> &conds);

//! helper function to get the Or of all the conditions.
Expr or_any(const std::vector<Expr> &conds);

//! Cast the expression \p e to type \type.
Expr cast(Expr e, Type type);

template <typename T>
Expr make_const(Type t, T v) {
  if (t.is_vector()) {
    if (t.type() == Type::type_t::Int) {
      return ir::Broadcast::Make(make_shared<ir::IntImm>(t.ElementOf(), v), t.lanes());
    } else {
      return ir::Broadcast::Make(make_shared<ir::FloatImm>(t.ElementOf(), v), t.lanes());
    }
  } else {
    if (t.type() == Type::type_t::Int) {
      return make_shared<ir::IntImm>(t, v);
    } else {
      return make_shared<ir::FloatImm>(t, v);
    }
  }
  return Expr();
}

}  // namespace common
}  // namespace cinn
