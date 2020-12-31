#include "cinn/lang/builtin.h"

#include <cmath>
#include <limits>
#include <utility>

#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/buffer.h"

namespace cinn {
namespace lang {

Expr logic_and(const std::vector<Expr>& conds) {
  CHECK(!conds.empty());
  auto start = ir::And::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = ir::And::Make(start, conds[i]);
  }
  return start;
}

Expr logic_or(const std::vector<Expr>& conds) {
  CHECK(!conds.empty());
  auto start = ir::Or::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = ir::Or::Make(start, conds[i]);
  }
  return start;
}

//! extern call op
#define EXTERN_CALL_IMP(name__, target__) \
  Expr name__(Expr e) { return ir::Call::Make(e->type(), #target__, {e}, {}, ir::CallType::Extern); }

#define EXTERN_CALL_IMP_NO_VEC(name__, target__)                                                               \
  Expr name__(Expr e) {                                                                                        \
    return ir::Call::Make(                                                                                     \
        e->type(), #target__, {e}, {}, ir::CallType::Extern, ir::FunctionRef(), 0, {{"vectorizable", false}}); \
  }

EXTERN_CALL_IMP(Exp, exp);
EXTERN_CALL_IMP_NO_VEC(Erf, erf);
EXTERN_CALL_IMP(Sqrt, sqrt);
EXTERN_CALL_IMP(Log, log);
EXTERN_CALL_IMP(Log2, log2);
EXTERN_CALL_IMP(Log10, log10);
EXTERN_CALL_IMP(Floor, floor);
EXTERN_CALL_IMP(Ceil, ceil);
EXTERN_CALL_IMP(Round, round);
EXTERN_CALL_IMP(Trunc, trunc);
EXTERN_CALL_IMP(Cos, cos);
EXTERN_CALL_IMP(Sin, sin);
EXTERN_CALL_IMP(Cosh, cosh);
EXTERN_CALL_IMP(Tan, tan);
EXTERN_CALL_IMP(Tanh, tanh);
EXTERN_CALL_IMP(Sinh, sinh);
EXTERN_CALL_IMP_NO_VEC(Acos, acos);
EXTERN_CALL_IMP_NO_VEC(Acosh, acosh);
EXTERN_CALL_IMP_NO_VEC(Asin, asin);
EXTERN_CALL_IMP_NO_VEC(Asinh, asinh);
EXTERN_CALL_IMP_NO_VEC(Atan, atan);
EXTERN_CALL_IMP_NO_VEC(Atanh, atanh);

Expr min_value(const Type& type) {
  CHECK_EQ(type.lanes(), 1);
#define FOR_CASE(type__)                             \
  if (type == type_of<type__>()) {                   \
    return Expr(std::numeric_limits<type__>::min()); \
  }
  FOR_CASE(int32_t)
  FOR_CASE(int64_t)
  FOR_CASE(uint32_t)
  FOR_CASE(uint64_t)
  FOR_CASE(float)
  FOR_CASE(double)
#undef FOR_CASE
  return Expr();
}

Expr max_value(const Type& type) {
  CHECK_EQ(type.lanes(), 1);

#define FOR_CASE(type__)                             \
  if (type == type_of<type__>()) {                   \
    return Expr(std::numeric_limits<type__>::max()); \
  }
  FOR_CASE(int32_t)
  FOR_CASE(int64_t)
  FOR_CASE(uint32_t)
  FOR_CASE(uint64_t)
  FOR_CASE(float)
  FOR_CASE(double)
#undef FOR_CASE

  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr Abs(Expr e) {
  Type type      = e->type();
  Type bool_type = Bool(type.lanes());
  if (type.is_uint()) {
    return e;
  } else if (type.is_int()) {
    auto node = e.As<ir::IntImm>();
    if (node) {
      return make_const(type, std::abs(node->value));
    }
    return ir::Select::Make(e > make_const(e->type(), 0), e, -e);
  } else if (type.is_float()) {
    auto node = e.As<ir::FloatImm>();
    if (node) {
      return make_const(type, std::fabs(node->value));
    }
    return CallExtern("fabs", {e});
  }
}

Expr IsNan(Expr e) {
  Type type = e->type();
  if (type.is_int() || type.is_uint()) {
    return common::make_bool(false, type.lanes());
  } else if (type.is_float()) {
    auto* node = e.As<ir::FloatImm>();
    if (node) {
      return common::make_bool(std::isnan(node->value), type.lanes());
    }
    Expr arg = e;
    if (type.bits() == 16) {
      arg = ir::Cast::Make(Float(32), std::move(e));
    }
    return CallExtern("isnan", {arg}, {{"vectorizable", false}});
  } else {
    LOG(FATAL) << type << "is not supported for isnan op.";
    return e;
  }
}

Expr Infinity(const Type& type) {
  CHECK_EQ(type.lanes(), 1U);
  if (type.is_float()) {
    if (type.bits() == 64) {
      return make_const(type, std::numeric_limits<double>::infinity());
    } else if (type.bits() == 32 || type.bits() == 16) {
      return make_const(type, std::numeric_limits<float>::infinity());
    }
  }
  LOG(FATAL) << "Cannot decide infinity for type " << type;
  return Expr();
}

Expr IsInf(Expr e) {
  Type type = e->type();
  if (type.is_int() || type.is_uint()) {
    return common::make_bool(false, type.lanes());
  } else if (type.is_float()) {
    return ir::EQ::Make(Abs(e), Infinity(type)) && !IsNan(e);
  } else {
    LOG(FATAL) << type << "is not supported for isinf op.";
    return e;
  }
}

Expr IsFinite(Expr e) { return !IsInf(e) && !IsNan(e); }

}  // namespace lang
}  // namespace cinn
