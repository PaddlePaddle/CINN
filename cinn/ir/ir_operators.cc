#include "cinn/ir/ir_operators.h"

#include <limits>

#include "cinn/lang/compute.h"

#include "cinn/common/type.h"

namespace cinn {
namespace ir {

Expr logic_and(const std::vector<Expr>& conds) {
  CHECK(!conds.empty());
  auto start = And::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = And::Make(start, conds[i]);
  }
  return start;
}

Expr logic_or(const std::vector<Expr>& conds) {
  CHECK(!conds.empty());
  auto start = Or::Make(conds[0], conds[1]);
  for (int i = 2; i < conds.size(); i++) {
    start = Or::Make(start, conds[i]);
  }
  return start;
}

Expr operator<<(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a   = a.type();
  Type t_b   = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      CHECK(int_b->value >= 0 && int_b->value < t_a.bits())
          << "Shift amount must be non-negative and less than " << t_a.bits() << " for type " << t_a << std::endl;
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value << int_b->value);
    }
  }
  return lang::CallExtern("left_shift", {a, b});
}

Expr operator>>(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a   = a.type();
  Type t_b   = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_b) {
      CHECK(int_b->value >= 0 && int_b->value < t_a.bits())
          << "Shift amount must be non-negative and less than " << t_a.bits() << " for type " << t_a << std::endl;
      if (int_b->value == 0) return a;
    }
    if (int_a && int_b) {
      return Expr(int_a->value >> int_b->value);
    }
  }
  return lang::CallExtern("right_shift", {a, b});
}

Expr operator|(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a   = a.type();
  Type t_b   = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value | int_b->value);
    }
  }
  return lang::CallExtern("bitwise_or", {a, b});
}

Expr operator&(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a   = a.type();
  Type t_b   = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value & int_b->value);
    }
  }
  return lang::CallExtern("bitwise_and", {a, b});
}

Expr operator^(Expr a, Expr b) {
  CHECK(a.type().is_int() || a.type().is_uint());
  CHECK(b.type().is_int() || b.type().is_uint());
  auto int_a = a.As<IntImm>();
  auto int_b = b.As<IntImm>();
  Type t_a   = a.type();
  Type t_b   = b.type();
  if (t_a.is_index_type() && t_b.is_index_type()) {
    if (int_a && int_b) {
      return Expr(int_a->value ^ int_b->value);
    }
  }
  return lang::CallExtern("bitwise_xor", {a, b});
}

Expr operator~(Expr a) {
  CHECK(a.type().is_int() || a.type().is_uint());
  return lang::CallExtern("bitwise_not", {a});
}

//! extern call op
#define EXTERN_CALL_IMP(name__, target__) \
  Expr name__(Expr e) { return lang::CallExtern(#target__, {e}); }

EXTERN_CALL_IMP(Exp, exp);
EXTERN_CALL_IMP(Erf, erf);
EXTERN_CALL_IMP(Sqrt, sqrt);
EXTERN_CALL_IMP(Log, log);
EXTERN_CALL_IMP(Log2, log2);
EXTERN_CALL_IMP(Log10, log10);
EXTERN_CALL_IMP(Floor, floor);
EXTERN_CALL_IMP(Ceil, ceil);
EXTERN_CALL_IMP(Round, round);
EXTERN_CALL_IMP(Trunc, trunc);
EXTERN_CALL_IMP(Cos, cos);
EXTERN_CALL_IMP(Cosh, cosh);
EXTERN_CALL_IMP(Tan, tan);
EXTERN_CALL_IMP(Sin, sin);
EXTERN_CALL_IMP(Sinh, sinh);
EXTERN_CALL_IMP(Acos, acos);
EXTERN_CALL_IMP(Acosh, acosh);
EXTERN_CALL_IMP(Asin, asin);
EXTERN_CALL_IMP(Asinh, asinh);
EXTERN_CALL_IMP(Atan, atan);
EXTERN_CALL_IMP(Atanh, atanh);
EXTERN_CALL_IMP(Isnan, isnan);
EXTERN_CALL_IMP(Tanh, tanh);
EXTERN_CALL_IMP(Isfinite, isfinite);
EXTERN_CALL_IMP(Isinf, isinf);

Expr min_value(const Type& type) {
  CHECK_EQ(type.lanes(), 1);
  if (type.is_int()) {
    if (type.bits() == 64) {
      return Expr(std::numeric_limits<int64_t>::lowest());
    } else if (type.bits() < 64) {
      int64_t val = 1;
      val         = -(val << (type.bits() - 1));
      return Expr(val);
    }
  } else if (type.is_uint()) {
    return Expr(0);
  } else if (type.is_float()) {
    if (type.bits() == 64) {
      return Expr(std::numeric_limits<double>::lowest());
    } else if (type.bits() == 32) {
      return Expr(std::numeric_limits<float>::lowest());
    } else if (type.bits() == 16) {
      return Expr(-65504.0);
    }
  }
  LOG(FATAL) << "Cannot decide min_value for type" << type;
  return Expr();
}

}  // namespace ir
}  // namespace cinn
