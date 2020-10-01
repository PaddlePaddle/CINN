#include "cinn/ir/ir_operators.h"

#include <limits>

#include "cinn/common/type.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace ir {

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

}  // namespace ir
}  // namespace cinn
