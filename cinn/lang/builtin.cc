#include "cinn/lang/builtin.h"

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
  Expr name__(Expr e) { return CallExtern(#target__, {e}); }

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

}  // namespace lang
}  // namespace cinn
