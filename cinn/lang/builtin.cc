#include "cinn/lang/builtin.h"

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/buffer.h"

namespace cinn {
namespace lang {

Expr Sum(Expr body) { return ir::Reduce::Make(ir::Reduce::kSum, common::make_const(body.type(), 0), body); }
Expr Mul(Expr body) { return ir::Reduce::Make(ir::Reduce::kMul, common::make_const(body.type(), 1), body); }

}  // namespace lang
}  // namespace cinn
