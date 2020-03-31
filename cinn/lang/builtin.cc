#include "cinn/ir/ir.h"
#include "cinn/lang/buffer.h"

namespace cinn {
namespace lang {

Expr Sum(Expr body) { return ir::Reduce::Make(ir::Reduce::kSum, ir::Zero(body.type()), body); }

}  // namespace lang
}  // namespace cinn
