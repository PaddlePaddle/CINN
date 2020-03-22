#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Replace the variable \p v to expression \p e in expression \p expr.
void IrReplace(ir::Expr* expr, ir::Var v, ir::Expr e);

}  // namespace optim
}  // namespace cinn
