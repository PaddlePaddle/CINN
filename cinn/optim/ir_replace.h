#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Replace the variable \p v to expression \p e in expression \p expr.
void IrReplace(ir::Expr* expr, ir::Expr from, ir::Expr to);

}  // namespace optim
}  // namespace cinn
