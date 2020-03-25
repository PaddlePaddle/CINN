#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void IrEliminateMod(Expr* expr);

}  // namespace optim
}  // namespace cinn
