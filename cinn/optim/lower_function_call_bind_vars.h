#pragma once

#include "cinn/ir/ir.h"
#include "cinn/ir/module.h"

namespace cinn {
namespace optim {

void LowerFunctionCallBindVars(Expr *m);

}  // namespace optim
}  // namespace cinn
