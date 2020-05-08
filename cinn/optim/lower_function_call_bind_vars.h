#pragma once

#include "cinn/ir/ir.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace optim {

void LowerFunctionCallBindVars(Expr *m);

}  // namespace optim
}  // namespace cinn
