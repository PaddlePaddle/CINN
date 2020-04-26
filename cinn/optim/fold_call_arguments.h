#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void FoldCallArguments(Expr* expr);

}  // namespace optim
}  // namespace cinn
