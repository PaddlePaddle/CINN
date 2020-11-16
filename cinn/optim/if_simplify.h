#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

void IfSimplify(Expr* e);

}  // namespace cinn::optim
