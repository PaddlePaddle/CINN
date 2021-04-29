#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

std::vector<std::string> CollectUndefinedVars(Expr* e);

}  // namespace cinn::optim
