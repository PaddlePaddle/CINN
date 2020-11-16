#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Compare expression simplify
 * e.g. (1 > 2) => 0, (1 < 2) => 1
 *
 * currently, this can only deal with the simple expression without variables.
 * TODO(Superjomn) Support more complex conditions.
 */
void CompareSimplify(Expr* e);

}  // namespace cinn::optim
