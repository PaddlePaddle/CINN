#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Replace the constant parameter(included in ISL param) to the corresponding integer.
 *
 * e.g.
 *
 * The expression:
 * for (int i = 0; i <= _const_0; i++) ...
 *
 * to
 *
 * for (int i = 0; i < 0; i++)
 */
void ReplaceConstParamToInteger(Expr* e);

}  // namespace cinn::optim
