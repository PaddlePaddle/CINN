#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 *
 * Replace the _cp_C_[0-9]+ to [0-9]+ in expression
 *
 * e.g.
 *
 * if ((1 && (_cp_C_0 <= -1))) { }
 *
 * to
 *
 * if ((1 && (0 <= -1))) { }
 *
 * @param expr
 */
void ProcessIslParamInExpr(Expr* expr);

}  // namespace optim
}  // namespace cinn

