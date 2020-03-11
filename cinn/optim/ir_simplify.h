#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Simplify the expression.
 * The following cases are supported:
 * a + 0 => a
 * a*0 => 0
 * A[i*0+2*a+3*a+1+2] => A[5*a+3]
 *
 * This only works on the simple IR nodes such as Load, Store, and the math operators such as Add, Sub and so on.
 */
void Simplify(Expr *expr);

}  // namespace optim
}  // namespace cinn
