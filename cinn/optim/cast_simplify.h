#pragma once

#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Simplify the Cast nodes.
 *
 * There are several patterns:
 * 1. the source and target type are the same, drop the Cast node
 * 2. for intermediate numbers, just replace the Cast node with a Node of the target type
 */
void CastSimplify(Expr* e);

}  // namespace cinn::optim
