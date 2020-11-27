#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Expand the Reduce Nodes to Stores.
 * @param e
 */
void ExpandReduce(Expr* e);

}  // namespace cinn::optim
