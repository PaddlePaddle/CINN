#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace Activate IR nodes with extern call if needed.
 * TODO(Suerjomn) consider different backends.
 */
void ActivateToExternCall(Expr *e, Target target);

}  // namespace optim
}  // namespace cinn
