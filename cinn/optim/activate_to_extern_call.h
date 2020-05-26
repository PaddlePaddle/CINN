#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace Activate IR nodes with extern call if needed.
 * TODO(Suerjomn) consider different backends.
 */
void ActivateToExternCall(Expr *e);

}  // namespace optim
}  // namespace cinn
