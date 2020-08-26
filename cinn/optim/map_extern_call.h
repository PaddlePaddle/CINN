#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Map the Call nodes to external function call.
 * TODO(Suerjomn) consider different backends.
 */
void MapExternCall(Expr *e, Target target);

}  // namespace optim
}  // namespace cinn
