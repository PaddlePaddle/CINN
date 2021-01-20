#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Simplify the ir like
 *
 * for (i, 0, 1) { statement }
 *
 * to
 *
 * statement
 */
void SimplifyIdentityDomainForloop(Expr* e);

}  // namespace optim
}  // namespace cinn