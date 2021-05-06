#pragma once
#include <string>
#include <vector>

#include "cinn/ir/ir.h"
namespace cinn::optim {

/**
 * Collect undefined vars in the scope.
 *
 * e.g.
 *
 * The expression:
 * for i
 *  for j
 *    a[i, j] = b[i, j]
 *
 * here a, b are vars without definition
 */
std::vector<std::string> CollectUndefinedVars(Expr* e);

}  // namespace cinn::optim
