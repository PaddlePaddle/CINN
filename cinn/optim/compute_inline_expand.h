#pragma once

#include <set>

#include "cinn/cinn.h"

namespace cinn {
namespace optim {

/**
 * Recursive expand the inlined tensors.
 * @param expr the expression to modify.
 * @param tensor_name name of the tensor to expand inline.
 * @param memo a memo to avoid duplicate expand.
 */
void ComputeInlineExpand(Expr* expr);

}  // namespace optim
}  // namespace cinn
