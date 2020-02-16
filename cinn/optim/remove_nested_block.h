/**
 * This file implements the strategy to remove the unnecessary nested block.
 */
#pragma once
#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Remove the unecessary nested block.
 */
void RemoveNestedBlock(Expr* e);

}  // namespace optim
}  // namespace cinn
