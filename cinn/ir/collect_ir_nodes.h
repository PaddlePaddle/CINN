#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

/**
 * Collect the tensors(without duplication) in the expressoin.
 */
std::set<Expr> CollectIRNodes(Expr x, std::function<bool(const Expr*)>&& teller);

}  // namespace ir
}  // namespace cinn
