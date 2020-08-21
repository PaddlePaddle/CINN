#pragma once

#include "cinn/ir/ir.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

/**
 * Assign buffer for tensors those are not marked as compute_inline.
 * @param expr
 * @param buffer_shared the clusters that each cluster share the same buffer.
 */
std::map<std::string, ir::Tensor> InitialAssignBuffer(Expr* expr,
                                                      poly::StageMap stages,
                                                      const std::vector<std::set<std::string>>& buffer_shared = {});

}  // namespace optim
}  // namespace cinn
