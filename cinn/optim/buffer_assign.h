#pragma once

#include <map>
#include <set>
#include <string>
#include "cinn/ir/ir.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

/**
 * Assign buffer for tensors those are not marked as compute_inline.
 * @param expr
 * @param stages The stage map.
 */
std::map<std::string, ir::Tensor> InitialAssignBuffer(Expr* expr,
                                                      poly::StageMap stages,
                                                      const std::map<std::string, ir::Tensor>& all_tensor_map,
                                                      const common::Graph* comp_graph,
                                                      const std::set<std::string>& temp_tensor_names);

}  // namespace optim
}  // namespace cinn
