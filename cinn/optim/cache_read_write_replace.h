#pragma once
#include <string>

#include "cinn/ir/ir.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

void CacheReadWriteReplace(Expr* expr, poly::StageMap stages, std::map<std::string, ir::Tensor>* global_tensor_map);

}  // namespace optim
}  // namespace cinn
