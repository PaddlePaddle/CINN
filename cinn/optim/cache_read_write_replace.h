#pragma once
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void CacheReadWriteReplace(Expr* expr, std::map<std::string, ir::Tensor>* global_tensor_map);

}  // namespace optim
}  // namespace cinn
