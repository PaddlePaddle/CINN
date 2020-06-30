#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void CacheReadWriteReplace(Expr* expr);

}  // namespace optim
}  // namespace cinn
