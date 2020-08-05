#pragma once
#include <tuple>
#include <utility>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void InsertDebugLogCallee(Expr* e);

}  // namespace optim
}  // namespace cinn
