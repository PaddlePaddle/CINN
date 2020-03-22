#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void UnrollLoop(Expr* expr);

}  // namespace optim
}  // namespace cinn
