#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void EliminateBroadcastInForloop(Expr* expr);

}  // namespace optim
}  // namespace cinn
