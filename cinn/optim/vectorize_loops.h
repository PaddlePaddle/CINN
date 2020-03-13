#pragma once

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

void VectorizeLoops(Expr* expr, const Target& target);

}  // namespace optim
}  // namespace cinn
