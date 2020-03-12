#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

Expr Optimize(Expr e);

}  // namespace optim
}  // namespace cinn
