#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Shallow copy an expression.
Expr IRCopy(Expr x);

}  // namespace optim
}  // namespace cinn
