#pragma once
#include <utility>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Shallow copy an expression.
Expr IRCopy(Expr x);

std::vector<Expr> IRCopy(const std::vector<Expr>& x);

}  // namespace optim
}  // namespace cinn
