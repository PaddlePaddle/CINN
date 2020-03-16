#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr ExpandTo1DIndice(const std::vector<Expr> &shape, const std::vector<Expr> &indices);

Expr ExpandTo1DIndice(const std::vector<int> &shape, const std::vector<Expr> &indices);

bool is_zero(Expr v);

}  // namespace common
}  // namespace cinn
