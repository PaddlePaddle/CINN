#pragma once
#include <string>
#include <vector>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {

template <typename T = int>
std::vector<Expr> ToCinnExprs(const std::vector<T>& args) {
  std::vector<Expr> exprs;
  std::transform(args.begin(), args.end(), std::back_inserter(exprs), [](const T& arg) { return Expr(arg); });
  return exprs;
}

}  // namespace hlir
}  // namespace cinn
