#pragma once
#include <functional>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {

//! Compute methods for one to five Vars as arguments.
// @{
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(Var)> fn);
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(Var, Var)> fn);
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(Var, Var, Var)> fn);
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(Var, Var, Var, Var)> fn);
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(Var, Var, Var, Var, Var)> fn);
ir::Tensor Compute(const std::vector<int>& dims, std::function<Expr(const std::vector<Var>&)> fn);
// @}

namespace detail {
//! Generate `naxis` axis using the global names (i,j,k...).
std::vector<Var> GenDefaultAxis(int naxis);
}  // namespace detail

}  // namespace lang
}  // namespace cinn
