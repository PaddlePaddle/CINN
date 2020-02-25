#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {

//! Compute methods for one to five Vars as arguments.
// @{
ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var)> fn, const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var)> fn, const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Var, Var, Var)> fn, const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Var, Var, Var, Var)> fn,
                   const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Var, Var, Var, Var, Var)> fn,
                   const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(const std::vector<Var> &)> fn,
                   const std::string &name = "");
// @}

}  // namespace lang
}  // namespace cinn
