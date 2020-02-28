#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {

using compute_handler_t = std::function<Expr(const std::vector<Expr> &)>;

//! Compute methods for one to five Vars as arguments.
// @{
ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Expr)> fn, const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims, std::function<Expr(Expr, Expr)> fn, const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name = "");
ir::Tensor Compute(const std::vector<int> &dims, compute_handler_t fn, const std::string &name = "");
// @}

}  // namespace lang
}  // namespace cinn
