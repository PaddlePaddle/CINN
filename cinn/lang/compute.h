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
// The shape are constant integers.
ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
ir::Tensor Compute(const std::vector<Expr> &dims,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});

ir::Tensor Compute(const std::vector<Expr> &dims,
                   compute_handler_t fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
// @}

}  // namespace lang
}  // namespace cinn
