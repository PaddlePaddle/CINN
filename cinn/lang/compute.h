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
                   std::function<Expr()> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {});
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

//! Call an internal function.
ir::Tensor Call(const std::string &target,
                Type type,
                const std::vector<Expr> &dims,
                const std::vector<Expr> &args,
                const std::string &name);

struct ReturnType {
  Type type;
  std::vector<Expr> dims;
  std::string name;
};

/**
 * Call a lowered function and return one or more tensors as result.
 * @param target The name of the function to call.
 * @param args The readonly arguments(while the mutable tensors are return result).
 * @param return_types The types of the return values.
 * @return Return one or more tensors as result.
 */
std::vector<ir::Tensor> Call(const std::string &target,
                             const std::vector<Expr> &args,
                             const std::vector<ReturnType> &return_types);

/**
 * \brief Call an extern function and get a tensor as result.
 *
 * \code
 * auto tanh_out = CallIntrinsic("tanh", {X}, ReturnType{x.type(), X.shape, "x.tanh");
 * \endcode
 *
 * \code
 * auto gemm_out = CallIntrinsic("gemm_mkl", {X.view(i), W}, {i}, ReturnType{x.type(), {}, "gemm.out");
 * \endcode
 *
 * Will generate something like
 *
 * \code
 * for (i) {
 *   gemm_mkl(X[i], gemm_out[i])
 * }
 * \endcode
 *
 * @param target The name of the function to call.
 * @param args The readonly arguments(while there should be only one tensor as result).
 * @param preceding_axis The preceeding axis.
 * @param ret_type The type of the return value.
 * @return The only one result.
 */
ir::Tensor CallExtern0(const std::string &target,
                       const std::vector<Expr> &args,
                       ReturnType ret_type,
                       int preceding_axis = -1);

Expr CallExtern(const std::string &target, const std::vector<Expr> &args);

}  // namespace lang
}  // namespace cinn
