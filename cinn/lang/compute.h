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
ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr()> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});
ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});
ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});
ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});
ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});

ir::Tensor Compute(const std::vector<Expr> &domain,
                   std::function<Expr(Expr, Expr, Expr, Expr, Expr)> fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});

ir::Tensor Compute(const std::vector<Expr> &domain,
                   compute_handler_t fn,
                   const std::string &name             = "",
                   const std::vector<Var> &reduce_axis = {},
                   const std::vector<Expr> &shape      = {});
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
 * \brief Call an extern function and get tensors as result.
 *
 * There are two kinds of extern functions distinguish by the return type.
 *
 * 1. Void, there are one or more mutable tensors in the argument list.
 * \code
 * Tensor tuple = Compute({M}, []() { return CallExtern("mkl_gemm", {X, W}); });
 * \endcode
 *
 * To support returning multiple value one time, we include the tuple concept, it is a Tensor with CallOp marked with
 * value_offset(from 0 to num_returns-1).
 *
 * 2. POD value, return an expression directlly, and it can be inline expand in following computations.
 * \code
 * Tensor tanh_out = Compute({M}, [](Var i) { return CallExtern("tanh", X(i)); });
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
 */
Expr CallExtern(const std::string &target, const std::vector<Expr> &args);

Expr CallExtern(const std::string &target, const std::vector<Expr> &args, ir::Tensor &mutable_tensor);  // NOLINT

}  // namespace lang
}  // namespace cinn
