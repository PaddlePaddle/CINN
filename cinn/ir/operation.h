#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace ir {

/**
 * @brief A placeholder op represents an input placeholder.
 */
struct PlaceholderOp : public _Operation_ {
  //! The shape of the input.
  std::vector<Expr> shape;
  //! The data type of the input.
  Type dtype;

  static Operation Make(const std::string &name, const std::vector<Expr> &shape, Type dtype);

  const char *func_type() const override;

  static char const *__func_type__;
};

struct CallOp : public _Operation_ {
  const std::string &target() const;

  Expr call_expr;

  std::vector<Expr> &read_args();
  std::vector<Expr> &write_args();
  const std::vector<Expr> &read_args() const;
  const std::vector<Expr> &write_args() const;
  std::vector<Expr> args() const;

  //! A reference to the target LoweredFunc if this CallOp calls an generated LoweredFunc.
  Expr func;

  // the offset int the tuple of return values.
  int arg_slot{-1};

  CallOp() = default;

  static Operation Make(const std::string &call_target,
                        const std::vector<Expr> &arg_list,
                        int value_slot,
                        Expr call_op);

  const char *func_type() const override;

  static char const *__func_type__;
};

/**
 * The operation of the preceding view of a tensor.
 */
struct PrecedingViewOp : public _Operation_ {
  Expr tensor;

  int preceding_axis{-1};

  static Operation Make(const Tensor &tensor, int preceding_axis);

  const char *func_type() const override;

  static char const *__func_type__;
};

/**
 * @brief A Compute op that compute a tensor on certain domain.
 */
struct ComputeOp : public _Operation_ {
  using handle_t = std::function<Expr(const std::vector<Expr> &)>;
  //! Var on each reduction axis, if the body is a Reduction.
  std::vector<Var> reduce_axis;
  //! Shape of the output.
  std::vector<Expr> shape;
  //! The compute expression.
  std::vector<Expr> body;
  //! The functor to generate the body, used to inline the expression if needed.
  handle_t producer_fn;

  ComputeOp() = default;

  static Operation Make(const std::string &name,
                        ComputeOp::handle_t handle,
                        const std::vector<Expr> &shape,
                        const std::vector<Expr> &domain,
                        const std::vector<Var> &reduce_axis           = {},
                        const std::map<std::string, IrNodeRef> &attrs = {},
                        const std::string &tag                        = "");

  const char *func_type() const override;

  static const char *__func_type__;
};

}  // namespace ir
}  // namespace cinn
