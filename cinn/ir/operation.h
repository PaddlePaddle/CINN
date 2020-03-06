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

/*
struct ExternOp : public _Operation_ {
  //! The input tensors.
  std::vector<Tensor> inputs;
  //! Symbolic placeholder representation of inputs.
  std::vector<Buffer> input_placeholders;
  //! Symbolic placeholder representation of outputs.
  std::vector<Buffer> output_placeholders;
  //! The statement that generates the computation.
  Expr body;

  ExternOp() = default;

  static Operation Make(std::string name,
                        std::string tag,
                        std::map<std::string, IrNodeRef> attrs,
                        std::vector<Tensor> inputs,
                        std::vector<Buffer> input_placeholders,
                        std::vector<Buffer> output_placeholders,
                        Expr body);

  static constexpr char buffer_get_element[] = "cinn_buffer_get_element";
};
 */

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

  static constexpr char *__func_type__ = "placeholder";
};

/**
 * @brief A Compute op that compute a tensor on certain domain.
 */
struct ComputeOp : public _Operation_ {
  using handle_t = std::function<Expr(const std::vector<Expr> &)>;
  //! Vars on each axis.
  std::vector<Var> axis;
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
                        const std::string &tag,
                        const std::map<std::string, IrNodeRef> &attrs,
                        handle_t handle,
                        const std::vector<Expr> &domain);

  static Operation Make(const std::string &name,
                        const std::string &tag,
                        const std::map<std::string, IrNodeRef> &attrs,
                        const std::vector<Var> &axis,
                        const std::vector<Expr> &body,
                        const std::vector<Expr> &shape);

  const char *func_type() const override;

  static constexpr char *__func_type__ = "compute";
};

}  // namespace ir
}  // namespace cinn
