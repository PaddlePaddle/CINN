#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace ir {

struct ExternOp : public _Operation_ {
  //! The input tensors.
  std::vector<Tensor> inputs;
  //! Symbolic placeholder representation of inputs.
  std::vector<Buffer> input_placeholders;
  //! Symbolic placeholder representation of outputs.
  std::vector<Buffer> output_placeholders;
  //! The statement that generates the computation.
  Stmt body;

  ExternOp() = default;

  static Operation Make(std::string name,
                        std::string tag,
                        std::map<std::string, IrNodeRef> attrs,
                        std::vector<Tensor> inputs,
                        std::vector<Buffer> input_placeholders,
                        std::vector<Buffer> output_placeholders,
                        Stmt body);

  static constexpr char* buffer_get_element = "cinn_buffer_get_element";
};

/**
 * @brief A placeholder op represents an input placeholder.
 */
struct PlaceholderOp : public _Operation_ {
  //! The shape of the input.
  std::vector<Expr> shape;
  //! The data type of the input.
  Type dtype;

  static Operation Make(std::string name, std::vector<Expr> shape, Type dtype);
};

/**
 * @brief A Compute op that compute a tensor on certain domain.
 */
struct ComputeOp : public _Operation_ {
  //! Vars on each axis.
  std::vector<Var> axis;
  //! Var on each reduction axis, if the body is a Reduction.
  std::vector<Var> reduce_axis;
  //! The compute expression.
  std::vector<Expr> body;

  ComputeOp() = default;

  static Operation Make(std::string name,
                        std::string tag,
                        std::map<std::string, IrNodeRef> attrs,
                        std::vector<Var> axis,
                        std::vector<Expr> body);
};

}  // namespace ir
}  // namespace cinn
