#pragma once

#include <map>
#include <string>
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
};

}  // namespace ir
}  // namespace cinn
