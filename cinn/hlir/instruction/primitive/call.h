#pragma once
#include <string>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/shape.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

/**
 * Implement the Call.
 * @param fn_name The call target.
 * @param args The readonly arguments.
 * @param shapes The shapes of the return tensors.
 * @param tensor_names The names of the return tensors.
 * @return The expression of the call.
 */
std::vector<ir::Tensor> CallImpl(const std::string& fn_name,
                                 const std::vector<Expr>& args,
                                 const std::vector<Shape>& shapes,
                                 const std::vector<std::string>& tensor_names,
                                 const std::vector<cinn::common::Type>& types) {
  CHECK_EQ(shapes.size(), tensor_names.size());
  std::vector<cinn::lang::ReturnType> return_types(shapes.size());
  for (int i = 0; i < shapes.size(); i++) {
    return_types[i].name = tensor_names[i];
    return_types[i].dims = shapes[i].ToCinnShape();
    return_types[i].type = types[i];
  }

  return cinn::lang::Call(fn_name, args, return_types);
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
