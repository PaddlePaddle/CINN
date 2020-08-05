#pragma once
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {
#define HLIR_DCL_BC_OP(name__) \
  ir::Tensor name__(const ir::Tensor& A, const ir::Tensor& B, const std::string& output_name = "");

HLIR_DCL_BC_OP(Add);
HLIR_DCL_BC_OP(Substract);
HLIR_DCL_BC_OP(Multiply);
HLIR_DCL_BC_OP(Divide);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
