#pragma once

#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {
/**
 * @brief Unary primitive emitters
 *
 * @param A The input Tensor
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
#define HLIR_DCL_UNARY_PE(name__) \
  ir::Tensor name__(const ir::Tensor& A, const std::string& output_name = "T_" #name__ "_out");

HLIR_DCL_UNARY_PE(Exp);
HLIR_DCL_UNARY_PE(Erf);
HLIR_DCL_UNARY_PE(Sqrt);
HLIR_DCL_UNARY_PE(Log);
HLIR_DCL_UNARY_PE(Log2);
HLIR_DCL_UNARY_PE(Log10);
HLIR_DCL_UNARY_PE(Floor);
HLIR_DCL_UNARY_PE(Ceil);
HLIR_DCL_UNARY_PE(Round);
HLIR_DCL_UNARY_PE(Trunc);
HLIR_DCL_UNARY_PE(Cos);
HLIR_DCL_UNARY_PE(Cosh);
HLIR_DCL_UNARY_PE(Tan);
HLIR_DCL_UNARY_PE(Sin);
HLIR_DCL_UNARY_PE(Sinh);
HLIR_DCL_UNARY_PE(Acos);
HLIR_DCL_UNARY_PE(Acosh);
HLIR_DCL_UNARY_PE(Asin);
HLIR_DCL_UNARY_PE(Asinh);
HLIR_DCL_UNARY_PE(Atan);
HLIR_DCL_UNARY_PE(Atanh);
HLIR_DCL_UNARY_PE(IsNan);
HLIR_DCL_UNARY_PE(Tanh);
HLIR_DCL_UNARY_PE(IsFinite);
HLIR_DCL_UNARY_PE(IsInf);

HLIR_DCL_UNARY_PE(Negative);
HLIR_DCL_UNARY_PE(Identity);
HLIR_DCL_UNARY_PE(LogicalNot);
HLIR_DCL_UNARY_PE(BitwiseNot);
HLIR_DCL_UNARY_PE(Sigmoid);
HLIR_DCL_UNARY_PE(Sign);
HLIR_DCL_UNARY_PE(Abs);
HLIR_DCL_UNARY_PE(Rsqrt);
HLIR_DCL_UNARY_PE(Cast);
HLIR_DCL_UNARY_PE(Clip);
HLIR_DCL_UNARY_PE(Reinterpret);
HLIR_DCL_UNARY_PE(ElementwiseSum);
HLIR_DCL_UNARY_PE(Full);
HLIR_DCL_UNARY_PE(FullLike);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
