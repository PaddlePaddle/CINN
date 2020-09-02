#pragma once
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {
/**
 * @brief Compute A && B with auto-broadcasting.
 *
 * @param A The first Tensor or Expr
 * @param B The second Tensor or Expr
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor or Expr.
 */
#define HLIR_DCL_BC_PE(name__)                                                                                     \
  ir::Tensor name__(const ir::Tensor& A,                                                                           \
                    const ir::Tensor& B,                                                                           \
                    const std::string& output_name = "T_" #name__ "_out",                                          \
                    const ir::Expr& axis           = ir::Expr());                                                            \
  ir::Tensor name__(const ir::Expr& A, const ir::Tensor& B, const std::string& output_name = "T_" #name__ "_out"); \
  ir::Tensor name__(const ir::Tensor& A, const ir::Expr& B, const std::string& output_name = "T_" #name__ "_out"); \
  ir::Expr name__(const ir::Expr& A, const ir::Expr& B);

//! Compute A + B with auto-broadcasting.
HLIR_DCL_BC_PE(Add);
//! Compute A - B with auto-broadcasting.
HLIR_DCL_BC_PE(Substract);
//! Compute A * B with auto-broadcasting.
HLIR_DCL_BC_PE(Multiply);
//! Compute A / B with auto-broadcasting.
HLIR_DCL_BC_PE(Divide);
//! Compute Floor(A / B) with auto-broadcasting.
HLIR_DCL_BC_PE(FloorDivide);
//! Compute A % B with auto-broadcasting.
HLIR_DCL_BC_PE(Mod);
//! Compute A - floor_div(A, B) * B with auto-broadcasting.
HLIR_DCL_BC_PE(FloorMod);
//! Compute Maximum(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Maximum);
//! Compute Minimum(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Minimum);
//! Compute Power(A, B) with auto-broadcasting.
HLIR_DCL_BC_PE(Power);
//! Compute A << B with auto-broadcasting.
HLIR_DCL_BC_PE(LeftShift);
//! Compute A >> B with auto-broadcasting.
HLIR_DCL_BC_PE(RightShift);
//! Compute A && B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicaAnd);
//! Compute A || B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalOr);
//! Compute A ^ B with auto-broadcasting.
HLIR_DCL_BC_PE(LogicalXOr);
//! Compute A & B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseAnd);
//! Compute A | B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseOr);
//! Compute A ^ B with auto-broadcasting.
HLIR_DCL_BC_PE(BitwiseXor);
//! Compute A > B with auto-broadcasting.
HLIR_DCL_BC_PE(Greater);
//! Compute A < B with auto-broadcasting.
HLIR_DCL_BC_PE(Less);
//! Compute A == B with auto-broadcasting.
HLIR_DCL_BC_PE(Equal);
//! Compute A != B with auto-broadcasting.
HLIR_DCL_BC_PE(NotEqual);
//! Compute A >= B with auto-broadcasting.
HLIR_DCL_BC_PE(GreaterEqual);
//! Compute A <= B with auto-broadcasting.
HLIR_DCL_BC_PE(LessEqual);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
