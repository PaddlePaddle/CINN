#include "cinn/hlir/pe/elementwise.h"

#include <vector>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Expr;
using ir::Tensor;

#define HLIR_IMP_UNARY_PE(name__)                                                                         \
  Tensor name__(const Tensor& A, const std::string& output_name) {                                        \
    return Compute(                                                                                       \
        A->shape, [&](const std::vector<Expr>& indice) { return lang::name__(A(indice)); }, output_name); \
  }

HLIR_IMP_UNARY_PE(Exp);
HLIR_IMP_UNARY_PE(Erf);
HLIR_IMP_UNARY_PE(Sqrt);
HLIR_IMP_UNARY_PE(Log);
HLIR_IMP_UNARY_PE(Log2);
HLIR_IMP_UNARY_PE(Log10);
HLIR_IMP_UNARY_PE(Floor);
HLIR_IMP_UNARY_PE(Ceil);
HLIR_IMP_UNARY_PE(Round);
HLIR_IMP_UNARY_PE(Trunc);
HLIR_IMP_UNARY_PE(Cos);
HLIR_IMP_UNARY_PE(Cosh);
HLIR_IMP_UNARY_PE(Tan);
HLIR_IMP_UNARY_PE(Sin);
HLIR_IMP_UNARY_PE(Sinh);
HLIR_IMP_UNARY_PE(Acos);
HLIR_IMP_UNARY_PE(Acosh);
HLIR_IMP_UNARY_PE(Asin);
HLIR_IMP_UNARY_PE(Asinh);
HLIR_IMP_UNARY_PE(Atan);
HLIR_IMP_UNARY_PE(Atanh);
HLIR_IMP_UNARY_PE(IsNan);
HLIR_IMP_UNARY_PE(Tanh);
HLIR_IMP_UNARY_PE(IsFinite);
HLIR_IMP_UNARY_PE(IsInf);

HLIR_IMP_UNARY_PE(Negative);
HLIR_IMP_UNARY_PE(Identity);
HLIR_IMP_UNARY_PE(LogicalNot);
HLIR_IMP_UNARY_PE(BitwiseNot);
HLIR_IMP_UNARY_PE(Sigmoid);
HLIR_IMP_UNARY_PE(Sign);
HLIR_IMP_UNARY_PE(Abs);
HLIR_IMP_UNARY_PE(Rsqrt);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
