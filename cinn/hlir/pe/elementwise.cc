#include "cinn/hlir/pe/elementwise.h"

#include <string>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"

namespace cinn {
namespace hlir {
namespace pe {

using ir::Expr;
using ir::Tensor;
using lang::Compute;

#define HLIR_IMP_UNARY_PE(name__)                                                                          \
  std::vector<ir::Tensor> name__(const Tensor& A, const std::string& output_name) {                        \
    return {Compute(                                                                                       \
        A->shape, [=](const std::vector<Expr>& indice) { return lang::name__(A(indice)); }, output_name)}; \
  }

#define HLIR_MKL_IMP_UNARY_PE(name__, ex_name__)                                                     \
  std::vector<ir::Tensor> name__##MKL(const Tensor& A, const std::string& output_name) {             \
    CHECK(A->type().is_float()) << "type should be float or double but get " << A->type();           \
    std::string fn_name = "cinn_mkl_" #ex_name__ "_v_fp" + std::to_string(A->type().bits());         \
    auto call           = Compute(                                                                   \
        {Expr(1)}, [=]() -> Expr { return lang::CallExtern(fn_name, {A}); }, output_name); \
    auto out = call->TupleGet(0);                                                                    \
    out->WithBuffer(A->type());                                                                      \
    return {out, call};                                                                              \
  }

HLIR_MKL_IMP_UNARY_PE(Exp, exp);
HLIR_MKL_IMP_UNARY_PE(Erf, erf);
HLIR_MKL_IMP_UNARY_PE(Sqrt, sqrt);
HLIR_MKL_IMP_UNARY_PE(Log, log);
HLIR_MKL_IMP_UNARY_PE(Log2, log2);
HLIR_MKL_IMP_UNARY_PE(Log10, log10);
HLIR_MKL_IMP_UNARY_PE(Floor, floor);
HLIR_MKL_IMP_UNARY_PE(Ceil, ceil);
HLIR_MKL_IMP_UNARY_PE(Round, round);
HLIR_MKL_IMP_UNARY_PE(Tanh, tanh);
HLIR_MKL_IMP_UNARY_PE(Trunc, trunc);
HLIR_MKL_IMP_UNARY_PE(Cos, cos);
HLIR_MKL_IMP_UNARY_PE(Sin, sin);
HLIR_MKL_IMP_UNARY_PE(Cosh, cosh);
HLIR_MKL_IMP_UNARY_PE(Tan, tan);
HLIR_MKL_IMP_UNARY_PE(Sinh, sinh);
HLIR_MKL_IMP_UNARY_PE(Acos, acos);
HLIR_MKL_IMP_UNARY_PE(Acosh, acosh);
HLIR_MKL_IMP_UNARY_PE(Asin, asin);
HLIR_MKL_IMP_UNARY_PE(Asinh, asinh);
HLIR_MKL_IMP_UNARY_PE(Atan, atan);
HLIR_MKL_IMP_UNARY_PE(Atanh, atanh);

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
