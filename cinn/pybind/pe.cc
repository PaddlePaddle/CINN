#include "cinn/common/target.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"
#include "cinn/utils/string.h"

namespace py = pybind11;

namespace cinn {
namespace pybind {

using common::Type;
using lang::Placeholder;
using py::arg;
using utils::GetStreamCnt;
using utils::StringFormat;

void BindPE(py::module* m) {
#define BIND_ELEMENTWISE(name__, fn__) m->def(#name__, &hlir::pe::fn__)
  BIND_ELEMENTWISE(exp, Exp);
  BIND_ELEMENTWISE(erf, Erf);
  BIND_ELEMENTWISE(sqrt, Sqrt);
  BIND_ELEMENTWISE(log, Log);
  BIND_ELEMENTWISE(log2, Log2);
  BIND_ELEMENTWISE(log10, Log10);
  BIND_ELEMENTWISE(floor, Floor);
  BIND_ELEMENTWISE(ceil, Ceil);
  BIND_ELEMENTWISE(round, Round);
  BIND_ELEMENTWISE(trunc, Trunc);
  BIND_ELEMENTWISE(cos, Cos);
  BIND_ELEMENTWISE(cosh, Cosh);
  BIND_ELEMENTWISE(tan, Tan);
  BIND_ELEMENTWISE(sin, Sin);
  BIND_ELEMENTWISE(sinh, Sinh);
  BIND_ELEMENTWISE(acos, Acos);
  BIND_ELEMENTWISE(acosh, Acosh);
  BIND_ELEMENTWISE(asin, Asin);
  BIND_ELEMENTWISE(asinh, Asinh);
  BIND_ELEMENTWISE(atan, Atan);
  BIND_ELEMENTWISE(atanh, Atanh);
  BIND_ELEMENTWISE(isnan, Isnan);
  BIND_ELEMENTWISE(tanh, Tanh);
  BIND_ELEMENTWISE(isfinite, Isfinite);
  BIND_ELEMENTWISE(isinf, Isinf);

  BIND_ELEMENTWISE(negative, Negative);
  BIND_ELEMENTWISE(identity, Identity);
  BIND_ELEMENTWISE(logical_not, LogicalNot);
  BIND_ELEMENTWISE(bitwise_not, BitwiseNot);
  BIND_ELEMENTWISE(sigmoid, Sigmoid);
  BIND_ELEMENTWISE(sign, Sign);
  BIND_ELEMENTWISE(abs, Abs);
  BIND_ELEMENTWISE(rsqrt, Rsqrt);
  BIND_ELEMENTWISE(cast, Cast);
  BIND_ELEMENTWISE(clip, Clip);
  BIND_ELEMENTWISE(reinterpret, Reinterpret);
  BIND_ELEMENTWISE(elementwise_sum, ElementwiseSum);
  BIND_ELEMENTWISE(full, Full);
  BIND_ELEMENTWISE(full_like, FullLike);
}

}  // namespace pybind
}  // namespace cinn
