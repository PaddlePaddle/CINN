#include "cinn/common/target.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/reduction.h"
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
#define BIND_UNARY(name__, fn__) m->def(#name__, &hlir::pe::fn__, py::arg("x"), py::arg("out") = "T_" #name__ "_out")
  BIND_UNARY(exp, Exp);
  BIND_UNARY(erf, Erf);
  BIND_UNARY(sqrt, Sqrt);
  BIND_UNARY(log, Log);
  BIND_UNARY(log2, Log2);
  BIND_UNARY(log10, Log10);
  BIND_UNARY(floor, Floor);
  BIND_UNARY(ceil, Ceil);
  BIND_UNARY(round, Round);
  BIND_UNARY(trunc, Trunc);
  BIND_UNARY(cos, Cos);
  BIND_UNARY(cosh, Cosh);
  BIND_UNARY(tan, Tan);
  BIND_UNARY(sin, Sin);
  BIND_UNARY(sinh, Sinh);
  BIND_UNARY(acos, Acos);
  BIND_UNARY(acosh, Acosh);
  BIND_UNARY(asin, Asin);
  BIND_UNARY(asinh, Asinh);
  BIND_UNARY(atan, Atan);
  BIND_UNARY(atanh, Atanh);
  BIND_UNARY(isnan, Isnan);
  BIND_UNARY(tanh, Tanh);
  BIND_UNARY(isfinite, Isfinite);
  BIND_UNARY(isinf, Isinf);

  BIND_UNARY(negative, Negative);
  BIND_UNARY(identity, Identity);
  BIND_UNARY(logical_not, LogicalNot);
  BIND_UNARY(bitwise_not, BitwiseNot);
  BIND_UNARY(sigmoid, Sigmoid);
  BIND_UNARY(sign, Sign);
  BIND_UNARY(abs, Abs);
  BIND_UNARY(rsqrt, Rsqrt);

#define BIND_REDUCE(name__, fn__)      \
  m->def(#name__,                      \
         &hlir::pe::fn__,              \
         py::arg("x"),                 \
         py::arg("stages"),            \
         py::arg("axes"),              \
         py::arg("keep_dims") = false, \
         py::arg("initial"),           \
         py::arg("out") = "T_" #name__ "_out")
  BIND_REDUCE(sum, Sum);
  BIND_REDUCE(prod, Prod);

#define BIND_REDUCE1(name__, fn__)     \
  m->def(#name__,                      \
         &hlir::pe::fn__,              \
         py::arg("x"),                 \
         py::arg("stages"),            \
         py::arg("axes"),              \
         py::arg("keep_dims") = false, \
         py::arg("out")       = "T_" #name__ "_out")
  BIND_REDUCE1(max, Max);
  BIND_REDUCE1(min, Min);
}

}  // namespace pybind
}  // namespace cinn
