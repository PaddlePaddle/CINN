#include "cinn/common/ir_util.h"
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"
#include "cinn/utils/string.h"

namespace py = pybind11;

namespace cinn::pybind {

using optim::Simplify;

namespace {

void BindSimplify(py::module* m) {
  m->def(
      "simplify",
      [](const Expr& expr) -> Expr {
        auto copied = optim::IRCopy(expr);
        Simplify(&copied);
        return copied;
      },
      py::arg("expr"));

  m->def("ir_copy", py::overload_cast<Expr>(&optim::IRCopy));
}

}  // namespace

void BindOptim(py::module* m) { BindSimplify(m); }

}  // namespace cinn::pybind
