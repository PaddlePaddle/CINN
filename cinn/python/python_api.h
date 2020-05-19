#pragma once

#include <pybind11/pybind11.h>

#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace python {

void BindHlirApi(pybind11::module* m);

PYBIND11_MODULE(cinn_core, m) {
  m.doc() = "CINN core";

  BindHlirApi(&m);
}

}  // namespace python
}  // namespace cinn
