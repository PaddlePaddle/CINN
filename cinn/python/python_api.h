#pragma once

#include <pybind11/pybind11.h>

#include "cinn/hlir/instruction/instruction.h"

namespace python {

void BindHlirApi(pybind11::module* m);

}  // namespace python
