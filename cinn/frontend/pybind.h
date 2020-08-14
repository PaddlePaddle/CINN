#pragma once
/**
 * \file This file contains the python binding for the frontend APIs.
 */
#include <pybind11/pybind11.h>

#include "cinn/common/type.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

void BindSyntax(pybind11::module *m);

}  // namespace frontend
}  // namespace cinn
